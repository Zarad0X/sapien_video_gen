import sapien.core as sapien
import numpy as np
import cv2
import json
import os
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional


class PartNetVideoRenderer:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, background_color=(1.0, 1.0, 1.0)):
        """
        Initialize PartNet Video Renderer.
        
        Args:
            width: Image width
            height: Image height 
            fps: Frames per second for video output
            background_color: 背景颜色 (r,g,b) 0~1 浮点
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize SAPIEN
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        # 设置背景色（如果API支持）
        if hasattr(self.renderer, 'set_clear_color'):
            try:
                self.renderer.set_clear_color(background_color)
            except Exception:
                pass
        self._background_color = background_color
        
        # Create scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 100.0)
        
        # 删除原先添加的大平面，改为后处理填充背景色
        
        # Setup lighting
        self._setup_lighting()
        
        # Setup camera
        self._setup_camera()
        
        # Storage for frames and camera parameters
        self.rgb_frames = []
        self.depth_frames = []
        self.camera_params = []
        
    def _setup_lighting(self):
        """Setup scene lighting."""
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([2, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([2, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-2, 0, 2], [1, 1, 1], shadow=True)
        
    def _setup_camera(self):
        """Setup camera with intrinsic parameters."""
        near, far = 0.1, 100
        self.camera = self.scene.add_camera(
            name="camera",
            width=self.width,
            height=self.height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        
        # Create camera mount for easy pose control
        self.camera_mount = self.scene.create_actor_builder().build_kinematic()
        self.camera.set_parent(parent=self.camera_mount, keep_pose=False)
        
        # Store intrinsic matrix
        self.intrinsic_matrix = self.camera.get_intrinsic_matrix()
        print(f"Camera intrinsic matrix:\n{self.intrinsic_matrix}")
        
        
    def load_partnet_object(self, urdf_path: str, scale: float = 1.0) -> sapien.Articulation:
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = scale  # 关键：设置缩放
        asset = loader.load_kinematic(urdf_path)
        if not asset:
            raise ValueError(f"Failed to load URDF from {urdf_path}")
        print(f"Loaded object: {urdf_path} (scale={scale})")
        return asset
        
    def generate_circular_trajectory(self, center: np.ndarray, radius: float, 
                                   height: float, n_frames: int, 
                                   full_rotation: bool = True) -> List[sapien.Pose]:
        
        poses = []
        angle_step = (2 * np.pi if full_rotation else np.pi) / n_frames
        
        for i in range(n_frames):
            angle = i * angle_step
            
            # Camera position
            cam_pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # Look at center
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            # Up vector
            up = np.array([0, 0, 1])
            
            # Right vector  
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            # Recompute up to ensure orthogonality
            up = np.cross(right, forward)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack([forward, -right, up])
            
            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = cam_pos
            
            pose = sapien.Pose.from_transformation_matrix(transform)
            poses.append(pose)
            
        return poses
        
    
    
    def generate_sphere_spiral_trajectory(self, center: np.ndarray, radius: float,
                                         start_elevation: float, end_elevation: float,
                                         rotations: float, n_frames: int) -> List[sapien.Pose]:
        
        poses = []
        
        # Convert elevation angles from degrees to radians
        start_elev_rad = np.radians(start_elevation)
        end_elev_rad = np.radians(end_elevation)
        
        for i in range(n_frames):
            t = i / (n_frames - 1) if n_frames > 1 else 0  # 0 to 1
            
            # Interpolate elevation angle
            elevation = start_elev_rad + t * (end_elev_rad - start_elev_rad)
            
            # Calculate azimuth angle (horizontal rotation)
            azimuth = t * rotations * 2 * np.pi
            
            # Convert spherical coordinates to Cartesian
            # elevation: angle from horizontal plane (0 = horizontal, π/2 = up, -π/2 = down)
            # azimuth: angle around vertical axis
            x = radius * np.cos(elevation) * np.cos(azimuth)
            y = radius * np.cos(elevation) * np.sin(azimuth)
            z = radius * np.sin(elevation)
            
            cam_pos = center + np.array([x, y, z])
            
            # Camera always looks at center
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            # Calculate up vector (tangent to sphere surface, pointing generally upward)
            # For a sphere, the up vector is perpendicular to the radial direction
            radial = cam_pos - center
            radial = radial / np.linalg.norm(radial)
            
            # Use world up as reference
            world_up = np.array([0, 0, 1])
            
            # Calculate right vector
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:  # Handle singularity when looking straight up/down
                # Use a different reference vector
                right = np.cross(forward, np.array([1, 0, 0]))
            right = right / np.linalg.norm(right)
            
            # Recompute up vector to ensure orthogonality
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Create rotation matrix (SAPIEN camera convention: X=forward, Y=left, Z=up)
            rotation_matrix = np.column_stack([forward, -right, up])
            
            # Create transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = cam_pos
            
            pose = sapien.Pose.from_transformation_matrix(transform)
            poses.append(pose)
            
        return poses
        
    def capture_frame(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Capture RGB and depth frame with camera parameters.
        
        Returns:
            RGB image, depth image, camera parameters dict
        """
        # Update scene and take picture
        self.scene.step()
        self.scene.update_render()
        self.camera.take_picture()
        
        # Get RGB
        rgba = self.camera.get_float_texture('Color')
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        
        # Get depth using position-based depth
        position = self.camera.get_float_texture('Position')
        depth = -position[..., 2]  
        
        # 用深度无效区域 (depth<=0) 作为背景掩码，填充背景颜色
        bg_mask = depth <= 0
        if np.any(bg_mask):
            bg_color_255 = (np.array(self._background_color) * 255).astype(np.uint8)
            rgb[bg_mask] = bg_color_255
        
        # 获取相机外参
        model_matrix = self.camera.get_model_matrix()
        camera_pose = self.camera_mount.get_pose()
        camera_params = {
            'model_matrix': model_matrix.tolist(),
            'camera_pose': {
                'position': camera_pose.p.tolist(),
                'quaternion': camera_pose.q.tolist()
            }
        }
        
        return rgb, depth, camera_params
        

        
    def create_videos(self, output_dir: str = "output", 
                     rgb_video_name: str = "rgb_video.mp4",
                     depth_video_name: str = "depth_video.mp4") -> None:
      
        if not self.rgb_frames or not self.depth_frames:
            print("No frames to create video. Render sequence first.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create RGB video
        rgb_path = f"{output_dir}/{rgb_video_name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rgb_writer = cv2.VideoWriter(rgb_path, fourcc, self.fps, (self.width, self.height))
        
        for rgb_frame in self.rgb_frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            rgb_writer.write(bgr_frame)
        rgb_writer.release()
        
        # Create depth video (colorized)
        depth_path = f"{output_dir}/{depth_video_name}"
        depth_writer = cv2.VideoWriter(depth_path, fourcc, self.fps, (self.width, self.height), True)  # isColor=True
        
        # Import matplotlib for colormap
        import matplotlib.cm as cm
        
        # Normalize depth across all frames for consistent visualization (excluding zeros)
        all_valid_depths = []
        for d in self.depth_frames:
            valid_depths = d[d > 0]
            if len(valid_depths) > 0:
                all_valid_depths.extend(valid_depths.flatten())
        
        if len(all_valid_depths) > 0:
            depth_min, depth_max = np.min(all_valid_depths), np.max(all_valid_depths)
        else:
            depth_min, depth_max = 0, 1
        
        for depth_frame in self.depth_frames:
            # Normalize depth excluding zeros
            depth_normalized = np.zeros_like(depth_frame)
            mask = depth_frame > 0
            if np.any(mask):
                depth_normalized[mask] = (depth_frame[mask] - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Apply colormap (viridis)
            colormap = cm.viridis
            depth_colored = colormap(depth_normalized)
            
            # Set zero areas to background color (转换到0-1范围)
            bg = list(self._background_color) + [1]
            depth_colored[~mask] = bg
            
            # Convert to BGR format for OpenCV (8-bit)
            depth_img_bgr = (depth_colored[:, :, :3] * 255).astype(np.uint8)
            depth_img_bgr = cv2.cvtColor(depth_img_bgr, cv2.COLOR_RGB2BGR)
            
            depth_writer.write(depth_img_bgr)
        depth_writer.release()
        
        print(f"Videos saved:")
        print(f"  RGB: {rgb_path}")
        print(f"  Depth: {depth_path}")


