"""
PartNet-Mobility Dataset Video Rendering with Camera Parameters

This script renders RGB and depth videos of PartNet-Mobility objects with controllable camera trajectories.
It also saves camera extrinsic parameters for each frame.

Features:
- Load PartNet-Mobility URDF objects
- Generate smooth camera trajectories (circular, spiral, etc.)
- Render RGB and depth sequences
- Save camera extrinsic parameters
- Export as videos or image sequences
"""

import sapien.core as sapien
from sapien.sensor import ActiveLightSensor
import numpy as np
import cv2
import json
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class PartNetVideoRenderer:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, use_realistic_depth: bool = False):
        """
        Initialize the renderer for PartNet-Mobility objects.
        
        Args:
            width: Image width
            height: Image height  
            fps: Frames per second for video output
            use_realistic_depth: Whether to use ActiveLightSensor for realistic depth (CPU-based)
                                Note: Set to False for more reliable traditional depth
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.use_realistic_depth = use_realistic_depth
        
        # Initialize SAPIEN
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        # Create scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 100.0)
        
        # Setup lighting
        self._setup_lighting()
        
        # Setup camera
        self._setup_camera()
        
        # Setup depth sensor if enabled
        if self.use_realistic_depth:
            self._setup_depth_sensor()
        
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
        
    def _setup_depth_sensor(self):
        """Setup ActiveLightSensor for realistic depth generation on CPU."""
        try:
            # 直接创建ActiveLightSensor，使用预设的sensor_type
            self.depth_sensor = ActiveLightSensor(
                sensor_name='depth_sensor',
                renderer=self.renderer,
                scene=self.scene,
                sensor_type='d415',  # 使用新的预设配置名称
                rgb_resolution=(self.width, self.height),
                ir_resolution=(self.width, self.height)
            )
            print(f"ActiveLightSensor initialized with resolution {self.width}x{self.height} (CPU-based)")
        except Exception as e:
            print(f"Warning: Failed to initialize ActiveLightSensor: {e}")
            print("Falling back to traditional depth rendering")
            self.use_realistic_depth = False
        
    def load_partnet_object(self, urdf_path: str) -> sapien.Articulation:
        """
        Load a PartNet-Mobility object from URDF file.
        
        Args:
            urdf_path: Path to the URDF file
            
        Returns:
            Loaded articulation object
        """
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        
        # Load as kinematic articulation (can also use load() for dynamic)
        asset = loader.load_kinematic(urdf_path)
        if not asset:
            raise ValueError(f"Failed to load URDF from {urdf_path}")
            
        print(f"Loaded object: {urdf_path}")
        return asset
        
    def generate_circular_trajectory(self, center: np.ndarray, radius: float, 
                                   height: float, n_frames: int, 
                                   full_rotation: bool = True) -> List[sapien.Pose]:
        """
        Generate circular camera trajectory around the object.
        
        Args:
            center: Center point to orbit around
            radius: Orbit radius
            height: Camera height
            n_frames: Number of frames
            full_rotation: Whether to complete full 360° rotation
            
        Returns:
            List of camera poses
        """
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
        
    def generate_spiral_trajectory(self, center: np.ndarray, radius_range: Tuple[float, float],
                                 height_range: Tuple[float, float], n_frames: int) -> List[sapien.Pose]:
        """
        Generate spiral camera trajectory.
        
        Args:
            center: Center point
            radius_range: (min_radius, max_radius)
            height_range: (min_height, max_height)
            n_frames: Number of frames
            
        Returns:
            List of camera poses
        """
        poses = []
        
        for i in range(n_frames):
            t = i / (n_frames - 1)  # 0 to 1
            angle = t * 4 * np.pi  # Two full rotations
            
            # Interpolate radius and height
            radius = radius_range[0] + t * (radius_range[1] - radius_range[0])
            height = height_range[0] + t * (height_range[1] - height_range[0])
            
            # Camera position
            cam_pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # Look at center
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            rotation_matrix = np.column_stack([forward, -right, up])
            
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = cam_pos
            
            pose = sapien.Pose.from_transformation_matrix(transform)
            poses.append(pose)
            
        return poses
    
    def generate_sphere_spiral_trajectory(self, center: np.ndarray, radius: float,
                                         start_elevation: float, end_elevation: float,
                                         rotations: float, n_frames: int) -> List[sapien.Pose]:
        """
        Generate spherical spiral camera trajectory that moves from top to bottom (or vice versa)
        while rotating around the center, with camera always looking at the center.
        
        Args:
            center: Center point to orbit around
            radius: Sphere radius (distance from center)
            start_elevation: Starting elevation angle in degrees (-90 to 90, where 90 is top, -90 is bottom)
            end_elevation: Ending elevation angle in degrees
            rotations: Number of complete rotations around the sphere
            n_frames: Number of frames
            
        Returns:
            List of camera poses
        """
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
        
        # 根据模式获取深度
        if self.use_realistic_depth and hasattr(self, 'depth_sensor'):
            print("Using ActiveLightSensor for depth")
            # 使用ActiveLightSensor获得真实深度图
            # 注意：ActiveLightSensor可能不需要compute_depth()步骤
            sensor_pose = self.camera_mount.get_pose()
            self.depth_sensor.set_pose(sensor_pose)
            self.scene.update_render()
            self.depth_sensor.take_picture()
            depth = self.depth_sensor.get_depth()
        
        else:
            # 使用position-based深度
            position = self.camera.get_float_texture('Position')
            depth = -position[..., 2]  # OpenGL convention
        
        # Get camera extrinsic parameters
        model_matrix = self.camera.get_model_matrix()
        camera_pose = self.camera_mount.get_pose()
        
        camera_params = {
            'model_matrix': model_matrix.tolist(),
            'camera_pose': {
                'position': camera_pose.p.tolist(),
                'quaternion': camera_pose.q.tolist()  # [w, x, y, z]
            }
        }
        
        return rgb, depth, camera_params
        
    def render_sequence(self, camera_poses: List[sapien.Pose], 
                       save_frames: bool = True, output_dir: str = "output") -> None:
        """
        Render a sequence of frames with given camera poses.
        
        Args:
            camera_poses: List of camera poses
            save_frames: Whether to save individual frames
            output_dir: Output directory
        """

        # Create output directory
        if save_frames:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/rgb", exist_ok=True)
            os.makedirs(f"{output_dir}/depth", exist_ok=True)
            
        self.rgb_frames = []
        self.depth_frames = []
        self.camera_params = []
        
        print(f"Rendering {len(camera_poses)} frames...")
        
        for i, pose in enumerate(camera_poses):
            # Set camera pose
            self.camera_mount.set_pose(pose)
            
            # 如果使用ActiveLightSensor，也需要设置其pose
            if self.use_realistic_depth and hasattr(self, 'depth_sensor'):
                self.depth_sensor.set_pose(pose)
            
            # Capture frame
            rgb, depth, params = self.capture_frame()
            
            # Store frames
            self.rgb_frames.append(rgb)
            self.depth_frames.append(depth)
            self.camera_params.append(params)
            
            # Save individual frames if requested
            if save_frames:
                # Save RGB
                rgb_pil = Image.fromarray(rgb)
                rgb_pil.save(f"{output_dir}/rgb/frame_{i:06d}.png")
                
                # Save depth (improved normalization for visualization)
                # 先检查深度值的有效性
                valid_depth = depth.copy()
                
                # 处理ActiveLightSensor可能产生的异常值
                if self.use_realistic_depth:
                    # 将无效值（nan, inf, 负值）替换为最大有效值
                    finite_mask = np.isfinite(valid_depth) & (valid_depth > 0)
                    if np.any(finite_mask):
                        max_valid_depth = np.max(valid_depth[finite_mask])
                        valid_depth = np.where(finite_mask, valid_depth, max_valid_depth)
                    else:
                        # 如果没有有效深度值，使用相机的远平面
                        valid_depth = np.full_like(depth, 100.0)
                
                # 归一化深度图用于可视化
                depth_min, depth_max = valid_depth.min(), valid_depth.max()
                if depth_max > depth_min:
                    depth_normalized = (valid_depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(valid_depth)
                
                depth_img = (depth_normalized * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img)
                depth_pil.save(f"{output_dir}/depth/frame_{i:06d}.png")
                
                # Save raw depth as numpy array
                np.save(f"{output_dir}/depth/frame_{i:06d}.npy", depth)
                
            if (i + 1) % 10 == 0:
                print(f"Rendered {i + 1}/{len(camera_poses)} frames")
                
        # Save camera parameters
        if save_frames:
            # 保存相机外参到JSON文件
            with open(f"{output_dir}/camera_params.json", 'w') as f:
                json.dump(self.camera_params, f, indent=2)
            
            # 保存相机内参到txt文件
            np.savetxt(f"{output_dir}/cam_K.txt", self.intrinsic_matrix, fmt='%.6f')
                
        print("Rendering complete!")
        
    def create_videos(self, output_dir: str = "output", 
                     rgb_video_name: str = "rgb_video.mp4",
                     depth_video_name: str = "depth_video.mp4") -> None:
        """
        Create MP4 videos from rendered frames.
        
        Args:
            output_dir: Output directory
            rgb_video_name: RGB video filename
            depth_video_name: Depth video filename
        """
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
            
            # Set zero areas to black
            depth_colored[~mask] = [0, 0, 0, 1]
            
            # Convert to BGR format for OpenCV (8-bit)
            depth_img_bgr = (depth_colored[:, :, :3] * 255).astype(np.uint8)
            depth_img_bgr = cv2.cvtColor(depth_img_bgr, cv2.COLOR_RGB2BGR)
            
            depth_writer.write(depth_img_bgr)
        depth_writer.release()
        
        print(f"Videos saved:")
        print(f"  RGB: {rgb_path}")
        print(f"  Depth: {depth_path}")


def main():
    """Example usage of the PartNet video renderer."""
    
    # Initialize renderer with traditional depth (more reliable)
    renderer = PartNetVideoRenderer(width=640, height=480, fps=30, use_realistic_depth=True)
    
    # Load PartNet-Mobility object
    # Replace with your actual URDF path
    urdf_path = "../assets/179/mobility.urdf"
    
    try:
        asset = renderer.load_partnet_object(urdf_path)
        
        # Generate camera trajectory
        # Circular trajectory around the object
        center = np.array([0, 0, 0.5])  # Adjust based on your object
        radius = 2.0
        height = 1.5
        n_frames = 120  # 4 seconds at 30 fps
        
        poses = renderer.generate_circular_trajectory(
            center=center, 
            radius=radius, 
            height=height, 
            n_frames=n_frames,
            full_rotation=True
        )
        
        # Alternative: spiral trajectory
        # poses = renderer.generate_spiral_trajectory(
        #     center=center,
        #     radius_range=(1.0, 3.0),
        #     height_range=(0.5, 2.5), 
        #     n_frames=n_frames
        # )
        
        # Alternative: spherical spiral trajectory
        # poses = renderer.generate_sphere_spiral_trajectory(
        #     center=center,
        #     radius=2.0,
        #     start_elevation=90,
        #     end_elevation=-90,
        #     rotations=2,
        #     n_frames=n_frames
        # )
        
        # Render sequence
        output_dir = "partnet_video_output"
        renderer.render_sequence(poses, save_frames=True, output_dir=output_dir)
        
        # Create videos
        renderer.create_videos(output_dir)
        
        print(f"\nOutput saved to: {output_dir}")
        print("Contents:")
        print("  - rgb/: Individual RGB frames")
        print("  - depth/: Individual depth frames (.png for visualization, .npy for raw data)")
        print("  - camera_params.json: Camera extrinsic parameters for each frame")
        print("  - rgb_video.mp4: RGB video")
        print("  - depth_video.mp4: Depth video")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the URDF path and ensure the assets are available.")


if __name__ == "__main__":
    main()
