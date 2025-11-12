import sapien.core as sapien
import numpy as np
import cv2
import json
import os
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R


class PartNetVideoRenderer:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, background_color=(1.0, 1.0, 1.0)):
        """
        Initialize PartNet Video Renderer.
        
        Args:
            width: Image width
            height: Image height 
            fps: Frames per second for video output
            background_color: Background color (r,g,b) values in 0~1 float range
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize SAPIEN
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        # Set background color (if API supports it)
        if hasattr(self.renderer, 'set_clear_color'):
            try:
                self.renderer.set_clear_color(background_color)
            except Exception:
                pass
        self._background_color = background_color
        
        # Create scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 100.0)
        
        # Setup lighting
        self._setup_lighting()
        
        # Setup camera
        self._setup_camera()
        
        # Storage for frames and camera parameters
        self.rgb_frames = []
        self.depth_frames = []
        self.camera_params = []

        # visualization settings
        # axis length in meters for joint axes visualization
        self.joint_axis_length = 0.05
        
    def _setup_lighting(self):
        """Setup scene lighting."""
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([2, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([2, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-2, 0, 2], [1, 1, 1], shadow=True)
        
    def _setup_camera(self):
        """Setup camera with intrinsic parameters."""
        near, far = 0.01, 100
        self.camera = self.scene.add_camera(
            name="camera",
            width=self.width,
            height=self.height,
            fovy=np.deg2rad(58),
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
        loader.scale = scale  # Important: set scaling
        asset = loader.load_kinematic(urdf_path)
        self.asset = asset
        if not asset:
            raise ValueError(f"Failed to load URDF from {urdf_path}")
        print(f"Loaded object: {urdf_path} (scale={scale})")
        return asset
        
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
        
        # Use invalid depth areas (depth<=0) as background mask and fill with background color
        bg_mask = depth <= 0
        if np.any(bg_mask):
            bg_color_255 = (np.array(self._background_color) * 255).astype(np.uint8)
            rgb[bg_mask] = bg_color_255
        
        # Get camera extrinsic parameters
        model_matrix = self.camera.get_model_matrix()
        camera_pose = self.camera_mount.get_pose()
        camera_params = {
            'model_matrix': model_matrix.tolist(),
            'camera_pose': {
                'position': camera_pose.p.tolist(),
                'quaternion': camera_pose.q.tolist()
            }
        }
        # Draw joint/link axes if an articulation/asset is loaded
        art = None
        if hasattr(self, 'articulation') and self.articulation:
            art = self.articulation
        elif hasattr(self, 'asset') and self.asset:
            art = self.asset

        if art is not None:
            # prepare camera rotation and translation (world -> camera)
            cam_q = camera_pose.q  # w,x,y,z
            cam_rot = R.from_quat(cam_q[[1, 2, 3, 0]])
            cam_trans = np.array(camera_pose.p)

            K = np.array(self.intrinsic_matrix)

            # iterate links if available, fallback to joints if not
            links = []
            if hasattr(art, 'get_links'):
                try:
                    links = art.get_links()
                except Exception:
                    links = []

            # If no links, try to use joints and draw at joint child link poses
            if not links and hasattr(art, 'get_joints'):
                try:
                    joints = art.get_joints()
                    # some joints have .get_child_link() or .child_link; try to extract poses
                    for j in joints:
                        try:
                            if hasattr(j, 'get_child_link'):
                                links.append(j.get_child_link())
                            elif hasattr(j, 'child_link'):
                                links.append(j.child_link)
                        except Exception:
                            continue
                except Exception:
                    links = []

            for link in links:
                try:
                    link_pose = link.get_pose()
                except Exception:
                    # some link representations might store pose differently
                    continue

                p_world = np.array(link_pose.p)
                # link rotation (local -> world)
                link_q = link_pose.q
                link_rot = R.from_quat(link_q[[1, 2, 3, 0]])

                origin = p_world
                axes_ends = [
                    origin + link_rot.apply([self.joint_axis_length, 0.0, 0.0]),
                    origin + link_rot.apply([0.0, self.joint_axis_length, 0.0]),
                    origin + link_rot.apply([0.0, 0.0, self.joint_axis_length]),
                ]

                # project origin and endpoints into image
                def project_point(pw: np.ndarray):
                    # point in camera coordinate: p_cam = R_cam_world.inv() * (Pw - cam_trans)
                    p_cam = cam_rot.inv().apply(pw - cam_trans)
                    # In SAPIEN camera coordinates forward is -Z (position[...,2] was negated for depth)
                    z_cam = -p_cam[2]
                    if z_cam <= 0:
                        return None
                    u = (K[0, 0] * p_cam[0] / -p_cam[2]) + K[0, 2]
                    v = (K[1, 1] * p_cam[1] / -p_cam[2]) + K[1, 2]
                    return int(round(u)), int(round(v))

                origin_px = project_point(origin)
                ends_px = [project_point(e) for e in axes_ends]

                if origin_px is None:
                    continue

                # draw axes: X=red, Y=green, Z=blue
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                for end_px, color in zip(ends_px, colors):
                    if end_px is None:
                        continue
                    cv2.line(rgb, origin_px, end_px, color, thickness=2, lineType=cv2.LINE_AA)
                # small circle at origin
                cv2.circle(rgb, origin_px, radius=3, color=(255, 255, 255), thickness=-1)

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
            
            # Set zero areas to background color (convert to 0-1 range)
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


