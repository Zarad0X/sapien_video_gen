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
import numpy as np
import cv2
import json
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class PartNetVideoRenderer:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the renderer for PartNet-Mobility objects.
        
        Args:
            width: Image width
            height: Image height  
            fps: Frames per second for video output
        """
        self.width = width
        self.height = height
        self.fps = fps
        
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
        
        # Get depth
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
            },
            'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            'width': self.width,
            'height': self.height
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
                
                # Save depth (normalized for visualization)
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_normalized * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img)
                depth_pil.save(f"{output_dir}/depth/frame_{i:06d}.png")
                
                # Save raw depth as numpy array
                np.save(f"{output_dir}/depth/frame_{i:06d}.npy", depth)
                
            if (i + 1) % 10 == 0:
                print(f"Rendered {i + 1}/{len(camera_poses)} frames")
                
        # Save camera parameters
        if save_frames:
            with open(f"{output_dir}/camera_params.json", 'w') as f:
                json.dump(self.camera_params, f, indent=2)
                
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
        
        # Create depth video (normalized)
        depth_path = f"{output_dir}/{depth_video_name}"
        depth_writer = cv2.VideoWriter(depth_path, fourcc, self.fps, (self.width, self.height), False)
        
        # Normalize depth across all frames for consistent visualization
        all_depths = np.concatenate([d.flatten() for d in self.depth_frames])
        depth_min, depth_max = all_depths.min(), all_depths.max()
        
        for depth_frame in self.depth_frames:
            depth_normalized = (depth_frame - depth_min) / (depth_max - depth_min + 1e-8)
            depth_img = (depth_normalized * 255).astype(np.uint8)
            depth_writer.write(depth_img)
        depth_writer.release()
        
        print(f"Videos saved:")
        print(f"  RGB: {rgb_path}")
        print(f"  Depth: {depth_path}")


def main():
    """Example usage of the PartNet video renderer."""
    
    # Initialize renderer
    renderer = PartNetVideoRenderer(width=640, height=480, fps=30)
    
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
