import json
import os
import numpy as np
from typing import List, Dict, Callable, Optional
from tqdm import tqdm

import sapien.core as sapien
from render_partnet_video import PartNetVideoRenderer


class AnimatedRenderer(PartNetVideoRenderer):
    """
    Extended renderer that supports joint animations during video rendering.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(width, height, fps)
        self.articulation = None
        self.joint_animations = {}
        
    def load_partnet_object(self, urdf_path: str, scale: float = 1.0) -> sapien.Articulation:
        """Load PartNet-Mobility object with scale, and store articulation reference."""
        self.articulation = super().load_partnet_object(urdf_path, scale=scale)
        self._analyze_joints()
        return self.articulation
    
    def _analyze_joints(self):
        """Analyze joint information"""
        if not self.articulation:
            return
            
        joints = self.articulation.get_joints()
        print(f"\n=== Joint Analysis ===")
        print(f"Total joints: {len(joints)}")
        
        movable_joints = []
        for i, joint in enumerate(joints):
            # Get joint type as string
            if hasattr(joint, 'type'):
                joint_type = str(joint.type).split('.')[-1].lower()  # Extract type name
            else:
                joint_type = "unknown"
                
            limits = joint.get_limits()
            print(f"Joint {i}: {joint.name}")
            print(f"  Type: {joint_type}")
            if len(limits) > 0 and len(limits[0]) == 2:
                print(f"  Limits: [{limits[0][0]:.3f}, {limits[0][1]:.3f}]")
                movable_joints.append(joint.name)
            else:
                print(f"  Limits: Unlimited")
        
        print(f"\n=== Movable Joints List ===")
        for joint_name in movable_joints:
            print(f"  - '{joint_name}'")
        
    
    def create_custom_animation(self, animation_config: Optional[Dict[str, Dict]] = None, 
                               auto_assign: bool = True, static_mode: bool = False,
                               speed: float = 1.0) -> Dict[str, Callable]:
        """
        Create joint animation configuration
        
        Args:
            animation_config: Manually specified joint animation configuration (optional)
            auto_assign: Whether to automatically assign default animations to unspecified joints
            static_mode: Whether to use static mode (joints don't move)
            speed: Global speed multiplier, >1 faster, <1 slower (default 1.0)
        
        Returns:
            Joint animation function dictionary
        """
        animations = {}
        
        # Get all movable joints
        movable_joints = self._get_movable_joints()
        
        # Static mode: all joints stay at middle position without movement
        if static_mode:
            for joint_name, joint_info in movable_joints.items():
                def make_static_func(joint_limits):
                    def static_func(t, frame_count):
                        # Return middle position of joint limits
                        return (joint_limits[0] + joint_limits[1]) / 2
                    return static_func
                animations[joint_name] = make_static_func(joint_info['limits'])
            return animations
        
        # If no configuration provided or auto-assignment needed, create default configuration
        if animation_config is None:
            animation_config = {}
        
        # Handle manually specified joint configurations (apply speed multiplier)
        for joint_name, config in animation_config.items():
            if joint_name in movable_joints:
                adj_config = dict(config)
                adj_config['frequency'] = config.get('frequency', 1.0) * speed
                animations[joint_name] = self._create_animation_function(adj_config, movable_joints[joint_name])
        
        # Automatically assign default animations to unspecified joints (apply speed multiplier)
        if auto_assign:
            assigned_joints = set(animation_config.keys())
            unassigned_joints = [name for name in movable_joints.keys() if name not in assigned_joints]
            
            for i, joint_name in enumerate(unassigned_joints):
                default_config = self._get_default_animation_config(i, movable_joints[joint_name])
                default_config['frequency'] = default_config.get('frequency', 1.0) * speed
                animations[joint_name] = self._create_animation_function(default_config, movable_joints[joint_name])
        
        return animations
    
    def _get_movable_joints(self) -> Dict[str, Dict]:
        """Get all movable joints and their information"""
        if not self.articulation:
            return {}
        
        movable_joints = {}
        joints = self.articulation.get_joints()
        
        for joint in joints:
            # Check if joint is movable
            if hasattr(joint, 'type'):
                joint_type_str = str(joint.type).lower()
                is_movable = 'revolute' in joint_type_str or 'prismatic' in joint_type_str
            else:
                limits = joint.get_limits()
                is_movable = len(limits) > 0 and len(limits[0]) == 2
            
            if is_movable and joint.name:
                limits = joint.get_limits()
                joint_info = {
                    'type': 'revolute' if 'revolute' in joint_type_str else 'prismatic',
                    'limits': limits[0] if len(limits) > 0 and len(limits[0]) == 2 else [0.0, 1.0]
                }
                movable_joints[joint.name] = joint_info
        
        return movable_joints
    
    def _get_default_animation_config(self, index: int, joint_info: Dict) -> Dict:
        """Generate default animation configuration for joint"""
        joint_type = joint_info['type']
        limits = joint_info['limits']
        
        # Predefined animation patterns
        animation_patterns = [
            {"type": "sine", "frequency": 1.0, "phase": 0.0},
            {"type": "linear", "frequency": 0.5, "phase": 0.0},
            {"type": "sine", "frequency": 1.5, "phase": np.pi/2},
            {"type": "sine", "frequency": 0.8, "phase": np.pi},
            {"type": "linear", "frequency": 0.7, "phase": np.pi/4},
            {"type": "sine", "frequency": 1.2, "phase": 3*np.pi/2}
        ]
        
        # Use animation patterns cyclically
        pattern = animation_patterns[index % len(animation_patterns)]
        
        # Adjust range based on joint type
        if joint_type == 'prismatic':
            # Moving joint: check if limits are infinite
            if np.isinf(limits[0]) or np.isinf(limits[1]):
                # Infinite limits: use default range
                animation_range = [-0.1, 0.1]  # Default movement range
            else:
                range_span = limits[1] - limits[0]
                safe_range = range_span * 0.8
                center = (limits[0] + limits[1]) / 2
                animation_range = [center - safe_range/2, center + safe_range/2]
        else:
            # Rotation joint: check if limits are infinite (continuous joint)
            if np.isinf(limits[0]) or np.isinf(limits[1]):
                # Infinite rotation: use full circle range
                animation_range = [0.0, 2 * np.pi]  # Full rotation from 0 to 2π
            else:
                range_span = limits[1] - limits[0]
                safe_range = range_span * 0.8
                center = (limits[0] + limits[1]) / 2
                animation_range = [center - safe_range/2, center + safe_range/2]
        
        return {
            "type": pattern["type"],
            "range": animation_range,
            "frequency": pattern["frequency"],
            "phase": pattern["phase"],
            "offset": 0.0
        }
    
    def _create_animation_function(self, config: Dict, joint_info: Dict) -> Callable:
        """Create animation function based on configuration"""
        anim_type = config.get("type", "sine")
        angle_range = config.get("range", joint_info['limits'])
        frequency = config.get("frequency", 1.0)
        phase = config.get("phase", 0.0)
        offset = config.get("offset", 0.0)
        
        if anim_type == "sine":
            def make_sine_func(min_angle, max_angle, freq, ph, off):
                def sine_func(t, frame_count):
                    center = (min_angle + max_angle) / 2 + off
                    amplitude = (max_angle - min_angle) / 2
                    return center + amplitude * np.sin(2 * np.pi * freq * t + ph)
                return sine_func
            return make_sine_func(angle_range[0], angle_range[1], frequency, phase, offset)
        
        elif anim_type == "linear":
            def make_linear_func(min_angle, max_angle, freq, ph, off):
                def linear_func(t, frame_count):
                    # Linear back-and-forth movement
                    cycle_t = (freq * t + ph / (2 * np.pi)) % 1.0
                    if cycle_t <= 0.5:
                        progress = cycle_t * 2  # 0 to 1
                    else:
                        progress = 2 - cycle_t * 2  # 1 to 0
                    return min_angle + (max_angle - min_angle) * progress + off
                return linear_func
            return make_linear_func(angle_range[0], angle_range[1], frequency, phase, offset)
        
        elif anim_type == "static":
            # Static function: joint stays at specified position without movement
            def make_static_func(min_angle, max_angle, off):
                def static_func(t, frame_count):
                    # Can choose to stay at middle position + offset, or specific position
                    return (min_angle + max_angle) / 2 + off
                return static_func
            return make_static_func(angle_range[0], angle_range[1], offset)
        
        else:
            # Default return static function (compatible with unknown types)
            def static_func(t, frame_count):
                return (angle_range[0] + angle_range[1]) / 2
            return static_func
    
    
    def animate_joints(self, t: float, frame_count: int):
        """
        Apply joint animations at specified time
        
        Args:
            t: Normalized time (0 to 1)
            frame_count: Total frame count
        """
        if not self.articulation or not self.joint_animations:
            return
        
        # Get current joint positions
        qpos = self.articulation.get_qpos().copy()
        joints_list = self.articulation.get_joints()
        
        # Create mapping from joint name to index
        joint_name_to_idx = {}
        movable_joint_idx = 0
        
        for i, joint in enumerate(joints_list):
            # Check if joint is movable
            if hasattr(joint, 'type'):
                joint_type_str = str(joint.type).lower()
                is_movable = 'revolute' in joint_type_str or 'prismatic' in joint_type_str
            else:
                # Backup check: see if it has limits
                limits = joint.get_limits()
                is_movable = len(limits) > 0 and len(limits[0]) == 2
            
            if is_movable and joint.name:
                joint_name_to_idx[joint.name] = movable_joint_idx
                movable_joint_idx += 1
        
        # Apply animations to each joint
        animated_joints = []
        for joint_name, animation_func in self.joint_animations.items():
            if joint_name in joint_name_to_idx:
                target_angle = animation_func(t, frame_count)
                joint_idx = joint_name_to_idx[joint_name]
                
                if joint_idx < len(qpos):
                    qpos[joint_idx] = target_angle
                    animated_joints.append(f"{joint_name}={target_angle:.3f}")
        
        # Set new joint positions
        if animated_joints:
            try:
                self.articulation.set_qpos(qpos)
                # Important: step simulation to update physics state
                self.scene.step()
            except Exception as e:
                print(f"Joint animation setting failed: {e}")
                print(f"  qpos length: {len(qpos)}, joint mapping: {joint_name_to_idx}")
    
    def render_animated_sequence(self, camera_poses: List[sapien.Pose], 
                               animations: Dict[str, Callable],
                               save_frames: bool = True, 
                               output_dir: str = "animated_output") -> None:
    
        # Set animations
        self.joint_animations = animations
        
        # Create output directory
        if save_frames:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/rgb", exist_ok=True)
            os.makedirs(f"{output_dir}/depth", exist_ok=True)  # Original depth data
            os.makedirs(f"{output_dir}/vis", exist_ok=True)    # Depth visualization
            
        self.rgb_frames = []
        self.depth_frames = []
        self.camera_params = []
        joint_states = []
        
        total_frames = len(camera_poses)
        print(f"Rendering {total_frames} animation frames...")

        # Create progress bar with tqdm
        pbar = tqdm(total=total_frames, desc="Rendering", unit="frame", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for i, pose in enumerate(camera_poses):
            # Calculate time progress
            t = i / (len(camera_poses) - 1) if len(camera_poses) > 1 else 0.0
            
            # Apply joint animations
            self.animate_joints(t, len(camera_poses))
            
            # Set camera pose
            self.camera_mount.set_pose(pose)
            
            # Update scene render state
            self.scene.update_render()
            
            # Record joint state
            joint_state = {
                'frame': i,
                'time': t,
                'qpos': self.articulation.get_qpos().tolist() if self.articulation else [],
                'qvel': self.articulation.get_qvel().tolist() if self.articulation else []
            }
            joint_states.append(joint_state)
            
            # Capture frame
            rgb, depth, params = self.capture_frame()
            
            # Store frames
            self.rgb_frames.append(rgb)
            self.depth_frames.append(depth)
            self.camera_params.append(params)
            
            # Save frames
            if save_frames:
                # Save RGB
                from PIL import Image
                rgb_pil = Image.fromarray(rgb)
                rgb_pil.save(f"{output_dir}/rgb/{i:05d}.png")
                
                # Save original depth data
                np.savez_compressed(f"{output_dir}/depth/{i:05d}.npz", depth=depth)
                
                # Save depth visualization to vis folder
                valid_depth = depth.copy()
                
                # Normalize depth map for visualization
                depth_min, depth_max = valid_depth.min(), valid_depth.max()
                if depth_max > depth_min:
                    depth_normalized = (valid_depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(valid_depth)
                
                # Generate colored depth map using color mapping
                import matplotlib.cm as cm
                colormap = cm.viridis  # Use viridis colormap
                depth_colored = colormap(depth_normalized)
                
                # Convert to RGB format (remove alpha channel) and convert to uint8
                depth_img_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img_rgb)
                depth_pil.save(f"{output_dir}/vis/{i:05d}.png")
                
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
                
        # Save camera parameters and joint states
        if save_frames:
            # Save camera extrinsic parameters to JSON file
            with open(f"{output_dir}/camera_params.json", 'w') as f:
                json.dump(self.camera_params, f, indent=2)
            
            # Save camera intrinsic parameters to txt file
            np.savetxt(f"{output_dir}/cam_K.txt", self.intrinsic_matrix, fmt='%.6f')
                
            # Save joint states
            with open(f"{output_dir}/joint_states.json", 'w') as f:
                json.dump(joint_states, f, indent=2)
                
        print("Animation rendering completed!")


