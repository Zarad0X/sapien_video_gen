import argparse
import sys
import os
import json
from pathlib import Path
from typing import List
from animated_renderer import AnimatedRenderer
from camera_traj import generate_circular_trajectory, generate_sphere_spiral_trajectory
import numpy as np
import sapien.core as sapien


def custom_animated_render(urdf_path: str, config: str = "high_quality", 
                          trajectory: str = "sphere_spiral_custom", lighting: str = "standard",
                          output_dir: str = None, static: bool = False, scale: float = 1.0,
                          rotation_angle: float = 0.0, speed: float = 1.0,
                          animation_config: dict = None):

    
    print(f"Starting custom animated render of: {urdf_path}")
    print(f"   Config: {config}, Trajectory: {trajectory}, Lighting: {lighting}")
    print("   Using custom per-joint animation modes")
    if static:
        print("   Joint animation: static mode")
    else:
        print(f"   Joint animation speed x{speed}")
    
    # Create animated renderer
    if config == "high_quality":
        renderer = AnimatedRenderer(width=1280, height=720, fps=30)
    elif config == "ultra_high":
        renderer = AnimatedRenderer(width=1920, height=1080, fps=60)
    elif config == "medium":
        renderer = AnimatedRenderer(width=640, height=480, fps=30)
    
    # Load object
    asset = renderer.load_partnet_object(urdf_path, scale=scale)
    print("Object loaded successfully")
    
    # Rotate object if rotation angle is specified
    if rotation_angle != 0.0:
        # Get current pose and rotate around world Z-axis
        current_pose = asset.get_root_pose()
        angle_rad = np.deg2rad(rotation_angle)
        # SAPIEN uses wxyz quaternion format: [w, x, y, z]
        cos_half = np.cos(angle_rad / 2)
        sin_half = np.sin(angle_rad / 2)
        # For Z-axis rotation: w=cos(θ/2), x=0, y=0, z=sin(θ/2)
        rotation_quat = [cos_half, 0, 0, sin_half]  # [w, x, y, z] for Z-axis rotation
        # Apply rotation to current orientation
        rotation_pose = sapien.Pose(p=current_pose.p, q=rotation_quat)
        new_pose = rotation_pose * current_pose  # Rotate the current pose
        asset.set_root_pose(new_pose)
        print(f"Object rotated by {rotation_angle} degrees around world Z-axis")
    
    # Generate camera trajectory around origin
    center = np.array([0, 0, 0])
    
    # Generate camera trajectory
    if trajectory == "circular_medium":
        poses = generate_circular_trajectory(
            center=center, radius=3.0, height=0, n_frames=180, full_rotation=True
        )
    elif trajectory == "sphere_spiral_custom":
        poses = generate_sphere_spiral_trajectory(
            center=center, radius=1, start_elevation=60, end_elevation=-60, 
            rotations=3, n_frames=300
        )
    else:
        poses = generate_circular_trajectory(
            center=center, radius=10, height=4, n_frames=180, full_rotation=True
        )
    
    print("Camera trajectory generated")

    # Create custom animations with automatic assignment for all joints
    if static:
        print("Using static mode - joints will not move")
        custom_animations = renderer.create_custom_animation(static_mode=True)
    else:
        print("Using animated mode - joints will move automatically")
        custom_animations = renderer.create_custom_animation(animation_config=animation_config, speed=speed)

    # Set output directory
    if output_dir is None:
        output_dir = "output"
    
    # Render with custom animations
    renderer.render_animated_sequence(
        poses,
        animations=custom_animations,
        output_dir=output_dir
    )
    
    # Create videos
    renderer.create_videos(output_dir, "rgb_video.mp4", "vis_video.mp4")

    return True
    


def main():
    parser = argparse.ArgumentParser(
        description="PartNet-Mobility Custom Animated Video Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main argument: URDF path (optional for utility commands)
    parser.add_argument('urdf_path', nargs='?', help='Path to URDF file')
    # Rendering options
    parser.add_argument('--config', '-c', default='high_quality',
                       help='Render configuration: standard, high_quality, ultra_high ')
    parser.add_argument('--trajectory', '-t', default='sphere_spiral_custom',
                       help='Camera trajectory: circular_medium, sphere_spiral_custom ')
    parser.add_argument('--lighting', '-l', default='high_quality',help='Lighting setup ')
    parser.add_argument('--output', '-o', help='output directory ')
    parser.add_argument('--static', '-s', action='store_true', 
                       help='Render with static joints (no animation)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for loaded object (default: 1.0)')
    parser.add_argument('--rotation', type=float, default=0.0,
                       help='Initial rotation angle in degrees around Z-axis (default: 0.0)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Global joint animation speed multiplier (>1 faster, <1 slower). Default: 1.0')
    parser.add_argument('--animation-config', type=str,default=None,
                       help='Animation configuration preset: path to JSON file')

    
    args = parser.parse_args()

    
    # Main rendering command
    if args.urdf_path:
        if not os.path.exists(args.urdf_path):
            print(f"URDF file not found: {args.urdf_path}")
            sys.exit(1)
        
        animation_config = None
        
        # Handle animation configuration
        if args.animation_config:
            
            with open(args.animation_config, 'r') as f:
                animation_config = json.load(f)
            print(f"Loaded animation config from: {args.animation_config}")

        success = custom_animated_render(args.urdf_path, args.config, args.trajectory, 
                                       args.lighting, args.output, args.static, args.scale, args.rotation, args.speed,
                                       animation_config=animation_config)
        sys.exit(0 if success else 1)
    else:
        # No URDF path provided and no utility command
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
