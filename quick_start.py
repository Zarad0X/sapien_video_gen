#!/usr/bin/env python3
"""
Quick start script for PartNet-Mobility video rendering.

This script provides a simple command-line interface for quick rendering tasks.
"""

import argparse
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import sapien.core as sapien
    except ImportError:
        missing_deps.append("sapien")
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print(" All dependencies are installed!")
    return True


def simple_render(urdf_path: str, output_dir: str = "quick_render"):
    """Simple rendering with default settings."""
    if not check_dependencies():
        return False
    
    try:
        from render_partnet_video import PartNetVideoRenderer
        import numpy as np
        
        print(f" Starting simple render of: {urdf_path}")
        
        # Create renderer with default settings
        renderer = PartNetVideoRenderer(width=640, height=480, fps=30)
        
        # Load object
        asset = renderer.load_partnet_object(urdf_path)
        print(" Object loaded successfully")
        
        # Generate circular trajectory
        center = np.array([0, 0, 0.5])
        poses = renderer.generate_circular_trajectory(
            center=center,
            radius=2.0,
            height=1.5,
            n_frames=90,  # 3 seconds
            full_rotation=True
        )
        print("Camera trajectory generated")
        
        # Render
        renderer.render_sequence(poses, save_frames=True, output_dir=output_dir)
        print("Frames rendered")
        
        # Create videos
        renderer.create_videos(output_dir)
        
        # Create enhanced depth visualizations
        print("Creating enhanced depth visualizations...")
        renderer.create_enhanced_depth_videos(output_dir)
        renderer.save_depth_analysis(output_dir)
        
        print(" Videos created")
        print("Enhanced depth videos created (jet, plasma, viridis colormaps)")
        
        print(f" Rendering complete! Check output in: {output_dir}")
        print("Files include:")
        print("  - Standard RGB and depth videos")
        print("  - Enhanced depth videos with color (depth_jet.mp4, etc.)")
        print("  - Depth analysis plots and statistics")
        return True
        
    except Exception as e:
        print(f" Rendering failed: {e}")
        return False


def advanced_render(urdf_path: str, config: str = "standard", 
                   trajectory: str = "circular_medium", lighting: str = "standard", 
                   output_dir: str = None):
    """Advanced rendering with configurations."""
    if not check_dependencies():
        return False
    
    try:
        from advanced_renderer import ConfigurableRenderer
        
        print(f"Starting advanced render of: {urdf_path}")
        print(f"   Config: {config}, Trajectory: {trajectory}, Lighting: {lighting}")
        
        # Create configurable renderer
        renderer = ConfigurableRenderer()
        
        # Render with configurations
        result_dir = renderer.render_object(
            urdf_path=urdf_path,
            render_config=config,
            trajectory_config=trajectory,
            lighting_setup=lighting,
            output_dir=output_dir
        )
        
        print(f"Advanced rendering complete! Check output in: {result_dir}")
        return True
        
    except Exception as e:
        print(f" Advanced rendering failed: {e}")
        return False


def animated_render(urdf_path: str, config: str = "standard", 
                   trajectory: str = "circular_medium", lighting: str = "standard",
                   animation: str = "periodic", animation_config: str = "standard",
                   output_dir: str = None):
    """Animated rendering with joint motions."""
    if not check_dependencies():
        return False
    
    try:
        from advanced_renderer import ConfigurableRenderer
        
        print(f"Starting animated render of: {urdf_path}")
        print(f"   Config: {config}, Trajectory: {trajectory}, Lighting: {lighting}")
        print(f"   Animation: {animation}, Animation Config: {animation_config}")
        
        # Create configurable renderer
        renderer = ConfigurableRenderer()
        
        # Render with animations
        result_dir = renderer.render_animated_object(
            urdf_path=urdf_path,
            render_config=config,
            trajectory_config=trajectory,
            lighting_setup=lighting,
            animation_type=animation,
            animation_config=animation_config,
            output_dir=output_dir
        )
        
        print(f"Animated rendering complete! Check output in: {result_dir}")
        return True
        
    except Exception as e:
        print(f" Animated rendering failed: {e}")
        return False


def list_configs():
    """List available configurations."""
    try:
        from advanced_renderer import ConfigurableRenderer
        renderer = ConfigurableRenderer()
        renderer.list_configurations()
    except Exception as e:
        print(f" Could not load configurations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick start script for PartNet-Mobility video rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple render with default settings
  python quick_start.py simple /path/to/mobility.urdf
  
  # Advanced render with specific configuration
  python quick_start.py advanced /path/to/mobility.urdf --config high_quality --trajectory spiral_outward --lighting dramatic
  
  # Animated render with joint motions
  python quick_start.py animated /path/to/mobility.urdf --config high_quality --trajectory spiral_outward --lighting dramatic --animation periodic --animation-config energetic
  
  # List available configurations
  python quick_start.py list-configs
  
  # Check dependencies
  python quick_start.py check-deps
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simple render command
    simple_parser = subparsers.add_parser('simple', help='Simple render with default settings')
    simple_parser.add_argument('urdf_path', help='Path to URDF file')
    simple_parser.add_argument('--output', '-o', default='quick_render', 
                              help='Output directory (default: quick_render)')
    
    # Advanced render command
    advanced_parser = subparsers.add_parser('advanced', help='Advanced render with configurations')
    advanced_parser.add_argument('urdf_path', help='Path to URDF file')
    advanced_parser.add_argument('--config', '-c', default='standard',
                                help='Render configuration (default: standard)')
    advanced_parser.add_argument('--trajectory', '-t', default='circular_medium',
                                help='Camera trajectory (default: circular_medium)')
    advanced_parser.add_argument('--lighting', '-l', default='standard',
                                help='Lighting setup (default: standard)')
    advanced_parser.add_argument('--output', '-o', help='Output directory')
    
    # Animated render command
    animated_parser = subparsers.add_parser('animated', help='Animated render with joint motions')
    animated_parser.add_argument('urdf_path', help='Path to URDF file')
    animated_parser.add_argument('--config', '-c', default='standard',
                                help='Render configuration (default: standard)')
    animated_parser.add_argument('--trajectory', '-t', default='circular_medium',
                                help='Camera trajectory (default: circular_medium)')
    animated_parser.add_argument('--lighting', '-l', default='high_quality',
                                help='Lighting setup (default: high_quality)')
    animated_parser.add_argument('--animation', '-a', default='periodic',
                                help='Animation type (periodic, oscillating, sequential, large_motion)')
    animated_parser.add_argument('--animation-config', '--anim-config', default='standard',
                                help='Animation intensity (gentle, standard, energetic, extreme)')
    animated_parser.add_argument('--output', '-o', help='Output directory')
    
    # List configurations command
    subparsers.add_parser('list-configs', help='List available configurations')
    
    # Check dependencies command
    subparsers.add_parser('check-deps', help='Check if all dependencies are installed')
    
    args = parser.parse_args()
    
    if args.command == 'simple':
        if not os.path.exists(args.urdf_path):
            print(f" URDF file not found: {args.urdf_path}")
            sys.exit(1)
        
        success = simple_render(args.urdf_path, args.output)
        sys.exit(0 if success else 1)
        
    elif args.command == 'advanced':
        if not os.path.exists(args.urdf_path):
            print(f" URDF file not found: {args.urdf_path}")
            sys.exit(1)
        
        success = advanced_render(args.urdf_path, args.config, args.trajectory, args.lighting, args.output)
        sys.exit(0 if success else 1)
        
    elif args.command == 'animated':
        if not os.path.exists(args.urdf_path):
            print(f" URDF file not found: {args.urdf_path}")
            sys.exit(1)
        
        success = animated_render(args.urdf_path, args.config, args.trajectory, args.lighting, args.animation, args.animation_config, args.output)
        sys.exit(0 if success else 1)
        
    elif args.command == 'list-configs':
        list_configs()
        
    elif args.command == 'check-deps':
        success = check_dependencies()
        sys.exit(0 if success else 1)
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
