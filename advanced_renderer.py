"""
Advanced PartNet-Mobility renderer with configuration support.

This script provides a more user-friendly interface with predefined configurations
for different rendering scenarios.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from render_partnet_video import PartNetVideoRenderer
from animated_renderer import AnimatedRenderer
import sapien.core as sapien


class ConfigurableRenderer:
    """
    A wrapper around PartNetVideoRenderer with configuration file support.
    """
    
    def __init__(self, config_file: str = "render_config.json"):
        """
        Initialize with configuration file.
        
        Args:
            config_file: Path to the configuration JSON file
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.renderer = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_configurations(self):
        """Print available configurations."""
        print("=== Available Render Configurations ===")
        for name, config in self.config["render_configs"].items():
            print(f"  {name}: {config['description']}")
            print(f"    Resolution: {config['width']}x{config['height']}, FPS: {config['fps']}")
        
        print("\n=== Available Camera Trajectories ===")
        for name, traj in self.config["camera_trajectories"].items():
            print(f"  {name}: {traj['description']}")
            
        print("\n=== Available Lighting Setups ===")
        for name, lighting in self.config["lighting_setups"].items():
            # Simple descriptions for lighting setups
            descriptions = {
                "standard": "Balanced lighting with moderate shadows",
                "soft": "Soft lighting with minimal shadows", 
                "dramatic": "High contrast lighting with strong shadows"
            }
            description = descriptions.get(name, "Custom lighting setup")
            print(f"  {name}: {description}")
            
        print("\n=== Available Animation Configurations ===")
        if "animation_configs" in self.config:
            for name, anim in self.config["animation_configs"].items():
                print(f"  {name}: {anim['description']}")
                print(f"    幅度比例: {anim['amplitude_ratio']:.1%}, 频率: {anim['frequency']}x")
        
        print("\n=== Available Animation Types ===")
        if "animation_types" in self.config:
            for name, anim_type in self.config["animation_types"].items():
                print(f"  {name}: {anim_type['description']}")
    
    def create_renderer(self, render_config: str = "standard", 
                       lighting_setup: str = "standard") -> PartNetVideoRenderer:
        """
        Create a renderer with specified configuration.
        
        Args:
            render_config: Name of render configuration
            lighting_setup: Name of lighting setup
            
        Returns:
            Configured PartNetVideoRenderer
        """
        if render_config not in self.config["render_configs"]:
            raise ValueError(f"Unknown render config: {render_config}")
            
        if lighting_setup not in self.config["lighting_setups"]:
            raise ValueError(f"Unknown lighting setup: {lighting_setup}")
        
        # Get render settings
        render_settings = self.config["render_configs"][render_config]
        
        # Create renderer
        self.renderer = PartNetVideoRenderer(
            width=render_settings["width"],
            height=render_settings["height"],
            fps=render_settings["fps"]
        )
        
        # Apply lighting setup
        self._apply_lighting(lighting_setup)
        
        return self.renderer
    
    def _apply_lighting(self, lighting_setup: str):
        """Apply lighting configuration to the renderer."""
        if not self.renderer:
            raise RuntimeError("Renderer not created yet")
            
        lighting = self.config["lighting_setups"][lighting_setup]
        
        # Clear existing lights
        scene = self.renderer.scene
        
        # Set ambient light
        scene.set_ambient_light(lighting["ambient"])
        
        # Add directional light
        if "directional" in lighting:
            dir_light = lighting["directional"]
            scene.add_directional_light(
                direction=dir_light["direction"],
                color=dir_light["color"],
                shadow=dir_light.get("shadow", True)
            )
        
        # Add point lights
        if "point_lights" in lighting:
            for point_light in lighting["point_lights"]:
                scene.add_point_light(
                    position=point_light["position"],
                    color=point_light["color"],
                    shadow=point_light.get("shadow", True)
                )
    
    def generate_trajectory(self, trajectory_config: str, 
                          object_center: Optional[List[float]] = None,
                          object_type: str = "default") -> List[sapien.Pose]:
        """
        Generate camera trajectory from configuration.
        
        Args:
            trajectory_config: Name of trajectory configuration
            object_center: Custom object center, if None uses config
            object_type: Type of object for default center
            
        Returns:
            List of camera poses
        """
        if trajectory_config not in self.config["camera_trajectories"]:
            raise ValueError(f"Unknown trajectory config: {trajectory_config}")
        
        traj_config = self.config["camera_trajectories"][trajectory_config]
        
        # Determine object center
        if object_center is None:
            if object_type in self.config["object_centers"]:
                center = np.array(self.config["object_centers"][object_type])
            else:
                center = np.array(self.config["object_centers"]["default"])
        else:
            center = np.array(object_center)
        
        # Generate trajectory based on type
        if traj_config["type"] == "circular":
            return self.renderer.generate_circular_trajectory(
                center=center,
                radius=traj_config["radius"],
                height=traj_config["height"],
                n_frames=traj_config["frames"],
                full_rotation=traj_config.get("full_rotation", True)
            )
        elif traj_config["type"] == "spiral":
            return self.renderer.generate_spiral_trajectory(
                center=center,
                radius_range=traj_config["radius_range"],
                height_range=traj_config["height_range"],
                n_frames=traj_config["frames"]
            )
        else:
            raise ValueError(f"Unknown trajectory type: {traj_config['type']}")
    
    def render_object(self, urdf_path: str, 
                     render_config: str = "standard",
                     trajectory_config: str = "circular_medium",
                     lighting_setup: str = "standard",
                     object_type: str = "default",
                     output_dir: Optional[str] = None) -> str:
        """
        Render a single object with specified configurations.
        
        Args:
            urdf_path: Path to URDF file
            render_config: Render quality configuration
            trajectory_config: Camera trajectory configuration
            lighting_setup: Lighting configuration
            object_type: Object type for centering
            output_dir: Output directory (auto-generated if None)
            
        Returns:
            Output directory path
        """
        # Create renderer
        self.create_renderer(render_config, lighting_setup)
        
        # Load object
        asset = self.renderer.load_partnet_object(urdf_path)
        
        # Generate trajectory
        poses = self.generate_trajectory(trajectory_config, object_type=object_type)
        
        # Generate output directory name
        if output_dir is None:
            object_name = Path(urdf_path).parent.name
            output_dir = f"render_{object_name}_{render_config}_{trajectory_config}"
        
        # Render sequence
        self.renderer.render_sequence(poses, save_frames=True, output_dir=output_dir)
        
        # Create videos
        video_prefix = f"{render_config}_{trajectory_config}"
        self.renderer.create_videos(
            output_dir, 
            f"{video_prefix}_rgb.mp4", 
            f"{video_prefix}_depth.mp4"
        )
        
        # Save metadata
        self._save_render_metadata(output_dir, urdf_path, render_config, 
                                 trajectory_config, lighting_setup)
        
        return output_dir
    
    def _save_render_metadata(self, output_dir: str, urdf_path: str,
                            render_config: str, trajectory_config: str,
                            lighting_setup: str):
        """Save rendering metadata."""
        metadata = {
            "urdf_path": urdf_path,
            "render_config": render_config,
            "trajectory_config": trajectory_config,
            "lighting_setup": lighting_setup,
            "render_settings": self.config["render_configs"][render_config],
            "trajectory_settings": self.config["camera_trajectories"][trajectory_config],
            "lighting_settings": self.config["lighting_setups"][lighting_setup]
        }
        
        metadata_path = os.path.join(output_dir, "render_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def batch_render(self, urdf_paths: List[str],
                    render_configs: List[str] = ["standard"],
                    trajectory_configs: List[str] = ["circular_medium"],
                    lighting_setups: List[str] = ["standard"],
                    base_output_dir: str = "batch_render_output") -> List[str]:
        """
        Batch render multiple objects with different configurations.
        
        Args:
            urdf_paths: List of URDF file paths
            render_configs: List of render configurations to use
            trajectory_configs: List of trajectory configurations
            lighting_setups: List of lighting setups
            base_output_dir: Base directory for outputs
            
        Returns:
            List of output directory paths
        """
        os.makedirs(base_output_dir, exist_ok=True)
        output_dirs = []
        
        total_jobs = (len(urdf_paths) * len(render_configs) * 
                     len(trajectory_configs) * len(lighting_setups))
        job_count = 0
        
        print(f"Starting batch render: {total_jobs} total jobs")
        
        for urdf_path in urdf_paths:
            object_name = Path(urdf_path).parent.name
            
            for render_config in render_configs:
                for trajectory_config in trajectory_configs:
                    for lighting_setup in lighting_setups:
                        job_count += 1
                        print(f"\nJob {job_count}/{total_jobs}: {object_name}")
                        print(f"  Config: {render_config}, Trajectory: {trajectory_config}, Lighting: {lighting_setup}")
                        
                        try:
                            output_dir = os.path.join(
                                base_output_dir,
                                f"{object_name}_{render_config}_{trajectory_config}_{lighting_setup}"
                            )
                            
                            result_dir = self.render_object(
                                urdf_path=urdf_path,
                                render_config=render_config,
                                trajectory_config=trajectory_config,
                                lighting_setup=lighting_setup,
                                output_dir=output_dir
                            )
                            
                            output_dirs.append(result_dir)
                            print(f"  ✓ Completed: {result_dir}")
                            
                        except Exception as e:
                            print(f"  ✗ Failed: {e}")
                            continue
        
        print(f"\nBatch render completed. {len(output_dirs)} successful renders.")
        return output_dirs
    
    def create_animated_renderer(self, render_config: str = "standard", 
                                lighting_setup: str = "standard") -> AnimatedRenderer:
        """
        Create an animated renderer with specified configuration.
        
        Args:
            render_config: Name of render configuration
            lighting_setup: Name of lighting setup
            
        Returns:
            Configured AnimatedRenderer
        """
        if render_config not in self.config["render_configs"]:
            raise ValueError(f"Unknown render config: {render_config}")
            
        if lighting_setup not in self.config["lighting_setups"]:
            raise ValueError(f"Unknown lighting setup: {lighting_setup}")
        
        # Get render settings
        render_settings = self.config["render_configs"][render_config]
        
        # Create animated renderer
        animated_renderer = AnimatedRenderer(
            width=render_settings["width"],
            height=render_settings["height"],
            fps=render_settings["fps"]
        )
        
        # Apply lighting setup
        self._apply_animated_lighting(animated_renderer, lighting_setup)
        
        return animated_renderer
    
    def _apply_animated_lighting(self, renderer: AnimatedRenderer, lighting_setup: str):
        """Apply lighting configuration to the animated renderer."""
        lighting = self.config["lighting_setups"][lighting_setup]
        
        # Clear existing lights
        scene = renderer.scene
        
        # Set ambient light
        scene.set_ambient_light(lighting["ambient"])
        
        # Add directional light
        if "directional" in lighting:
            dir_light = lighting["directional"]
            scene.add_directional_light(
                direction=dir_light["direction"],
                color=dir_light["color"],
                shadow=dir_light.get("shadow", True)
            )
        
        # Add point lights
        if "point_lights" in lighting:
            for point_light in lighting["point_lights"]:
                scene.add_point_light(
                    position=point_light["position"],
                    color=point_light["color"],
                    shadow=point_light.get("shadow", True)
                )
    
    def render_animated_object(self, urdf_path: str, 
                             render_config: str = "standard",
                             trajectory_config: str = "circular_medium",
                             lighting_setup: str = "standard",
                             animation_type: str = "periodic",
                             animation_config: str = "standard",
                             object_type: str = "default",
                             output_dir: Optional[str] = None) -> str:
        """
        Render a single object with joint animations.
        
        Args:
            urdf_path: Path to URDF file
            render_config: Render quality configuration
            trajectory_config: Camera trajectory configuration
            lighting_setup: Lighting configuration
            animation_type: Joint animation type ("periodic", "oscillating", "sequential", "large_motion")
            animation_config: Animation intensity ("gentle", "standard", "energetic", "extreme")
            object_type: Object type for centering
            output_dir: Output directory (auto-generated if None)
            
        Returns:
            Output directory path
        """
        # Create animated renderer
        renderer = self.create_animated_renderer(render_config, lighting_setup)
        
        # Load object
        asset = renderer.load_partnet_object(urdf_path)
        
        # Generate trajectory
        poses = self.generate_animated_trajectory(renderer, trajectory_config, object_type=object_type)
        
        # Generate output directory name
        if output_dir is None:
            object_name = Path(urdf_path).parent.name
            output_dir = f"animated_{object_name}_{render_config}_{trajectory_config}_{animation_type}_{animation_config}"
        
        # Render animated sequence
        renderer.render_animated_sequence(
            poses, 
            animation_type=animation_type,
            animation_config=animation_config,
            save_frames=True, 
            output_dir=output_dir
        )
        
        # Create videos
        video_prefix = f"{render_config}_{trajectory_config}_{animation_type}"
        renderer.create_videos(
            output_dir, 
            f"{video_prefix}_rgb.mp4", 
            f"{video_prefix}_depth.mp4"
        )
        
        # Save metadata
        self._save_animated_metadata(output_dir, urdf_path, render_config, 
                                   trajectory_config, lighting_setup, animation_type)
        
        return output_dir
    
    def generate_animated_trajectory(self, renderer: AnimatedRenderer, trajectory_config: str, 
                                   object_center: Optional[List[float]] = None,
                                   object_type: str = "default") -> List[sapien.Pose]:
        """Generate camera trajectory for animated renderer."""
        if trajectory_config not in self.config["camera_trajectories"]:
            raise ValueError(f"Unknown trajectory config: {trajectory_config}")
        
        traj_config = self.config["camera_trajectories"][trajectory_config]
        
        # Determine object center
        if object_center is None:
            if object_type in self.config["object_centers"]:
                center = np.array(self.config["object_centers"][object_type])
            else:
                center = np.array(self.config["object_centers"]["default"])
        else:
            center = np.array(object_center)
        
        # Generate trajectory based on type
        if traj_config["type"] == "circular":
            return renderer.generate_circular_trajectory(
                center=center,
                radius=traj_config["radius"],
                height=traj_config["height"],
                n_frames=traj_config["frames"],
                full_rotation=traj_config.get("full_rotation", True)
            )
        elif traj_config["type"] == "spiral":
            return renderer.generate_spiral_trajectory(
                center=center,
                radius_range=traj_config["radius_range"],
                height_range=traj_config["height_range"],
                n_frames=traj_config["frames"]
            )
        else:
            raise ValueError(f"Unknown trajectory type: {traj_config['type']}")
    
    def _save_animated_metadata(self, output_dir: str, urdf_path: str,
                              render_config: str, trajectory_config: str,
                              lighting_setup: str, animation_type: str):
        """Save animated rendering metadata."""
        metadata = {
            "urdf_path": urdf_path,
            "render_config": render_config,
            "trajectory_config": trajectory_config,
            "lighting_setup": lighting_setup,
            "animation_type": animation_type,
            "render_settings": self.config["render_configs"][render_config],
            "trajectory_settings": self.config["camera_trajectories"][trajectory_config],
            "lighting_settings": self.config["lighting_setups"][lighting_setup]
        }
        
        metadata_path = os.path.join(output_dir, "animated_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    """Example usage of the configurable renderer."""
    
    # Create configurable renderer
    renderer = ConfigurableRenderer()
    
    # List available configurations
    renderer.list_configurations()
    
    # Example 1: Single object rendering
    print("\n=== Single Object Rendering ===")
    urdf_path = "../assets/179/mobility.urdf"
    
    try:
        output_dir = renderer.render_object(
            urdf_path=urdf_path,
            render_config="standard",
            trajectory_config="circular_medium",
            lighting_setup="standard"
        )
        print(f"Single render completed: {output_dir}")
    except Exception as e:
        print(f"Single render failed: {e}")
    
    # Example 2: Batch rendering with multiple configurations
    print("\n=== Batch Rendering ===")
    urdf_paths = [
        "../assets/179/mobility.urdf",
        "../assets/180/mobility.urdf"
    ]
    
    try:
        output_dirs = renderer.batch_render(
            urdf_paths=urdf_paths,
            render_configs=["standard", "high_quality"],
            trajectory_configs=["circular_close", "spiral_outward"],
            lighting_setups=["standard", "dramatic"]
        )
        print(f"Batch render completed: {len(output_dirs)} outputs")
    except Exception as e:
        print(f"Batch render failed: {e}")


if __name__ == "__main__":
    main()
