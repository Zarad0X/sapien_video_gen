# SAPIEN PartNet-Mobility Video Renderer

A 3D object video rendering tool that supports multiple camera trajectories, joint animations, and high-quality rendering.Used in generating articulation objects from partnet-mobility dataset.

##  Installation

### Install Dependencies

#### Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n sapien python=3.10
conda activate sapien

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage
```bash
# Basic animated rendering
python quick_start.py /path/to/mobility.urdf

# High quality rendering with custom settings
python quick_start.py /path/to/mobility.urdf \
  --config high_quality \
  --trajectory sphere_spiral_custom \
  --output my_output \
  --scale 0.15 \
  --speed 3

# With custom animation configuration
python quick_start.py /path/to/mobility.urdf \
  --config high_quality \
  --trajectory sphere_spiral_custom \
  --animation-config config.json \
  --speed 2
```

##  Features

###  Camera Trajectory Types
- **Circular trajectory** (`circular_medium`): Circular orbit around the object at fixed height
- **Sphere spiral** (`sphere_spiral_custom`): Spiral movement on sphere surface with configurable elevation angles

#### Trajectory Parameters
**Circular Trajectory:**
- `radius`: Orbit radius (meters)
- `height`: Height offset from center (meters)  
- `n_frames`: Number of camera positions
- `full_rotation`: Complete 360° orbit (True/False)

**Sphere Spiral Trajectory:**
- `radius`: Sphere radius (meters)
- `start_elevation`: Starting elevation angle (-90° to 90°)
- `end_elevation`: Ending elevation angle (-90° to 90°)
- `rotations`: Number of horizontal rotations
- `n_frames`: Number of camera positions


###  Rendering Quality

- **Standard** (`standard`): 640×480, 30fps - Daily use
- **High quality** (`high_quality`): 1280×720, 30fps - For presentation
- **Ultra high** (`ultra_high`): 1920×1080, 60fps - For publication



## 🔨 Usage Examples

### 1. Command Line Usage
```bash

python quick_start.py /path/to/mobility.urdf   --config high_quality   --trajectory sphere_spiral_custom  --output usb --scale 0.15 --rotation 90 --speed 3 --animation-config config

```

### 2. Python API Usage
```python
from animated_renderer import AnimatedRenderer
from camera_traj import generate_sphere_spiral_trajectory
import numpy as np

# Create renderer
renderer = AnimatedRenderer(width=1280, height=720, fps=30)

# Load object
asset = renderer.load_partnet_object("mobility.urdf", scale=0.15)

# Generate camera trajectory
poses = generate_sphere_spiral_trajectory(
    center=np.array([0, 0, 0]),
    radius=1,
    start_elevation=60,
    end_elevation=-60,
    rotations=3,
    n_frames=300
)

# Create animations
animations = renderer.create_custom_animation(speed=2.0)

# Render
renderer.render_animated_sequence(poses, animations, output_dir="output")
renderer.create_videos("output", "rgb_video.mp4", "depth_video.mp4")
```


## 📁 Output File Structure
```
output_directory/
├── rgb/                    # RGB image sequence
│   ├── 00000.png
│   └── ...
├── depth/                  # Raw depth data (.npz, compressed)
│   ├── 00000.npz
│   └── ...
├── vis/                    # Colored depth visualization (.png)
│   ├── 00000.png
│   └── ...
├── camera_params.json      # Camera extrinsic parameters (per frame)
├── cam_K.txt               # Camera intrinsic parameters (single copy)
├── joint_states.json       # Joint states (animation rendering)
├── rgb_video.mp4           # RGB video
├── vis_video.mp4           # Depth visualization video
```






## 🤝 Contributing

Issues and Pull Requests are welcome!

---
**Note:**
- Please download the partnet-mobility-v0/ dataset through the sapien platform yourself

## 📄 License

MIT License
