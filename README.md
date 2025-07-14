# PartNet-Mobility 视频渲染工具 🎬

一个功能强大的 SAPIEN 渲染工具包，专为 PartNet-Mobility 数据集设计。支持生成高质量 RGB/深度视频、关节动画渲染，以及完整的相机参数记录。

## ✨ 核心特性

- 🎥 **高质量视频渲染** - 生成 RGB 和深度视频序列
- � **关节动画系统** - 支持可动部件的真实关节运动
- �📸 **完整相机参数** - 记录每帧的内参、外参和姿态信息
- 🔄 **多样化轨迹** - 圆形、螺旋形等多种相机运动模式
- 🎛️ **统一配置管理** - JSON 配置文件支持，参数化渲染
- � **批量处理能力** - 支持大规模数据集批量渲染
- �💾 **多格式输出** - 图片序列、视频文件、参数数据等

## 🛠️ 环境配置

### 核心依赖

```bash
# SAPIEN 3D 仿真器 (推荐最新版本)
pip install sapien>=3.0.0b1

# 必需的 Python 库
pip install numpy opencv-python pillow

# 可选增强功能
pip install matplotlib open3d trimesh
```

### 快速验证安装

```bash
python quick_start.py check-deps
```

## 📁 项目结构

```
sapien/
├── render_partnet_video.py      # 核心渲染引擎
├── animated_renderer.py         # 关节动画渲染器 ✨
├── advanced_renderer.py         # 高级配置渲染器
├── quick_start.py               # 命令行工具
├── render_config.json           # 统一配置文件 ✨
├── example_usage.py             # 编程接口示例
├── camera.py                    # SAPIEN 相机基础示例
└── rt_stereodepth.py            # 双目深度传感器示例
```

### 核心组件说明

- **`render_partnet_video.py`** - 基础渲染类，提供核心视频生成功能
- **`animated_renderer.py`** - 动画渲染器，支持关节运动和统一配置管理
- **`advanced_renderer.py`** - 高级渲染器，支持批量处理和配置组合
- **`render_config.json`** - 统一配置文件，包含渲染质量、轨迹、光照和动画参数
- **`quick_start.py`** - 命令行接口，提供简单到高级的多种使用模式

## 🚀 快速开始

### 1. 环境检查

首先确认所有依赖都已正确安装：

```bash
python quick_start.py check-deps
```

### 2. 静态渲染 - 最简使用

```bash
# 基础渲染，使用默认设置
python quick_start.py simple /path/to/your/mobility.urdf

# 指定输出目录
python quick_start.py simple /path/to/your/mobility.urdf --output my_output
```

### 3. 高级配置渲染

```bash
# 查看所有可用配置选项
python quick_start.py list-configs

# 使用特定质量和轨迹设置
python quick_start.py advanced /path/to/your/mobility.urdf \
    --config high_quality \
    --trajectory spiral_outward \
    --lighting dramatic

# 完整配置示例
python quick_start.py advanced /path/to/your/mobility.urdf \
    --config ultra_high \
    --trajectory circular_far \
    --lighting soft \
    --output premium_output
```

### 4. 关节动画渲染 ✨

```bash
# 基础动画渲染
python quick_start.py animated /path/to/your/mobility.urdf

# 使用配置强度控制动画幅度
python quick_start.py animated /path/to/your/mobility.urdf \
    --animation-config gentle \
    --animation periodic

# 高强度动画配置
python quick_start.py animated /path/to/your/mobility.urdf \
    --animation-config extreme \
    --animation large_motion \
    --config high_quality
```

### 配置选项说明

#### 渲染质量配置 (`--config`)
- **`low_quality`** - 快速预览 (512×384, 30fps)
- **`standard`** - 标准质量 (640×480, 30fps) 
- **`high_quality`** - 高质量 (1280×720, 30fps)
- **`ultra_high`** - 超高质量 (1920×1080, 60fps)

#### 相机轨迹配置 (`--trajectory`)
- **`circular_close`** - 近距离圆形轨迹 (半径1.5m)
- **`circular_medium`** - 中距离圆形轨迹 (半径2.5m)
- **`circular_far`** - 远距离圆形轨迹 (半径4.0m)
- **`spiral_inward`** - 由远及近螺旋轨迹
- **`spiral_outward`** - 由近及远螺旋轨迹

#### 光照设置 (`--lighting`)
- **`standard`** - 标准光照设置
- **`soft`** - 柔和光照，适合细节展示
- **`dramatic`** - 戏剧性光照，增强视觉效果

#### 动画强度配置 (`--animation-config`) ✨
- **`gentle`** - 轻柔动画 (30% 幅度, 0.5x 频率)
- **`standard`** - 标准动画 (60% 幅度, 1.0x 频率)
- **`energetic`** - 活跃动画 (85% 幅度, 1.5x 频率)
- **`extreme`** - 极限动画 (95% 幅度, 2.0x 频率)



## 💻 编程接口使用

### 基础静态渲染

```python
from render_partnet_video import PartNetVideoRenderer
import numpy as np

# 创建渲染器
renderer = PartNetVideoRenderer(width=640, height=480, fps=30)

# 加载 PartNet-Mobility 对象
urdf_path = "path/to/your/mobility.urdf"
asset = renderer.load_partnet_object(urdf_path)

# 生成圆形相机轨迹
poses = renderer.generate_circular_trajectory(
    center=np.array([0, 0, 0.5]),
    radius=2.0,
    height=1.5,
    n_frames=120,
    full_rotation=True
)

# 渲染序列并生成视频
renderer.render_sequence(poses, save_frames=True, output_dir="output")
renderer.create_videos("output")
```

### 关节动画渲染 ✨

```python
from animated_renderer import AnimatedRenderer
import numpy as np

# 创建动画渲染器
renderer = AnimatedRenderer(width=640, height=480, fps=30)

# 加载对象
asset = renderer.load_partnet_object("path/to/mobility.urdf")

# 生成相机轨迹
poses = renderer.generate_circular_trajectory(
    center=np.array([0, 0, 0.5]),
    radius=2.0,
    height=1.5,
    n_frames=120
)

# 使用配置文件渲染动画
renderer.render_animated_sequence(
    poses,
    animation_config="energetic",    # 动画强度
    animation_type="periodic",       # 动画类型
    output_dir="animated_output"
)

# 创建视频
renderer.create_videos("animated_output")
```

### 高级配置渲染

```python
from advanced_renderer import ConfigurableRenderer

# 创建配置化渲染器
renderer = ConfigurableRenderer()

# 使用配置文件单个对象渲染
output_dir = renderer.render_object(
    urdf_path="path/to/mobility.urdf",
    render_config="high_quality",      # 渲染质量
    trajectory_config="circular_far",   # 相机轨迹
    lighting_setup="dramatic"          # 光照设置
)

# 关节动画渲染
output_dir = renderer.render_animated_object(
    urdf_path="path/to/mobility.urdf",
    render_config="standard",
    trajectory_config="spiral_outward",
    lighting_setup="soft",
    animation_config="gentle",         # 动画强度
    animation_type="oscillating"       # 动画类型
)
```

### 多种相机轨迹

```python
# 螺旋轨迹
poses = renderer.generate_spiral_trajectory(
    center=np.array([0, 0, 0.8]),
    radius_range=(1.0, 3.0),
    height_range=(0.5, 2.0),
    n_frames=150
)

# 自定义轨迹
poses = []
for i in range(n_frames):
    # 定义相机位置和朝向
    position = [x, y, z]
    # 计算旋转矩阵...
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    
    pose = sapien.Pose.from_transformation_matrix(transform)
    poses.append(pose)
```
```

## 📁 输出文件结构

### 静态渲染输出

```
output/
├── rgb/                    # RGB 图片序列
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── depth/                  # 深度图片和数据
│   ├── frame_000000.png    # 可视化深度图
│   ├── frame_000000.npy    # 原始深度数据 (float32)
│   └── ...
├── camera_params.json      # 相机参数记录
├── rgb_video.mp4          # RGB 视频文件
└── depth_video.mp4        # 深度视频文件
```

### 动画渲染输出 ✨

```
animated_output/
├── rgb/                    # RGB 图片序列
├── depth/                  # 深度图片序列  
├── camera_params.json      # 相机参数记录
├── joint_states.json       # 关节状态记录 ✨
├── animation_config.json   # 动画配置记录 ✨
├── rgb_video.mp4          # RGB 视频文件
└── depth_video.mp4        # 深度视频文件
```

## 📊 数据格式说明

### 相机参数格式

`camera_params.json` 包含每帧的完整相机信息：

```json
[
  {
    "model_matrix": [[...], [...], [...], [...]],  // 4x4 模型矩阵
    "camera_pose": {
      "position": [x, y, z],                       // 相机世界坐标位置
      "quaternion": [w, x, y, z]                   // 相机朝向四元数
    },
    "intrinsic_matrix": [[...], [...], [...]],     // 3x3 内参矩阵
    "width": 640,                                  // 图像宽度
    "height": 480                                  // 图像高度
  }
  // ... 每帧都有对应记录
]
```

### 关节状态格式 ✨

`joint_states.json` 记录动画过程中的关节信息：

```json
[
  {
    "frame": 0,                      // 帧编号
    "time": 0.0,                     // 时间戳 (秒)
    "qpos": [0.0, 0.5, ...],         // 关节位置数组
    "qvel": [0.0, 0.0, ...],         // 关节速度数组
    "joint_names": ["joint1", ...]   // 关节名称映射
  }
  // ... 每帧的关节状态
]
```

### 动画配置格式 ✨

`animation_config.json` 保存使用的动画参数：

```json
{
  "animation_config": "energetic",   // 动画强度配置
  "animation_type": "periodic",      // 动画类型
  "settings": {
    "amplitude_ratio": 0.85,         // 幅度比例
    "frequency_multiplier": 1.5,     // 频率倍数
    "n_frames": 120,                 // 总帧数
    "fps": 30                        // 帧率
  },
  "joint_count": 8,                  // 可动关节数量
  "timestamp": "2025-07-14T10:30:00" // 渲染时间戳
}
```

## 🔧 高级功能

### 批量渲染处理

```python
from advanced_renderer import ConfigurableRenderer

renderer = ConfigurableRenderer()

# 批量渲染多个对象，使用不同配置组合
urdf_paths = [
    "assets/chair/mobility.urdf",
    "assets/cabinet/mobility.urdf", 
    "assets/table/mobility.urdf"
]

output_dirs = renderer.batch_render(
    urdf_paths=urdf_paths,
    render_configs=["standard", "high_quality"],           # 2种质量
    trajectory_configs=["circular_close", "spiral_outward"], # 2种轨迹
    lighting_setups=["standard", "dramatic"],              # 2种光照
    # 总共生成 3×2×2×2 = 24 个视频
)
```

### 自定义关节动画

```python
from animated_renderer import AnimatedRenderer

renderer = AnimatedRenderer()

# 创建自定义动画配置
custom_animations = {
    "drawer_joint": {
        "type": "sine",           # 正弦波动画
        "range": [0.0, 0.5],      # 运动范围 (米)
        "frequency": 1.0,         # 频率 (Hz)
        "phase": 0.0,            # 相位偏移
        "offset": 0.0            # 基础偏移
    },
    "door_joint": {
        "type": "linear",         # 线性动画
        "range": [0.0, np.pi/2], # 0到90度
        "frequency": 0.8,
        "phase": np.pi/4,
        "offset": 0.0
    },
    "lid_joint": {
        "type": "oscillating",    # 振荡动画
        "range": [0.0, np.pi/3],
        "frequency": 1.2,
        "phase": 0.0,
        "offset": 0.0
    }
}

# 应用自定义动画
animations = renderer.create_custom_animation(custom_animations)
renderer.render_animated_sequence(poses, animations=animations)
```

### 高质量渲染设置

```python
# 超高质量渲染配置
renderer = PartNetVideoRenderer(
    width=3840,      # 4K 分辨率
    height=2160,
    fps=60,          # 高帧率
    samples=4        # 抗锯齿采样
)

# 高级光照设置
renderer.scene.set_ambient_light([0.3, 0.3, 0.3])
renderer.scene.add_directional_light([1, 1, -1], [0.8, 0.8, 0.8], shadow=True)
renderer.scene.add_point_light([2, 2, 2], [0.5, 0.5, 0.5])

# 启用更多渲染特性
renderer.enable_shadow = True
renderer.enable_reflection = True
```

### 深度数据分析

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载并分析深度数据
depth_data = np.load("output/depth/frame_000000.npy")

print(f"深度范围: {depth_data.min():.3f} - {depth_data.max():.3f} 米")
print(f"平均深度: {depth_data.mean():.3f} 米")
print(f"深度标准差: {depth_data.std():.3f} 米")

# 深度分布直方图
plt.figure(figsize=(10, 6))
plt.hist(depth_data.flatten(), bins=50, alpha=0.7)
plt.xlabel('深度 (米)')
plt.ylabel('像素数量')
plt.title('深度分布直方图')
plt.show()

# 转换为点云 (需要相机内参)
def depth_to_pointcloud(depth, intrinsic_matrix):
    height, width = depth.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 转换为3D坐标
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    points = np.stack([x, y, z], axis=-1)
    return points.reshape(-1, 3)
```

## 📐 坐标系统说明

### 相机坐标系

- **SAPIEN 相机坐标系**: X→前向, Y→左, Z→上 (右手坐标系)
- **OpenGL 相机坐标系**: X→右, Y→上, Z→后向 (深度为负值)
- **图像坐标系**: 原点在左上角，u→右，v→下

### 变换矩阵说明

- **模型矩阵**: 从 OpenGL 相机空间到 SAPIEN 世界空间的 4×4 变换矩阵
- **内参矩阵**: 3×3 相机内参，包含焦距 (fx, fy) 和主点 (cx, cy)
- **外参**: 相机在世界坐标系中的位置和旋转 (四元数表示)

## 🚨 故障排除

### 常见问题及解决方案

#### 1. 环境配置问题

**URDF 加载失败**
```bash
# 问题：FileNotFoundError 或加载失败
# 解决：检查文件路径和依赖文件
ls -la /path/to/your/mobility.urdf
python -c "import sapien; print(sapien.__version__)"
```

**依赖库版本冲突**
```bash
# 检查 SAPIEN 版本兼容性
pip show sapien
pip install --upgrade sapien>=3.0.0b1
```

#### 2. 渲染问题

**渲染结果异常 (空白或错误)**
- 调整相机轨迹参数：增加半径或改变高度
- 检查物体边界框：`renderer.get_bounding_box()`
- 验证 URDF 文件完整性

**关节动画无效果**
- 检查 URDF 是否包含可动关节
- 确认关节限制范围：`renderer.get_joint_limits()`
- 尝试不同动画强度：`--animation-config extreme`

#### 3. 性能问题

**渲染速度慢**
```python
# 性能优化配置
renderer = PartNetVideoRenderer(
    width=640,      # 降低分辨率
    height=480,
    fps=24,         # 降低帧率
    samples=1       # 减少抗锯齿采样
)
```

**内存占用过高**
- 分批处理大数据集
- 使用较低分辨率进行测试
- 及时清理中间文件

#### 4. 输出问题

**视频创建失败**
```bash
# 检查 OpenCV 安装
python -c "import cv2; print(cv2.__version__)"

# 检查输出目录权限
chmod 755 output_directory
```

**文件格式不支持**
- 确保 OpenCV 支持 H.264 编解码器
- 尝试不同的视频格式 (.avi, .mov)

### 性能优化建议

#### 渲染优化

```python
# 针对不同用途的性能配置

# 快速预览配置
preview_config = {
    "width": 512, "height": 384, "fps": 15,
    "n_frames": 60, "samples": 1
}

# 标准生产配置  
production_config = {
    "width": 1280, "height": 720, "fps": 30,
    "n_frames": 120, "samples": 2
}

# 高质量展示配置
showcase_config = {
    "width": 1920, "height": 1080, "fps": 60,
    "n_frames": 180, "samples": 4
}
```

#### 批量处理优化

```python
# 并行处理建议
from multiprocessing import Pool
import os

def render_single_object(args):
    urdf_path, config, output_base = args
    # 单个对象渲染逻辑
    pass

# 使用进程池并行处理
if __name__ == "__main__":
    cpu_count = os.cpu_count()
    with Pool(processes=cpu_count // 2) as pool:  # 使用一半CPU核心
        pool.map(render_single_object, args_list)
```

## 🎯 应用场景与扩展

### 典型应用场景

#### 1. 机器人学研究
- **物体操作学习**: 生成多视角物体视频用于视觉感知训练
- **关节运动分析**: 分析可动部件的运动轨迹和动力学特性
- **仿真环境构建**: 为机器人仿真提供高质量视觉数据

#### 2. 计算机视觉
- **数据增强**: 为深度学习模型生成多样化的训练数据
- **3D 物体检测**: 提供标准化的多视角数据集
- **运动估计**: 基于关节动画的运动序列分析

#### 3. 工业设计
- **产品展示**: 生成专业的产品演示视频
- **运动机构验证**: 验证机械结构的运动合理性
- **用户界面设计**: 为AR/VR应用提供物体模型

### 扩展功能开发

```python
# 示例：添加语义分割渲染
class SemanticRenderer(PartNetVideoRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_semantic_rendering()
    
    def render_semantic_frame(self, pose):
        # 实现语义分割渲染
        rgb_img = self.render_rgb(pose)
        semantic_img = self.render_semantic_labels(pose)
        return rgb_img, semantic_img

# 示例：多相机同时渲染
class MultiCameraRenderer(PartNetVideoRenderer):
    def __init__(self, camera_configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cameras = [self.create_camera(config) for config in camera_configs]
    
    def render_multicamera_frame(self, poses):
        frames = []
        for camera, pose in zip(self.cameras, poses):
            frame = self.render_with_camera(camera, pose)
            frames.append(frame)
        return frames
```

## 📚 示例与教程

### 完整使用示例

```bash
# 基础功能测试
python quick_start.py check-deps                    # 环境检查
python quick_start.py list-configs                  # 查看配置选项

# 静态渲染示例
python quick_start.py simple /path/to/chair.urdf    # 最简渲染
python quick_start.py advanced /path/to/chair.urdf \
    --config high_quality --lighting dramatic       # 高级渲染

# 动画渲染示例  
python quick_start.py animated /path/to/cabinet.urdf \
    --animation-config energetic \                   # 动画强度
    --animation periodic \                           # 动画类型
    --config standard                                # 渲染质量

# 批量处理示例
for urdf in /dataset/partnet-mobility/*.urdf; do
    python quick_start.py animated "$urdf" \
        --animation-config standard \
        --output "batch_output/$(basename "$urdf" .urdf)"
done
```

### 编程接口完整示例

查看详细的编程接口使用方法：

```bash
python example_usage.py          # 基础渲染示例
python camera.py                 # SAPIEN 相机功能演示  
python rt_stereodepth.py         # 双目深度传感器示例
```

### 数据集处理流水线

```python
# 完整的数据集处理示例
import os
from pathlib import Path
from advanced_renderer import ConfigurableRenderer

def process_partnet_dataset(dataset_root, output_root):
    """处理整个 PartNet-Mobility 数据集"""
    renderer = ConfigurableRenderer()
    
    # 遍历数据集
    for category_dir in Path(dataset_root).iterdir():
        if not category_dir.is_dir():
            continue
            
        print(f"处理类别: {category_dir.name}")
        
        for obj_dir in category_dir.iterdir():
            urdf_path = obj_dir / "mobility.urdf"
            if not urdf_path.exists():
                continue
                
            output_dir = Path(output_root) / category_dir.name / obj_dir.name
            
            try:
                # 渲染静态视频
                renderer.render_object(
                    str(urdf_path),
                    render_config="standard",
                    trajectory_config="circular_medium",
                    lighting_setup="standard",
                    output_dir=str(output_dir / "static")
                )
                
                # 渲染动画视频
                renderer.render_animated_object(
                    str(urdf_path),
                    render_config="standard", 
                    animation_config="standard",
                    animation_type="periodic",
                    output_dir=str(output_dir / "animated")
                )
                
                print(f"✅ 完成: {obj_dir.name}")
                
            except Exception as e:
                print(f"❌ 失败: {obj_dir.name} - {e}")

# 使用示例
if __name__ == "__main__":
    process_partnet_dataset(
        dataset_root="/path/to/partnet-mobility",
        output_root="/path/to/rendered_videos"
    )
```

## 🤝 贡献与支持

### 参与贡献

欢迎提交 Issue 和 Pull Request 来改进这个工具！

**常见贡献方向:**
- 添加新的相机轨迹类型
- 实现更多动画模式
- 优化渲染性能
- 完善文档和示例
- 添加单元测试

### 获得帮助

如果遇到问题，请按以下顺序寻求帮助：

1. **查看本 README** - 大多数常见问题都有解答
2. **运行诊断命令** - `python quick_start.py check-deps`
3. **查看示例代码** - `example_usage.py` 等文件
4. **提交 Issue** - 提供详细的错误信息和环境配置


