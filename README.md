# SAPIEN PartNet-Mobility 视频渲染器

一个功能强大的 3D 物体视频渲染工具，支持多种相机轨迹、关节动画和高质量渲染。

## 🚀 快速开始

### 安装依赖
```bash
pip install sapien==2.2 numpy opencv-python pillow matplotlib
```

### 基础使用
```bash
# 简单渲染（默认设置）
python quick_start.py simple /path/to/mobility.urdf

# 高级渲染（自定义配置）
python quick_start.py advanced /path/to/mobility.urdf --config high_quality --trajectory sphere_spiral_custom

# 动画渲染（关节运动）
python quick_start.py animated /path/to/mobility.urdf --config high_quality --trajectory sphere_spiral_custom --animation periodic --animation-config energetic
```

## 📋 功能特性

### 🎥 相机轨迹类型
- **圆形轨迹** (`circular`): 围绕物体环形拍摄
- **螺旋轨迹** (`spiral`): 由远及近或由近及远的螺旋运动
- **球面螺旋** (`sphere_spiral`): 球面上的螺旋运动，可设置起始/结束仰角

### � 动画类型
- **周期性** (`periodic`): 关节在范围内来回摆动
- **振荡** (`oscillating`): 基于正弦波的平滑运动
- **顺序** (`sequential`): 不同关节依次运动
- **大幅度** (`large_motion`): 使用关节的完整运动范围

### 🔧 渲染质量
- **低质量** (`low_quality`): 320×240, 15fps - 快速预览
- **标准** (`standard`): 640×480, 30fps - 日常使用
- **高质量** (`high_quality`): 1280×720, 30fps - 展示用
- **超高质量** (`ultra_high`): 1920×1080, 60fps - 发布用

### � 光照设置
- **标准** (`standard`): 平衡的三点照明
- **柔和** (`soft`): 柔和均匀的照明
- **戏剧性** (`dramatic`): 强对比度照明

## 📊 配置参数详解

### 视频时长控制
```json
{
  "frames": 120,     // 总帧数
  "fps": 30          // 每秒帧数
}
```
**视频时长 = frames ÷ fps** (例: 120÷30 = 4秒)

### 球面螺旋轨迹参数
```json
{
  "type": "sphere_spiral",
  "radius": 4.0,           // 球体半径（米）
  "start_elevation": 60,   // 起始仰角（度）-90~90
  "end_elevation": -60,    // 结束仰角（度）
  "rotations": 5,          // 旋转圈数
  "frames": 120            // 总帧数
}
```

### 仰角说明
- **+90°**: 天顶（正上方俯视）
- **0°**: 水平线（侧面拍摄）
- **-90°**: 天底（正下方仰视）

### 动画强度配置
```json
{
  "amplitude_ratio": 0.6,    // 运动幅度比例 (0-1)
  "frequency": 1.0,          // 运动频率
  "default_range_revolute": 0.79,    // 旋转关节默认范围（弧度）
  "default_range_prismatic": 0.1     // 直线关节默认范围（米）
}
```

## 🔧 文件结构
- `render_partnet_video.py` - 核心渲染引擎
- `animated_renderer.py` - 关节动画渲染器
- `advanced_renderer.py` - 高级渲染器
- `quick_start.py` - 命令行工具
- `render_config.json` - 配置文件

## 🔨 使用示例

### 1. 命令行快速渲染
```bash
# 检查依赖
python quick_start.py check-deps

# 查看所有配置选项
python quick_start.py list-configs

# 高质量球面螺旋渲染
python quick_start.py advanced mobility.urdf \
  --config high_quality \
  --trajectory sphere_spiral_custom \
  --lighting dramatic \
  --output my_output

# 动画渲染with关节运动
python quick_start.py animated mobility.urdf \
  --config high_quality \
  --trajectory sphere_spiral_custom \
  --animation periodic \
  --animation-config energetic
```

### 2. Python API 调用
```python
from advanced_renderer import ConfigurableRenderer

# 创建渲染器
renderer = ConfigurableRenderer()

# 单个物体渲染
output_dir = renderer.render_object(
    urdf_path="mobility.urdf",
    render_config="high_quality",
    trajectory_config="sphere_spiral_custom",
    lighting_setup="dramatic"
)

# 动画渲染
output_dir = renderer.render_animated_object(
    urdf_path="mobility.urdf",
    render_config="high_quality",
    trajectory_config="circular_medium",
    animation_type="periodic",
    animation_config="energetic"
)
```

### 3. 球面螺旋轨迹示例
```python
from render_partnet_video import PartNetVideoRenderer
import numpy as np

renderer = PartNetVideoRenderer()
asset = renderer.load_partnet_object("mobility.urdf")

# 球面螺旋轨迹：相机在球面上螺旋运动，始终对着中心
poses = renderer.generate_sphere_spiral_trajectory(
    center=np.array([0, 0, 0.5]),
    radius=3.0,                    # 球面半径
    start_elevation=85,            # 起始仰角（度），85度接近顶部
    end_elevation=-85,             # 结束仰角（度），-85度接近底部
    rotations=3,                   # 转圈数
    n_frames=180                   # 帧数
)

renderer.render_sequence(poses, output_dir="sphere_spiral_output")
renderer.create_videos("sphere_spiral_output")
```

## 📁 输出文件结构
```
output_directory/
├── rgb/                    # RGB图像序列
│   ├── 00000.png
│   └── ...
├── depth/                  # 原始深度数据（.npz，压缩）
│   ├── 00000.npz
│   └── ...
├── depth_vis/              # 彩色深度可视化（.png，0区为黑色）
│   ├── 00000.png
│   └── ...
├── camera_params.json      # 相机外参（每帧）
├── cam_K.txt               # 相机内参（仅一份）
├── joint_states.json       # 关节状态（动画渲染）
├── *_rgb.mp4               # RGB视频
├── *_depth.mp4             # 深度视频
```

## ⚙️ 自定义配置

### 修改 `render_config.json`
```json
{
  "camera_trajectories": {
    "sphere_spiral_custom": {
      "type": "sphere_spiral",
      "radius": 5.0,
      "start_elevation": 45,
      "end_elevation": -45,
      "rotations": 3,
      "frames": 180
    }
  }
}
```

### 创建自定义动画
```
python quick_start.py /home/zhanghan/sapien/partnet-mobility-v0/dataset/100109/mobility.urdf   --config high_quality   --trajectory sphere_spiral_custom  --output usb --scale 0.15 --speed 3 
```



## 🔍 故障排除

### 常见问题
1. **URDF加载失败**: 检查文件路径和依赖的mesh文件
2. **渲染质量不佳**: 调整光照设置或相机距离
3. **动画不自然**: 修改动画强度配置或关节限制
4. **内存不足**: 降低渲染分辨率或减少帧数

### 调试模式
```python
renderer = AnimatedRenderer(debug=True)  
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---
**注意：**
- partnet-mobility-v0/ 数据集请通过sapien平台自行下载

## 📄 许可证

MIT License
