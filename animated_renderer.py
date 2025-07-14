"""
Animated PartNet-Mobility renderer with joint animation support.

This script extends the basic renderer to support articulated object                    # 处理无限制关节：设置合理的默认范围
                    if np.isinf(limit_min) or np.isinf(limit_max):
                        # 使用配置中的默认范围
                        if 'revolute' in str(joint.type).lower():
                            half_range = anim_settings["default_range_revolute"] / 2
                            limit_min = -half_range
                            limit_max = half_range
                            print(f"  无限制关节 {joint.name} 使用配置范围: ±{np.rad2deg(half_range):.1f}度")
                        elif 'prismatic' in str(joint.type).lower():
                            half_range = anim_settings["default_range_prismatic"] / 2
                            limit_min = -half_range
                            limit_max = half_range
                            print(f"  无限制关节 {joint.name} 使用配置范围: ±{half_range:.3f}米")during video rendering, making the joints move realistically.
"""

import json
import os
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from pathlib import Path

import sapien.core as sapien
from render_partnet_video import PartNetVideoRenderer


class AnimatedRenderer(PartNetVideoRenderer):
    """
    Extended renderer that supports joint animations during video rendering.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, debug: bool = False, config_file: str = "render_config.json"):
        super().__init__(width, height, fps)
        self.articulation = None
        self.joint_animations = {}
        self.debug = debug
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            print(f"Warning: Config file {self.config_file} not found, using defaults")
            return {}
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}, using defaults")
            return {}
    
    def _get_animation_settings(self, animation_config: str) -> Dict:
        """获取动画配置参数"""
        if "animation_configs" in self.config and animation_config in self.config["animation_configs"]:
            return self.config["animation_configs"][animation_config]
        
        # 默认设置
        defaults = {
            "gentle": {"amplitude_ratio": 0.3, "frequency": 0.5, "default_range_revolute": 0.52, "default_range_prismatic": 0.05},
            "standard": {"amplitude_ratio": 0.6, "frequency": 1.0, "default_range_revolute": 0.79, "default_range_prismatic": 0.1},
            "energetic": {"amplitude_ratio": 0.85, "frequency": 1.5, "default_range_revolute": 1.05, "default_range_prismatic": 0.15},
            "extreme": {"amplitude_ratio": 0.95, "frequency": 1.0, "default_range_revolute": 1.57, "default_range_prismatic": 0.2}
        }
        return defaults.get(animation_config, defaults["standard"])
    
    def _get_animation_type_settings(self, animation_type: str) -> Dict:
        """获取动画类型参数"""
        if "animation_types" in self.config and animation_type in self.config["animation_types"]:
            return self.config["animation_types"][animation_type]
        
        # 默认设置
        defaults = {
            "periodic": {"cycles": 1.5},
            "oscillating": {"cycles": 2.0},
            "sequential": {"phase_offset": 0.25, "active_ratio": 0.6},
            "large_motion": {"cycles": 1.0}
        }
        return defaults.get(animation_type, defaults["periodic"])
        
    def load_partnet_object(self, urdf_path: str) -> sapien.Articulation:
        """Load PartNet-Mobility object and store articulation reference."""
        self.articulation = super().load_partnet_object(urdf_path)
        self._analyze_joints()
        return self.articulation
    
    def _analyze_joints(self):
        """分析关节信息"""
        if not self.articulation:
            return
            
        joints = self.articulation.get_joints()
        print(f"\n 关节分析 ")
        print(f"总关节数: {len(joints)}")
        
        for i, joint in enumerate(joints):
            # Get joint type as string
            if hasattr(joint, 'type'):
                joint_type = str(joint.type).split('.')[-1].lower()  # Extract type name
            else:
                joint_type = "unknown"
                
            limits = joint.get_limits()
            print(f"关节 {i}: {joint.name}")
            print(f"  类型: {joint_type}")
            if len(limits) > 0 and len(limits[0]) == 2:
                print(f"  限制: [{limits[0][0]:.3f}, {limits[0][1]:.3f}] rad")
                print(f"  限制: [{np.rad2deg(limits[0][0]):.1f}, {np.rad2deg(limits[0][1]):.1f}] deg")
            else:
                print(f"  限制: 无限制")
    
    def create_simple_animation(self, animation_type: str = "periodic", animation_config: str = "standard") -> Dict[str, Callable]:
        """
        创建简单的关节动画函数
        
        Args:
            animation_type: 动画类型 ("periodic", "oscillating", "sequential", "large_motion")
            animation_config: 动画配置 ("gentle", "standard", "energetic", "extreme")
            
        Returns:
            关节动画函数字典
        """
        if not self.articulation:
            return {}
        
        # 获取动画配置参数
        anim_settings = self._get_animation_settings(animation_config)
        type_settings = self._get_animation_type_settings(animation_type)
        
        joints = self.articulation.get_joints()
        animations = {}
        
        for joint in joints:
            # Check if joint is movable (revolute or prismatic)
            if hasattr(joint, 'type'):
                joint_type_str = str(joint.type).lower()
                is_movable = 'revolute' in joint_type_str or 'prismatic' in joint_type_str
            else:
                # Fallback: check if joint has limits
                limits = joint.get_limits()
                is_movable = len(limits) > 0 and len(limits[0]) == 2
            
            if is_movable:
                limits = joint.get_limits()
                if len(limits) > 0 and len(limits[0]) == 2:
                    limit_min, limit_max = limits[0]
                    
                    # 处理无限制关节：设置合理的默认范围
                    if np.isinf(limit_min) or np.isinf(limit_max):
                        # 对于无限制的旋转关节，使用 ±π (180度) 作为默认范围
                        if 'revolute' in str(joint.type).lower():
                            limit_min = -np.pi  # -180度
                            limit_max = np.pi    # +180度
                            print(f"  无限制关节 {joint.name} 使用默认范围: ±180度")
                        # 对于无限制的滑动关节，使用 ±0.5 作为默认范围
                        elif 'prismatic' in str(joint.type).lower():
                            limit_min = -0.5 # -0.5米
                            limit_max = 0.5  # +0.5米
                            print(f"  无限制关节 {joint.name} 使用默认范围: ±0.5米")

                    if animation_type == "periodic":
                        # 周期性运动：使用配置参数
                        def make_periodic_func(min_val, max_val, settings, type_settings):
                            def periodic_func(t, frame_count):
                                cycles = type_settings.get("cycles", 1.5)
                                angle = np.sin(t * 2 * np.pi * cycles)
                                center = (min_val + max_val) / 2
                                amplitude = (max_val - min_val) * settings["amplitude_ratio"]
                                return center + amplitude * angle / 2
                            return periodic_func
                        animations[joint.name] = make_periodic_func(limit_min, limit_max, anim_settings, type_settings)
                    
                    elif animation_type == "oscillating":
                        # 振荡运动：使用配置参数
                        def make_oscillating_func(min_val, max_val, settings, type_settings):
                            def oscillating_func(t, frame_count):
                                cycles = type_settings.get("cycles", 2.0)
                                center = (min_val + max_val) / 2
                                amplitude = (max_val - min_val) * settings["amplitude_ratio"]
                                return center + amplitude * np.sin(t * 2 * np.pi * cycles)
                            return oscillating_func
                        animations[joint.name] = make_oscillating_func(limit_min, limit_max, anim_settings, type_settings)
                    
                    elif animation_type == "sequential":
                        # 顺序动画：使用配置参数
                        def make_sequential_func(min_val, max_val, joint_index, settings, type_settings):
                            def sequential_func(t, frame_count):
                                phase_offset = type_settings.get("phase_offset", 0.25)
                                active_ratio = type_settings.get("active_ratio", 0.6)
                                
                                phase = (joint_index * phase_offset) % 1.0
                                active_t = (t - phase) % 1.0
                                if 0.0 <= active_t <= active_ratio:
                                    motion_t = active_t / active_ratio
                                    angle = np.sin(motion_t * np.pi)
                                    amplitude = (max_val - min_val) * settings["amplitude_ratio"]
                                    center = (min_val + max_val) / 2
                                    return center + amplitude * (angle - 0.5)
                                else:
                                    return (min_val + max_val) / 2
                            return sequential_func
                        
                        joint_index = len(animations)
                        animations[joint.name] = make_sequential_func(limit_min, limit_max, joint_index, anim_settings, type_settings)
                    
                    elif animation_type == "large_motion":
                        # 大幅度运动：使用配置参数（通常是最大幅度）
                        def make_large_motion_func(min_val, max_val, settings, type_settings):
                            def large_motion_func(t, frame_count):
                                cycles = type_settings.get("cycles", 1.0)
                                angle = np.sin(t * 2 * np.pi * cycles)
                                center = (min_val + max_val) / 2
                                amplitude = (max_val - min_val) * settings["amplitude_ratio"]
                                return center + amplitude * angle / 2
                            return large_motion_func
                        animations[joint.name] = make_large_motion_func(limit_min, limit_max, anim_settings, type_settings)

        print(f"\n=== 动画创建完成 ===")
        print(f"动画类型: {animation_type}")
        print(f"动画配置: {animation_config} (幅度比例: {anim_settings['amplitude_ratio']:.1%}, 频率: {anim_settings['frequency']}x)")
        print(f"可动关节数: {len(animations)}")
        for joint_name in animations.keys():
            print(f"  - {joint_name}")
        
        return animations
    
    def create_custom_animation(self, animation_config: Dict[str, Dict]) -> Dict[str, Callable]:
        """
        创建自定义关节动画
        
        Args:
            animation_config: 关节动画配置
            格式: {
                "joint_name": {
                    "type": "sine",  # sine, linear, step
                    "range": [min_angle, max_angle],  # 运动范围
                    "frequency": 2.0,  # 频率
                    "phase": 0.0,  # 相位偏移
                    "offset": 0.0   # 偏移量
                }
            }
        """
        animations = {}
        
        for joint_name, config in animation_config.items():
            anim_type = config.get("type", "sine")
            angle_range = config.get("range", [0, np.pi/2])
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
                animations[joint_name] = make_sine_func(angle_range[0], angle_range[1], frequency, phase, offset)
            
            elif anim_type == "linear":
                def make_linear_func(min_angle, max_angle, freq, ph, off):
                    def linear_func(t, frame_count):
                        # 线性往返运动
                        cycle_t = (freq * t + ph / (2 * np.pi)) % 1.0
                        if cycle_t <= 0.5:
                            progress = cycle_t * 2  # 0 to 1
                        else:
                            progress = 2 - cycle_t * 2  # 1 to 0
                        return min_angle + (max_angle - min_angle) * progress + off
                    return linear_func
                animations[joint_name] = make_linear_func(angle_range[0], angle_range[1], frequency, phase, offset)
        
        return animations
    
    def set_joint_animations(self, animations: Dict[str, Callable]):
        """设置关节动画函数"""
        self.joint_animations = animations
    
    def animate_joints(self, t: float, frame_count: int):
        """
        在指定时间应用关节动画
        
        Args:
            t: 归一化时间 (0 到 1)
            frame_count: 总帧数
        """
        if not self.articulation or not self.joint_animations:
            return
        
        # 获取当前关节位置
        qpos = self.articulation.get_qpos().copy()
        joints_list = self.articulation.get_joints()
        
        # 创建关节名称到索引的映射
        joint_name_to_idx = {}
        movable_joint_idx = 0
        
        for i, joint in enumerate(joints_list):
            # 检查关节是否可动（非fixed类型）
            if hasattr(joint, 'type'):
                joint_type_str = str(joint.type).lower()
                is_movable = 'revolute' in joint_type_str or 'prismatic' in joint_type_str
            else:
                # 备用检查：看是否有限制
                limits = joint.get_limits()
                is_movable = len(limits) > 0 and len(limits[0]) == 2
            
            if is_movable and joint.name:
                joint_name_to_idx[joint.name] = movable_joint_idx
                movable_joint_idx += 1
        
        # 应用动画到每个关节
        animated_joints = []
        for joint_name, animation_func in self.joint_animations.items():
            if joint_name in joint_name_to_idx:
                target_angle = animation_func(t, frame_count)
                joint_idx = joint_name_to_idx[joint_name]
                
                if joint_idx < len(qpos):
                    qpos[joint_idx] = target_angle
                    animated_joints.append(f"{joint_name}={target_angle:.3f}")
        
        # 设置新的关节位置
        if animated_joints:
            try:
                self.articulation.set_qpos(qpos)
                # 重要：步进仿真以更新物理状态
                self.scene.step()
                
                # 调试信息（只在前几帧打印）
                if self.debug and (t == 0.0 or (frame_count > 1 and abs(t - 0.5) < 0.01)):
                    print(f"Frame {int(t * frame_count)}: 关节动画 - {', '.join(animated_joints)}")
                    
            except Exception as e:
                print(f"关节动画设置失败: {e}")
                print(f"  qpos长度: {len(qpos)}, 关节映射: {joint_name_to_idx}")
    
    def render_animated_sequence(self, camera_poses: List[sapien.Pose], 
                               animations: Optional[Dict[str, Callable]] = None,
                               animation_type: str = "periodic",
                               animation_config: str = "standard",
                               save_frames: bool = True, 
                               output_dir: str = "animated_output") -> None:
        """
        渲染带关节动画的序列
        
        Args:
            camera_poses: 相机姿态列表
            animations: 自定义动画函数 (可选)
            animation_type: 动画类型 ("periodic", "oscillating", "sequential", "large_motion")
            animation_config: 动画配置 ("gentle", "standard", "energetic", "extreme")
            save_frames: 是否保存帧
            output_dir: 输出目录
        """
        # 设置动画
        if animations is None:
            animations = self.create_simple_animation(animation_type, animation_config)
        
        self.set_joint_animations(animations)
        
        # 创建输出目录
        if save_frames:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/rgb", exist_ok=True)
            os.makedirs(f"{output_dir}/depth", exist_ok=True)
            
        self.rgb_frames = []
        self.depth_frames = []
        self.camera_params = []
        joint_states = []
        
        print(f"渲染 {len(camera_poses)} 帧动画...")
        
        for i, pose in enumerate(camera_poses):
            # 计算时间进度
            t = i / (len(camera_poses) - 1) if len(camera_poses) > 1 else 0.0
            
            # 应用关节动画
            self.animate_joints(t, len(camera_poses))
            
            # 设置相机姿态
            self.camera_mount.set_pose(pose)
            
            # 更新场景渲染状态
            self.scene.update_render()
            
            # 记录关节状态
            joint_state = {
                'frame': i,
                'time': t,
                'qpos': self.articulation.get_qpos().tolist() if self.articulation else [],
                'qvel': self.articulation.get_qvel().tolist() if self.articulation else []
            }
            joint_states.append(joint_state)
            
            # 捕获帧
            rgb, depth, params = self.capture_frame()
            
            # 存储帧
            self.rgb_frames.append(rgb)
            self.depth_frames.append(depth)
            self.camera_params.append(params)
            
            # 保存帧
            if save_frames:
                # 保存RGB
                from PIL import Image
                rgb_pil = Image.fromarray(rgb)
                rgb_pil.save(f"{output_dir}/rgb/frame_{i:06d}.png")
                
                # 保存深度
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_normalized * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img)
                depth_pil.save(f"{output_dir}/depth/frame_{i:06d}.png")
                
                # 保存原始深度
                np.save(f"{output_dir}/depth/frame_{i:06d}.npy", depth)
                
            if (i + 1) % 10 == 0:
                print(f"渲染进度: {i + 1}/{len(camera_poses)} 帧")
                
        # 保存相机参数和关节状态
        if save_frames:
            with open(f"{output_dir}/camera_params.json", 'w') as f:
                json.dump(self.camera_params, f, indent=2)
                
            with open(f"{output_dir}/joint_states.json", 'w') as f:
                json.dump(joint_states, f, indent=2)
                
        print("动画渲染完成！")


def create_example_animation_config():
    """创建示例动画配置"""
    return {
        # 示例：抽屉动画
        "drawer_joint": {
            "type": "sine",
            "range": [0.0, 0.5],  # 0到0.5米
            "frequency": 1.0,
            "phase": 0.0,
            "offset": 0.0
        },
        
        # 示例：门动画
        "door_joint": {
            "type": "sine", 
            "range": [0.0, np.pi/2],  # 0到90度
            "frequency": 0.8,
            "phase": np.pi/4,
            "offset": 0.0
        },
        
        # 示例：旋转关节
        "rotation_joint": {
            "type": "linear",
            "range": [-np.pi/4, np.pi/4],  # -45到45度
            "frequency": 2.0,
            "phase": 0.0,
            "offset": 0.0
        }
    }


def main():
    """示例用法"""
    # 创建动画渲染器
    renderer = AnimatedRenderer(width=640, height=480, fps=30)
    
    # 加载PartNet-Mobility对象
    urdf_path = "../assets/179/mobility.urdf"  # 替换为你的URDF路径
    
    try:
        asset = renderer.load_partnet_object(urdf_path)
        
        # 生成相机轨迹
        center = np.array([0, 0, 0.5])
        poses = renderer.generate_circular_trajectory(
            center=center,
            radius=2.0,
            height=1.5,
            n_frames=120,
            full_rotation=True
        )
        
        # 方式1：使用预设动画类型
        print("\n=== 使用周期性动画 ===")
        renderer.render_animated_sequence(
            poses, 
            animation_type="periodic",
            output_dir="animated_periodic"
        )
        
        # 方式2：使用自定义动画配置
        print("\n=== 使用自定义动画 ===")
        custom_animations = renderer.create_custom_animation(create_example_animation_config())
        renderer.render_animated_sequence(
            poses,
            animations=custom_animations,
            output_dir="animated_custom"
        )
        
        # 创建视频
        renderer.create_videos("animated_periodic", "periodic_rgb.mp4", "periodic_depth.mp4")
        renderer.create_videos("animated_custom", "custom_rgb.mp4", "custom_depth.mp4")
        
        print("\n=== 渲染完成 ===")
        print("输出目录:")
        print("  - animated_periodic/: 周期性动画")
        print("  - animated_custom/: 自定义动画")
        print("文件包含:")
        print("  - RGB/depth图片序列")
        print("  - camera_params.json: 相机参数")
        print("  - joint_states.json: 关节状态")
        print("  - *.mp4: 视频文件")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查URDF路径和资源文件")


if __name__ == "__main__":
    main()
