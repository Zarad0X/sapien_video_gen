import json
import os
import numpy as np
from typing import List, Dict, Callable, Optional

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
        print(f"\n=== 关节分析 ===")
        print(f"总关节数: {len(joints)}")
        
        movable_joints = []
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
                print(f"  限制: [{limits[0][0]:.3f}, {limits[0][1]:.3f}]")
                movable_joints.append(joint.name)
            else:
                print(f"  限制: 无限制")
        
        print(f"\n=== 可动关节列表 ===")
        for joint_name in movable_joints:
            print(f"  - '{joint_name}'")
        
        print(f"\n=== 动画配置说明 ===")
        print("默认模式：自动为所有关节分配不同的动画")
        print("手动模式：可以指定特定关节的动画配置")
        print("示例手动配置：")
        print("  config = {'joint_0': {'type': 'sine', 'frequency': 2.0}}")
        print("  animations = renderer.create_custom_animation(config)")
        print("示例纯自动配置：")
        print("  animations = renderer.create_custom_animation()")  # 使用默认参数
    
    def create_custom_animation(self, animation_config: Optional[Dict[str, Dict]] = None, 
                               auto_assign: bool = True) -> Dict[str, Callable]:
        """
        创建关节动画配置
        
        Args:
            animation_config: 手动指定的关节动画配置（可选）
            auto_assign: 是否自动为未指定的关节分配默认动画
        
        Returns:
            关节动画函数字典
        """
        animations = {}
        
        # 获取所有可动关节
        movable_joints = self._get_movable_joints()
        
        # 如果没有提供配置或需要自动分配，创建默认配置
        if animation_config is None:
            animation_config = {}
        
        # 处理手动指定的关节配置
        for joint_name, config in animation_config.items():
            if joint_name in movable_joints:
                animations[joint_name] = self._create_animation_function(config, movable_joints[joint_name])
        
        # 自动为未指定的关节分配默认动画
        if auto_assign:
            assigned_joints = set(animation_config.keys())
            unassigned_joints = [name for name in movable_joints.keys() if name not in assigned_joints]
            
            for i, joint_name in enumerate(unassigned_joints):
                default_config = self._get_default_animation_config(i, movable_joints[joint_name])
                animations[joint_name] = self._create_animation_function(default_config, movable_joints[joint_name])
        
        return animations
    
    def _get_movable_joints(self) -> Dict[str, Dict]:
        """获取所有可动关节及其信息"""
        if not self.articulation:
            return {}
        
        movable_joints = {}
        joints = self.articulation.get_joints()
        
        for joint in joints:
            # 检查关节是否可动
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
        """为关节生成默认动画配置"""
        joint_type = joint_info['type']
        limits = joint_info['limits']
        
        # 预定义的动画模式
        animation_patterns = [
            {"type": "sine", "frequency": 1.0, "phase": 0.0},
            {"type": "linear", "frequency": 0.5, "phase": 0.0},
            {"type": "sine", "frequency": 1.5, "phase": np.pi/2},
            {"type": "sine", "frequency": 0.8, "phase": np.pi},
            {"type": "linear", "frequency": 0.7, "phase": np.pi/4},
            {"type": "sine", "frequency": 1.2, "phase": 3*np.pi/2}
        ]
        
        # 循环使用动画模式
        pattern = animation_patterns[index % len(animation_patterns)]
        
        # 根据关节类型调整范围
        if joint_type == 'prismatic':
            # 移动关节：使用实际限制的80%
            range_span = limits[1] - limits[0]
            safe_range = range_span * 0.8
            center = (limits[0] + limits[1]) / 2
            animation_range = [center - safe_range/2, center + safe_range/2]
        else:
            # 旋转关节：使用实际限制的80%
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
        """根据配置创建动画函数"""
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
                    # 线性往返运动
                    cycle_t = (freq * t + ph / (2 * np.pi)) % 1.0
                    if cycle_t <= 0.5:
                        progress = cycle_t * 2  # 0 to 1
                    else:
                        progress = 2 - cycle_t * 2  # 1 to 0
                    return min_angle + (max_angle - min_angle) * progress + off
                return linear_func
            return make_linear_func(angle_range[0], angle_range[1], frequency, phase, offset)
        
        else:
            # 默认返回静止函数
            def static_func(t, frame_count):
                return (angle_range[0] + angle_range[1]) / 2
            return static_func
    
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
            # 检查关节是否可动
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
            except Exception as e:
                print(f"关节动画设置失败: {e}")
                print(f"  qpos长度: {len(qpos)}, 关节映射: {joint_name_to_idx}")
    
    def render_animated_sequence(self, camera_poses: List[sapien.Pose], 
                               animations: Dict[str, Callable],
                               save_frames: bool = True, 
                               output_dir: str = "animated_output") -> None:
    
        # 设置动画
        self.set_joint_animations(animations)
        
        # 创建输出目录
        if save_frames:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}/rgb", exist_ok=True)
            os.makedirs(f"{output_dir}/depth", exist_ok=True)  # 原始深度数据
            os.makedirs(f"{output_dir}/vis", exist_ok=True)    # 深度图可视化
            
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
                rgb_pil.save(f"{output_dir}/rgb/{i:05d}.png")
                
                # 保存原始深度数据
                np.savez_compressed(f"{output_dir}/depth/{i:05d}.npz", depth=depth)
                
                # 保存深度图可视化到vis文件夹
                valid_depth = depth.copy()
                
                # 归一化深度图用于可视化
                depth_min, depth_max = valid_depth.min(), valid_depth.max()
                if depth_max > depth_min:
                    depth_normalized = (valid_depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(valid_depth)
                
                # 使用彩色映射生成彩色深度图
                import matplotlib.cm as cm
                colormap = cm.viridis  # 使用viridis色彩映射
                depth_colored = colormap(depth_normalized)
                
                # 转换为RGB格式（移除alpha通道）并转为uint8
                depth_img_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img_rgb)
                depth_pil.save(f"{output_dir}/vis/{i:05d}.png")
                
            if (i + 1) % 10 == 0:
                print(f"渲染进度: {i + 1}/{len(camera_poses)} 帧")
                
        # 保存相机参数和关节状态
        if save_frames:
            # 保存相机外参到JSON文件
            with open(f"{output_dir}/camera_params.json", 'w') as f:
                json.dump(self.camera_params, f, indent=2)
            
            # 保存相机内参到txt文件
            np.savetxt(f"{output_dir}/cam_K.txt", self.intrinsic_matrix, fmt='%.6f')
                
            # 保存关节状态
            with open(f"{output_dir}/joint_states.json", 'w') as f:
                json.dump(joint_states, f, indent=2)
                
        print("动画渲染完成！")


def create_example_animation_config():
    """
    创建示例动画配置 - 演示手动配置特定关节
    返回部分关节的手动配置，其他关节将自动分配
    """
    return {
        # 只手动配置前两个关节，其他关节会自动分配
        "joint_0": {
            "type": "sine",
            "range": [0.0, 0.3],  # 可以手动调整范围
            "frequency": 2.0,     # 快速运动
            "phase": 0.0,
            "offset": 0.0
        },
        
        "joint_1": {
            "type": "linear",
            "frequency": 0.5,     # 慢速线性运动
            "phase": 0.0,
            "offset": 0.0
            # 注意：没有指定range，将使用关节的安全范围（限制的80%）
        }
        
        # joint_2, joint_3 等将自动分配不同的动画模式
    }


def create_full_manual_config():
    """创建完整手动配置的示例 - 所有关节都手动指定"""
    return {
        "joint_0": {
            "type": "sine",
            "range": [0.0, 0.368],
            "frequency": 1.0,
            "phase": 0.0,
            "offset": 0.0
        },
        
        "joint_1": {
            "type": "linear",
            "range": [0.0, 0.368],
            "frequency": 0.5,
            "phase": 0.0,
            "offset": 0.0
        },
        
        "joint_2": {
            "type": "sine",
            "range": [0.0, 0.368],
            "frequency": 1.5,
            "phase": np.pi/2,
            "offset": 0.0
        },
        
        "joint_3": {
            "type": "sine",
            "range": [0.0, 0.368],
            "frequency": 0.8,
            "phase": np.pi,
            "offset": 0.0
        }
    }


