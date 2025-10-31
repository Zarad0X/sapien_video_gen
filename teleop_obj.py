import argparse
import sys
import os
import json
from pathlib import Path
from typing import List
import numpy as np
import sapien.core as sapien
from vr_bridge import VRBridge
import rospy
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as R
from render_partnet_video import PartNetVideoRenderer
import copy

class TeleopObject:
    def __init__(self, urdf_path: str = "", width: int = 1280, height: int = 720, fps: int = 60, background_color=(1.0, 1.0, 1.0), output_dir: str = "test"):
        self.vr_bridge = VRBridge()
        self.sim = PartNetVideoRenderer(width=width, height=height, fps=fps, background_color=background_color)
        self.sim.load_partnet_object(urdf_path)
        # output setup
        self.output_dir = Path(output_dir)
        self.rgb_log = []
        self.depth_log = []
        self.mask_log = []
        self.camera_params_log = []
        self.object_poses_log = []
        self.joint_states_log = []

        self.delta_qpos = 0.04
        self.joint_index = 2

        self.record_start_trigger = False
        
    def start(self):
        pbar = tqdm(total=None)
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "rgb", exist_ok=True)
        os.makedirs(self.output_dir / "depth", exist_ok=True)
        os.makedirs(self.output_dir / "masks", exist_ok=True)
        os.makedirs(self.output_dir / "vis", exist_ok=True)
        # save camera intrinsics once
        try:
            np.savetxt(self.output_dir / "cam_K.txt", self.sim.intrinsic_matrix, fmt='%.6f')
        except Exception:
            pass
        cam_pos_mat = np.eye(4)
        obj_pos_mat = np.eye(4)
        obj_pos_mat[0, 3] = 3.0
        cam_pose_sapien = sapien.Pose.from_transformation_matrix(cam_pos_mat)
        obj_pos_sapien = sapien.Pose.from_transformation_matrix(obj_pos_mat)
        self.sim.camera_mount.set_pose(cam_pose_sapien)
        self.sim.asset.set_pose(obj_pos_sapien)
        cnt = 0
        current_pose = copy.deepcopy(obj_pos_sapien)
        new_pose = copy.deepcopy(obj_pos_sapien)
        while not rospy.is_shutdown():
            hand_trigger = rospy.get_param("/vr/hand_trigger")
            pos = self.vr_bridge.transmat[:3, 3]
            rot = self.vr_bridge.transmat[:3, :3]
            if hand_trigger and not np.all(np.isclose(pos, np.array([0, 0, 0]))):
                
                if self.vr_bridge.keyone_trigger:
                    qpos = self.sim.asset.get_joints()[self.joint_index].articulation.get_qpos()
                    qpos[0] += self.delta_qpos
                    self.sim.asset.get_joints()[self.joint_index].articulation.set_qpos(qpos)
                if self.vr_bridge.keytwo_trigger:
                    qpos = self.sim.asset.get_joints()[self.joint_index].articulation.get_qpos()
                    qpos[0] -= self.delta_qpos
                    self.sim.asset.get_joints()[self.joint_index].articulation.set_qpos(qpos)
                pos_new = current_pose.p + pos
                rot_ori = R.from_quat(current_pose.q[[1,2,3,0]]) # wxyz --> xyzw
                rot_new = R.from_matrix(rot) * rot_ori
                rot_new_quat = rot_new.as_quat()[[3,0,1,2]]
                new_pose = sapien.Pose(p=pos_new, q=rot_new_quat)
                self.sim.asset.set_pose(new_pose)
            else:
                current_pose = new_pose
                
            rgb, depth, camera_params = self.sim.capture_frame()            
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("test", rgb)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.record_start_trigger = not self.record_start_trigger
                print(f"record_start_trigger: {self.record_start_trigger}")
            elif key == ord('s'):
                self.save_logs()
            elif key == ord('c'):
                self.clear_logs()

            if self.record_start_trigger and hand_trigger: # save per-frame data
                self.rgb_log.append(rgb)
                self.depth_log.append(depth)
                mask = (depth > 0.000001).astype(np.uint8) * 255
                self.mask_log.append(mask)

                self.camera_params_log.append(camera_params)

                obj_pose = self.sim.asset.get_pose()
                self.object_poses_log.append({
                    'position': obj_pose.p.tolist(),
                    'quaternion': obj_pose.q.tolist()
                })
                qpos = self.sim.asset.get_joints()[self.joint_index].articulation.get_qpos().tolist()
                self.joint_states_log.append(qpos)
            
            cnt += 1
            pbar.update(1)

    # rgb, depth, mask, obj pose, joint angle, cam pose, cam intr,+
    def save_logs(self):
        max_length = 300
        interval = len(self.rgb_log) // max_length
        index = 0
        for i in range(0, len(self.rgb_log), interval):
            
            rgb = self.rgb_log[i]
            depth = self.depth_log[i]
            mask = self.mask_log[i]            
            depth_vis = (depth / depth.max() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_dir, "rgb", f"{index:05d}.png"), rgb)
            np.savez_compressed(os.path.join(self.output_dir, "depth", f"{index:05d}.npz"), depth=depth)
            cv2.imwrite(os.path.join(self.output_dir, "vis", f"{index:05d}.png"), depth_vis)
            cv2.imwrite(os.path.join(self.output_dir, "masks", f"{index:05d}.png"), mask)
            index += 1            
        with open(os.path.join(self.output_dir, "object_poses.json"), 'w') as f:
            json.dump(self.object_poses_log[::interval], f, indent=2)
        with open(os.path.join(self.output_dir, "joint_states.json"), 'w') as f:
            json.dump(self.joint_states_log[::interval], f, indent=2)
        with open(os.path.join(self.output_dir, "camera_params.json"), 'w') as f:
            json.dump(self.camera_params_log[::interval], f, indent=2)

        self.clear_logs()
        self.record_start_trigger = False
    def clear_logs(self):
        self.rgb_log = []
        self.depth_log = []
        self.mask_log = []
        self.object_poses_log = []
        self.joint_states_log = []
        self.camera_params_log = []
        self.record_start_trigger = False
if __name__ == "__main__":
    urdf_path = os.environ.get("urdf_path", "")
    object_name = urdf_path.split("/")[-2]
    teleop = TeleopObject(urdf_path=urdf_path, width=1280, height=720, output_dir=f"examples/{object_name}")
    teleop.start()