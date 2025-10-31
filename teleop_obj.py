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
    def __init__(self, urdf_path: str = "", width: int = 1280, height: int = 720, fps: int = 60, background_color=(1.0, 1.0, 1.0)):
        self.vr_bridge = VRBridge()
        self.sim = PartNetVideoRenderer(width=width, height=height, fps=fps, background_color=background_color)
        self.sim.load_partnet_object(urdf_path)
        
    def start(self):
        pbar = tqdm(total=None)
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        os.makedirs("test", exist_ok=True)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cnt += 1
            pbar.update(1)
            
    # rgb, depth, mask, obj pose, joint angle, cam pose, cam intr,
if __name__ == "__main__":
    urdf_path = os.environ.get("urdf_path", "")
    teleop = TeleopObject(urdf_path=urdf_path, width=1280, height=720)
    teleop.start()