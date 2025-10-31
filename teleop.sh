export urdf_path=${urdf_path:-"/home/hwfan/workspace/partnet-mobility-v0/dataset/100109/mobility.urdf"}
unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash
python teleop_obj.py