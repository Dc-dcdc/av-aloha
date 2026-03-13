import numpy as np
from headset_utils import (
    convert_right_to_left_coordinates,
    HeadsetFeedback
)
from transform_utils import (
    align_rotation_to_z_axis, 
    within_pose_threshold, 
    quat2mat,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
    pose2mat,
    mat2pose,
    transform_coordinates,
)
from constants import REAL_DT
import time
import os

DT = REAL_DT

class HeadsetControl():
    def __init__(
            self,
            # start_ctrl_position_threshold=0.03,
            # start_ctrl_rotation_threshold=0.2,
            start_head_position_threshold=0.03,
            start_head_rotation_threshold=0.2,
            # ctrl_position_threshold=0.04,
            # ctrl_rotation_threshold=0.3,
            head_position_threshold=0.05,
            head_rotation_threshold=0.3
        ):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

        # self.start_ctrl_position_threshold = start_ctrl_position_threshold
        # self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold
        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold
        # self.ctrl_position_threshold = ctrl_position_threshold
        # self.ctrl_rotation_threshold = ctrl_rotation_threshold
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
    
    def reset(self):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

    def is_running(self):
        return self.started

    def start(self, headset_data, middle_arm_pose):
        aligned_headset_pose = np.eye(4)
        aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat))
        aligned_headset_pose[:3, 3] = headset_data.h_pos
        self.start_headset_pose = aligned_headset_pose

        aligned_middle_arm_pose = np.eye(4)
        aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(wxyz_to_xyzw(middle_arm_pose[3:])))
        aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3]
        self.start_middle_arm_pose = aligned_middle_arm_pose

        self.started = True

    def run(self, headset_data, middle_arm_pose):
        middle_arm_pose = pose2mat(middle_arm_pose[:3], wxyz_to_xyzw(middle_arm_pose[3:]))
        # left_arm_pose = pose2mat(left_arm_pose[:3], wxyz_to_xyzw(left_arm_pose[3:]))
        # right_arm_pose = pose2mat(right_arm_pose[:3], wxyz_to_xyzw(right_arm_pose[3:]))
        headset_pose = pose2mat(headset_data.h_pos, headset_data.h_quat)
        # left_pose = pose2mat(headset_data.l_pos, headset_data.l_quat)
        # right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat)

        if self.started:
            start_headset_pose = self.start_headset_pose
            start_middle_arm_pose = self.start_middle_arm_pose
        else:
            aligned_headset_pose = np.eye(4)
            aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(headset_pose[:3, :3])
            aligned_headset_pose[:3, 3] = headset_pose[:3, 3]
            start_headset_pose = aligned_headset_pose

            aligned_middle_arm_pose = np.eye(4)
            aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
            aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
            start_middle_arm_pose = aligned_middle_arm_pose

        # calculate offset between current and saved headset pose
        new_middle_arm_pose = transform_coordinates(headset_pose, start_headset_pose, start_middle_arm_pose)
        # new_left_arm_pose = transform_coordinates(left_pose, start_headset_pose, start_middle_arm_pose)
        # new_right_arm_pose = transform_coordinates(right_pose, start_headset_pose, start_middle_arm_pose)

        # convert to position and quaternion
        new_middle_arm_pos, new_middle_arm_quat = mat2pose(new_middle_arm_pose)
        # new_left_arm_pos, new_left_arm_quat = mat2pose(new_left_arm_pose)
        # new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_pose)
        new_middle_arm_quat = xyzw_to_wxyz(new_middle_arm_quat)
        # new_left_arm_quat = xyzw_to_wxyz(new_left_arm_quat)
        # new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)

        # grippers 
        new_left_gripper = np.array([headset_data.l_index_trigger])
        new_right_gripper = np.array([headset_data.r_index_trigger])

        # concatenate the new action
        # action = np.concatenate([
        #     new_left_arm_pos, new_left_arm_quat, new_left_gripper,
        #     new_right_arm_pos, new_right_arm_quat, new_right_gripper,
        #     new_middle_arm_pos, new_middle_arm_quat
        # ])

        headset_action = np.concatenate([
            new_middle_arm_pos, new_middle_arm_quat
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        unity_middle_arm_pose = transform_coordinates(middle_arm_pose, start_middle_arm_pose, start_headset_pose)
        # unity_left_arm_pose = transform_coordinates(left_arm_pose, start_middle_arm_pose, start_headset_pose)
        # unity_right_arm_pose = transform_coordinates(right_arm_pose, start_middle_arm_pose, start_headset_pose)        

        # unity_left_arm_pos, unity_left_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_left_arm_pose))
        # unity_right_arm_pos, unity_right_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_right_arm_pose))
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_pose))
        
        headOutOfSync = not within_pose_threshold(
            middle_arm_pose[:3, 3],
            middle_arm_pose[:3, :3],
            new_middle_arm_pose[:3, 3], 
            new_middle_arm_pose[:3, :3],
            self.head_position_threshold if self.started else self.start_head_position_threshold,
            self.head_rotation_threshold if self.started else self.start_head_rotation_threshold
        )
        # leftOutOfSync = not within_pose_threshold(
        #     left_arm_pose[:3, 3],
        #     left_arm_pose[:3, :3],
        #     new_left_arm_pose[:3, 3], 
        #     new_left_arm_pose[:3, :3],
        #     self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
        #     self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        # )
        # rightOutOfSync = not within_pose_threshold(
        #     right_arm_pose[:3, 3],
        #     right_arm_pose[:3, :3],
        #     new_right_arm_pose[:3, 3], 
        #     new_right_arm_pose[:3, :3],
        #     self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
        #     self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        # )

        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = headOutOfSync
        feedback.left_out_of_sync = False
        feedback.right_out_of_sync = False
        feedback.left_arm_position = np.zeros(3)
        feedback.left_arm_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        feedback.right_arm_position = np.zeros(3)
        feedback.right_arm_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat        

        return headset_action, feedback
    
# 
class HeadsetFullControl():
    def __init__(
            self,
            start_ctrl_position_threshold=0.06, #控制器起始位置阈值
            start_ctrl_rotation_threshold=0.4, #控制器起始旋转阈值
            start_head_position_threshold=0.03, #头部起始位置阈值
            start_head_rotation_threshold=0.2, #头部起始旋转阈值
            ctrl_position_threshold=0.04, #控制器位置阈值
            ctrl_rotation_threshold=0.3, #控制器旋转阈值
            head_position_threshold=0.05, #头部位置阈值
            head_rotation_threshold=0.3 #头部旋转阈值
        ):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

        self.start_ctrl_position_threshold = start_ctrl_position_threshold
        self.start_ctrl_rotation_threshold = start_ctrl_rotation_threshold
        self.start_head_position_threshold = start_head_position_threshold
        self.start_head_rotation_threshold = start_head_rotation_threshold
        self.ctrl_position_threshold = ctrl_position_threshold
        self.ctrl_rotation_threshold = ctrl_rotation_threshold
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
    
    def reset(self):
        self.start_middle_arm_pose = None
        self.start_headset_pose = None
        self.started = False

    def is_running(self):
        return self.started

    # 保存当前的头戴设备位姿和中间臂位姿作为起始位姿，并将started标志设置为True，表示已经开始执行轨迹
    def start(self, headset_data, middle_arm_pose):
        # 对齐头戴设备的位姿
        aligned_headset_pose = np.eye(4) # 创建一个4x4的单位矩阵，表示对齐后的头戴设备位姿
        aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat)) # 将头戴设备的旋转对齐到z轴
        aligned_headset_pose[:3, 3] = headset_data.h_pos # 将头戴设备的位置设置为对齐后的位姿的平移部分
        self.start_headset_pose = aligned_headset_pose

        # 对齐中间臂的位姿
        aligned_middle_arm_pose = np.eye(4)
        aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(quat2mat(wxyz_to_xyzw(middle_arm_pose[3:])))
        aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3]
        self.start_middle_arm_pose = aligned_middle_arm_pose

        self.started = True # 设置started标志为True，表示已经开始执行轨迹

    # 
    def run(self, headset_data, left_arm_pose, right_arm_pose, middle_arm_pose):
        # 将输入的头戴设备数据和左右臂、中间臂的位姿转换为4x4的变换矩阵，以便后续计算
        middle_arm_pose = pose2mat(middle_arm_pose[:3], wxyz_to_xyzw(middle_arm_pose[3:])) # 中间臂
        left_arm_pose = pose2mat(left_arm_pose[:3], wxyz_to_xyzw(left_arm_pose[3:])) # 左臂
        right_arm_pose = pose2mat(right_arm_pose[:3], wxyz_to_xyzw(right_arm_pose[3:])) # 右臂
        
        headset_pose = pose2mat(headset_data.h_pos, headset_data.h_quat) # 头戴设备
        left_pose = pose2mat(headset_data.l_pos, headset_data.l_quat) # 左手控制器
        right_pose = pose2mat(headset_data.r_pos, headset_data.r_quat) # 右手控制器

        # 如果已经开始执行轨迹，则使用保存的起始位姿，否则将当前的头戴设备位姿和中间臂位姿对齐到z轴并保存为起始位姿
        if self.started: 
            start_headset_pose = self.start_headset_pose
            start_middle_arm_pose = self.start_middle_arm_pose
        else:
            aligned_headset_pose = np.eye(4)
            aligned_headset_pose[:3, :3] = align_rotation_to_z_axis(headset_pose[:3, :3])
            aligned_headset_pose[:3, 3] = headset_pose[:3, 3]
            start_headset_pose = aligned_headset_pose

            aligned_middle_arm_pose = np.eye(4)
            aligned_middle_arm_pose[:3, :3] = align_rotation_to_z_axis(middle_arm_pose[:3, :3])
            aligned_middle_arm_pose[:3, 3] = middle_arm_pose[:3, 3]
            start_middle_arm_pose = aligned_middle_arm_pose

        # calculate offset between current and saved headset pose
        # 根据当前的头戴设备位姿和保存的起始位姿计算出新的中间臂位姿，以及新的左右臂位姿
        new_middle_arm_pose = transform_coordinates(headset_pose, start_headset_pose, start_middle_arm_pose)#提取头显相对于初始姿态的空间偏移量，并将其等效映射至中间臂的初始坐标系，从而解算出中间臂的当前期望位姿
        new_left_arm_pose = transform_coordinates(left_pose, start_headset_pose, start_middle_arm_pose)
        new_right_arm_pose = transform_coordinates(right_pose, start_headset_pose, start_middle_arm_pose)

        # convert to position and quaternion
        # 将新的中间臂位姿和左右臂位姿转换为位置和四元数的形式
        new_middle_arm_pos, new_middle_arm_quat = mat2pose(new_middle_arm_pose)
        new_left_arm_pos, new_left_arm_quat = mat2pose(new_left_arm_pose)
        new_right_arm_pos, new_right_arm_quat = mat2pose(new_right_arm_pose)
        # 将四元数从xyzw格式转换为wxyz格式，以便与环境中的位姿表示一致
        new_middle_arm_quat = xyzw_to_wxyz(new_middle_arm_quat)
        new_left_arm_quat = xyzw_to_wxyz(new_left_arm_quat)
        new_right_arm_quat = xyzw_to_wxyz(new_right_arm_quat)

        # grippers 
        # 将左右手控制器的扳机值作为夹持器的状态，拼接成一个数组
        new_left_gripper = np.array([headset_data.l_index_trigger]) 
        new_right_gripper = np.array([headset_data.r_index_trigger])

        # concatenate the new action
        action = np.concatenate([
            new_left_arm_pos, new_left_arm_quat, new_left_gripper,
            new_right_arm_pos, new_right_arm_quat, new_right_gripper,
            new_middle_arm_pos, new_middle_arm_quat
        ])

        # transform middle_arm_pose from mujoco coords to start_headset_pose coords
        # 将中间臂的位姿从mujoco坐标系转换到以start_headset_pose为坐标系的坐标系下，以便与环境中的位姿表示一致
        unity_middle_arm_pose = transform_coordinates(middle_arm_pose, start_middle_arm_pose, start_headset_pose) 
        unity_left_arm_pose = transform_coordinates(left_arm_pose, start_middle_arm_pose, start_headset_pose)
        unity_right_arm_pose = transform_coordinates(right_arm_pose, start_middle_arm_pose, start_headset_pose)        

        # 将转换后的中间臂位姿和左右臂位姿转换为位置和四元数的形式，并将四元数从xyzw格式转换为wxyz格式，以便与环境中的位姿表示一致
        unity_left_arm_pos, unity_left_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_left_arm_pose))
        unity_right_arm_pos, unity_right_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_right_arm_pose))
        unity_middle_arm_pos, unity_middle_arm_quat = convert_right_to_left_coordinates(*mat2pose(unity_middle_arm_pose))
        
        # 根据新的中间臂位姿和左右臂位姿与环境中的位姿进行比较，判断它们是否在预设的阈值范围内，如果超出阈值范围则认为不同步，并将结果保存在feedback中以便发送给头戴设备
        headOutOfSync = not within_pose_threshold(
            middle_arm_pose[:3, 3],
            middle_arm_pose[:3, :3],
            new_middle_arm_pose[:3, 3], 
            new_middle_arm_pose[:3, :3],
            self.head_position_threshold if self.started else self.start_head_position_threshold,
            self.head_rotation_threshold if self.started else self.start_head_rotation_threshold
        )
        leftOutOfSync = not within_pose_threshold(
            left_arm_pose[:3, 3],
            left_arm_pose[:3, :3],
            new_left_arm_pose[:3, 3], 
            new_left_arm_pose[:3, :3],
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        )
        rightOutOfSync = not within_pose_threshold(
            right_arm_pose[:3, 3],
            right_arm_pose[:3, :3],
            new_right_arm_pose[:3, 3], 
            new_right_arm_pose[:3, :3],
            self.ctrl_position_threshold if self.started else self.start_ctrl_position_threshold,
            self.ctrl_rotation_threshold if self.started else self.start_ctrl_rotation_threshold
        )
        # 创建一个HeadsetFeedback对象，并将不同步的结果、反馈信息字符串以及转换后的左右臂和中间臂的位姿保存在其中，以便发送给头戴设备
        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = headOutOfSync
        feedback.left_out_of_sync = leftOutOfSync
        feedback.right_out_of_sync = rightOutOfSync
        feedback.left_arm_position = unity_left_arm_pos
        feedback.left_arm_rotation = unity_left_arm_quat
        feedback.right_arm_position = unity_right_arm_pos
        feedback.right_arm_rotation = unity_right_arm_quat
        feedback.middle_arm_position = unity_middle_arm_pos
        feedback.middle_arm_rotation = unity_middle_arm_quat        

        return action, feedback # 返回生成的动作和反馈信息，以便在环境中执行动作并将反馈发送给头戴设备
    
    
# def main():
#     from real_env import RealEnv
#     from webrtc_headset import WebRTCHeadset

#     # setup the headset
#     headset = WebRTCHeadset()
#     headset.run_in_thread()
#     # setup the headset control
#     headset_control = HeadsetControl()

#     # setup the environment
#     env = RealEnv(init_node=True, headset=headset)
#     observation, info = env.reset(seed=42)

#     # setup the initial action
#     init_action = np.concatenate([
#         observation['poses']['left'],
#         np.array([0.0]),
#         observation['poses']['right'],
#         np.array([0.0]),
#         observation['poses']['middle'],
#     ])
#     action = init_action

#     while True:
#         step_start = time.time()
#         observation, reward, terminated, truncated, info = env.step(action)

#         headset_data = headset.receive_data()
#         if headset_data is not None:
#             new_action, feedback = headset_control.run(
#                 headset_data, 
#                 observation['poses']['left'], 
#                 observation['poses']['right'], 
#                 observation['poses']['middle']
#             )
#             headset.send_feedback(feedback)

#             if not headset_control.is_running() and headset_data.r_button_one == True:
#                 headset_control.start(
#                     headset_data, 
#                     observation['poses']['middle']
#                 )

#             if headset_control.is_running():
#                 action = new_action

#             if headset_control.is_running() and headset_data.r_button_one == False:
#                 terminated = True

#         # Check if the episode is over (terminated) or max steps reached (truncated)
#         if terminated or truncated:
#             action = init_action
#             headset_control.reset()
#             observation, info = env.reset()

#         # Rudimentary time keeping, will drift relative to wall clock.
#         time_until_next_step = DT - (time.time() - step_start)
#         time.sleep(max(0, time_until_next_step))

# if __name__ == "__main__":
#     import rospy

#     def shutdown():
#         print("Shutting down...")
#         os._exit(42)
#     rospy.on_shutdown(shutdown)
    
#     main()