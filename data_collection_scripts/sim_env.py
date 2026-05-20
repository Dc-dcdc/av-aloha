import time
import numpy as np
import mujoco.viewer
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from constants import (
    XML_DIR,  #src/av-aloha/data_collection_scripts/assets
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE, MIDDLE_BASE_LINK,
    LEFT_GRIPPER_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES
)
from diff_ik import DiffIK
from grad_ik import GradIK
import os
from kinematics import create_fk_fn, create_safety_fn
from transform_utils import xyzw_to_wxyz, mat2pose, pose2mat, wxyz_to_xyzw

CAMERAS = ['zed_cam', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low']

#根据任务名称创建仿真环境，仿真环境会返回指定相机的图像数据
def make_sim_env(task_name, cameras=CAMERAS):
    if 'sim_insert_peg' in task_name:  # 轴孔装配任务
        return InsertPegEnv(cameras=cameras) 
    elif 'sim_slot_insertion' in task_name: # 狭缝插入任务
        return SlotInsertionEnv(cameras=cameras)
    elif 'sim_sew_needle' in task_name: # 缝合针穿引任务
        return SewNeedleEnv(cameras=cameras)
    elif 'sim_tube_transfer' in task_name: # 管道小球传输任务
        return TubeTransferEnv(cameras=cameras)
    elif 'sim_hook_package' in task_name: # 挂钩挂包任务
        return HookPackageEnv(cameras=cameras)
    else:
        raise NotImplementedError # 

# 仿真环境基类，封装了Mujoco物理引擎的基本操作，包括加载模型、获取观测、执行动作、渲染等功能
class GuidedVisionEnv(gym.Env):

    def __init__(self, xml, cameras=CAMERAS): 
        self._mjcf_root = mjcf.from_path(xml)  
        self._mjcf_root.option.timestep = SIM_PHYSICS_DT  
        
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 

        # 用于强化学习的观测空间和动作空间定义，观测空间包含了机器人关节状态、全局状态、控制输入、末端执行器位姿以及相机图像等信息，动作空间是一个连续空间，表示机器人关节的目标位置（包括 gripper 的开合状态） 
        self.observation_space = spaces.Dict({
            'joints': spaces.Dict({
                'position': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
                'velocity': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,))  # 21 joint velocities
            }),
            'qpos': spaces.Box(low=-float('inf'), high=float('inf')), 
            'control': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
            'poses': spaces.Dict({
                'left': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # left arm pose
                'right': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # right arm pose
                'middle': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,))  # middle arm pose
            }),
            'images': spaces.Dict({
                'zed_cam': spaces.Box(low=0, high=255, shape=(720, 2*1280, 3)),  # zed camera image
                'cam_left_wrist': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # left wrist camera image
                'cam_right_wrist': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # right wrist camera image
                'cam_high': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # high camera image
                'cam_low': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # low camera image
            }),
        })
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
        )   
        # 在 Python 代码和底层的 MuJoCo XML 物理模型之间建立连接
        
        self._middle_base_link = self._mjcf_root.find('body', MIDDLE_BASE_LINK)
        self._middle_base_link_init_pos = self._middle_base_link.pos.copy()
        # 绑定关节，用于读取关节角度
        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        # 绑定动作，用于发送控制指令
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        # 绑定末端执行器，用于读取末端执行器位姿
        self._left_eef_site = self._mjcf_root.find('site', LEFT_EEF_SITE)
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        self._middle_eef_site = self._mjcf_root.find('site', MIDDLE_EEF_SITE)
        # 创建正运动学(FK)函数，后续可以输入角度值计算末端执行器位姿
        self._left_fk_fn = create_fk_fn(self._physics, self._left_joints[:6], self._left_eef_site)
        self._right_fk_fn = create_fk_fn(self._physics, self._right_joints[:6], self._right_eef_site)
        self._middle_fk_fn = create_fk_fn(self._physics, self._middle_joints[:7], self._middle_eef_site)
        # 绑定夹爪关节进行单独控制
        self._left_gripper_joints = [self._mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self._right_gripper_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]

        # set up controllers
        # 左臂控制器，采用梯度下降逆解法
        self._left_controller = GradIK(
            physics=self._physics,
            joints = self._left_joints[:6],
            actuators=self._left_actuators[:6],
            eef_site=self._left_eef_site,
            step_size=0.0001, 
            min_cost_delta=1.0e-12, 
            max_iterations=50, 
            position_weight=500.0,
            rotation_weight=100.0,
            joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
            joint_displacement_weight=np.array(6*[50.0]),
            position_threshold=0.001,
            rotation_threshold=0.001,
            max_pos_diff=0.1,
            max_rot_diff=0.3,
            joint_p = 0.9,
        )
        self._right_controller = GradIK(
            physics=self._physics,
            joints=self._right_joints[:6],
            actuators=self._right_actuators[:6],
            eef_site=self._right_eef_site,
            step_size=0.0001, 
            min_cost_delta=1.0e-12, 
            max_iterations=50, 
            position_weight=500.0,
            rotation_weight=100.0,
            joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
            joint_displacement_weight=np.array(6*[50.0]),
            position_threshold=0.001,
            rotation_threshold=0.001,
            max_pos_diff=0.1,
            max_rot_diff=0.3,
            joint_p = 0.9,
        )
        # 中间臂控制器，采用差分逆解法 
        self._middle_controller = DiffIK(
            physics=self._physics,
            joints=self._middle_joints, # 中间臂关节
            actuators=self._middle_actuators, # 中间臂执行器
            eef_site=self._middle_eef_site, # 中间臂末端执行器位置
            k_pos=0.9, # 中间臂末端执行器位置权重
            k_ori=0.9, # 中间臂末端执行器姿态权重
            damping=1.0e-4, # 阻尼系数
            k_null=np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]), # 关节中心位置权重，越大越倾向于回到初始位姿
            q0=np.array(MIDDLE_ARM_POSE), # 中间臂初始位姿
            max_angvel=3.14, # 中间臂最大关节角速度
            integration_dt=SIM_DT, # 中间臂控制器的积分时间步长
            iterations=10, # 中间臂每个控制步骤的迭代次数，越大越精确但计算越慢
        )
        # 读取真实物理引擎中夹爪电机的控制限位
        self.left_gripper_range = self._physics.bind(self._left_actuators[-1]).ctrlrange # 
        self.right_gripper_range = self._physics.bind(self._right_actuators[-1]).ctrlrange
        # 归一化：(x - min) / (max - min)
        self.left_gripper_norm_fn = lambda x: (x - self.left_gripper_range[0]) / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_norm_fn = lambda x: (x - self.right_gripper_range[0]) / (self.right_gripper_range[1] - self.right_gripper_range[0])
        # 反归一化:将模型输出映射到实际控制量 y = x * (max - min) + min
        self.left_gripper_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0]) + self.left_gripper_range[0]
        self.right_gripper_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0]) + self.right_gripper_range[0]
        # 速度归一化
        self.left_gripper_vel_norm_fn = lambda x: x / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_vel_norm_fn = lambda x: x / (self.right_gripper_range[1] - self.right_gripper_range[0])
        # 速度反归一化：将模型输出的比例速度放大回物理速度
        self.left_gripper_vel_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_vel_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0])

        # for GUI and time keeping
        self._viewer = None 

        # check all cameras are valid
        for camera in cameras:
            assert camera in CAMERAS, f"Invalid camera name: {camera}" # 

        self._cameras = cameras

    def get_obs(self) -> np.ndarray:
        """
            获取机器人的当前观测状态。
            该方法从物理仿真环境中读取机器人的关节位置、速度、控制输入，
            计算末端执行器的位姿信息，并渲染多个相机的图像数据。
            返回值：
                dict: 包含以下键值对的字典
                    - 'joints' (dict): 机器人关节数据
                        - 'position' (np.ndarray): 所有关节的位置数据，包括左臂(6个关节+1个gripper)、
                          右臂(6个关节+1个gripper)、中臂的位置。Gripper位置已归一化到0-1范围。
                        - 'velocity' (np.ndarray): 所有关节的速度数据，Gripper速度已归一化到-1~1范围
                          (正值表示闭合，负值表示张开)。
                    
                    - 'qpos' (np.ndarray): 上帝视角的全局关节数据，包含环境中所有对象的状态信息。
                    
                    - 'control' (np.ndarray): 控制器的目标控制输入，由微分逆解(DiffIK)计算得出，
                      相当于模型的预期动作标签。
                    
                    - 'poses' (dict): 末端执行器的当前位姿信息，基于控制器计算的目标位姿
                        - 'left' (np.ndarray): 左臂末端位置(3个坐标) + 姿态(4元数，wxyz格式)
                        - 'right' (np.ndarray): 右臂末端位置(3个坐标) + 姿态(4元数，wxyz格式)
                        - 'middle' (np.ndarray): 中臂末端位置(3个坐标) + 姿态(4元数，wxyz格式)
                    
                    - 'images' (dict): 各相机渲染的图像数据
                        - 'zed_cam': 左右双目相机图像拼接(1440x720)
                        - 'cam_left_wrist': 左腕部相机图像(640x480)
                        - 'cam_right_wrist': 右腕部相机图像(640x480)
                        - 'cam_high': 顶部俯视相机图像(640x480)
                        - 'cam_low': 底部仰视相机图像(640x480)
        """
        left_qpos = self._physics.bind(self._left_joints).qpos.copy() # 获取左臂关节位置数据，包含了6个自由度的机械臂关节和1个 gripper 关节
        left_qpos[6] = self.left_gripper_norm_fn(left_qpos[6]) # 将 gripper 的 qpos 归一化到 0-1 范围内，方便后续训练模型使用
        right_qpos = self._physics.bind(self._right_joints).qpos.copy()
        right_qpos[6] = self.right_gripper_norm_fn(right_qpos[6])
        middle_qpos = self._physics.bind(self._middle_joints).qpos.copy()
        left_qvel = self._physics.bind(self._left_joints).qvel.copy() # 获取左臂关节速度数据，包含了机械臂关节和 gripper 关节的速度信息
        left_qvel[6] = self.left_gripper_vel_norm_fn(left_qvel[6]) # 将 gripper 的 qvel 归一化到 -1 到 1 的范围内，表示 gripper 的开合速度，正值表示闭合，负值表示张开
        right_qvel = self._physics.bind(self._right_joints).qvel.copy()
        right_qvel[6] = self.right_gripper_vel_norm_fn(right_qvel[6])
        middle_qvel = self._physics.bind(self._middle_joints).qvel.copy()
        left_ctrl = self._physics.bind(self._left_actuators).ctrl.copy() # 获取左臂控制器的当前控制输入，包含了机械臂关节和 gripper 关节的目标位置（经过 DiffIK 计算出来的目标角度）
        left_ctrl[6] = self.left_gripper_norm_fn(left_ctrl[6])
        right_ctrl = self._physics.bind(self._right_actuators).ctrl.copy() 
        right_ctrl[6] = self.right_gripper_norm_fn(right_ctrl[6])
        middle_ctrl = self._physics.bind(self._middle_actuators).ctrl.copy()
        qpos = self._physics.data.qpos.copy()

        # send back ctrl instead of qpos because we want to send back the commanded position
        # real position might be affected by gravity and other forces
        # 返回 ctrl 而不是 qpos，因为我们想返回“被指令要求的位姿”。真实的物理位姿可能会受到重力和其他外力的影响
        left_pos, left_quat = mat2pose(self._left_fk_fn(self._physics.bind(self._left_actuators[:6]).ctrl))
        right_pos, right_quat = mat2pose(self._right_fk_fn(self._physics.bind(self._right_actuators[:6]).ctrl))
        middle_pos, middle_quat = mat2pose(self._middle_fk_fn(self._physics.bind(self._middle_actuators).ctrl))
        # Mujoco 底层标准是 (w, x, y, z)，这里统一做了一次格式对齐
        left_quat = xyzw_to_wxyz(left_quat)
        right_quat = xyzw_to_wxyz(right_quat)
        middle_quat = xyzw_to_wxyz(middle_quat)

        images = {}
        for camera in self._cameras: # 根据指定的相机列表，渲染对应的图像数据并保存到images字典中
            if 'zed_cam' in camera:#保存左右相机的图像数据
                images['zed_cam'] = np.concatenate([
                    self._physics.render(height=720, width=720, camera_id='zed_cam_left'), 
                    self._physics.render(height=720, width=720, camera_id='zed_cam_right'),
                ], axis=1)
            elif 'cam_left_wrist' in camera: # 左手腕相机
                images['cam_left_wrist'] = self._physics.render(height=480, width=640, camera_id='wrist_cam_left')
            elif 'cam_right_wrist' in camera: # 右手腕
                images['cam_right_wrist'] = self._physics.render(height=480, width=640, camera_id='wrist_cam_right')
            elif 'cam_high' in camera:
                images['cam_high'] = self._physics.render(height=480, width=640, camera_id='overhead_cam')
            elif 'cam_low' in camera:
                images['cam_low'] = self._physics.render(height=480, width=640, camera_id='worms_eye_cam')
            else:
                raise NotImplementedError(f"Camera {camera} not implemented")

        return {
            'joints': { #机器人自身关节数据，包括角度和速度，gripper的开合状态也被归一化到0-1范围内
                'position': np.concatenate([left_qpos, right_qpos, middle_qpos]), # 关节位置
                'velocity': np.concatenate([left_qvel, right_qvel, middle_qvel]) # 关节速度
            },
            'qpos': qpos, # 上帝视角的全局数据，其他环境信息（如物体位置）也包含在qpos中
            'control': np.concatenate([left_ctrl, right_ctrl, middle_ctrl]),#经过 DiffIK（微积分逆解）算出来的目标角度，相当于训练模型的输出动作（标签）
            'poses': { # 末端执行器的当前位姿信息，包括位置和姿态（四元数），这里的位姿是根据控制器计算出来的目标位姿，而不是实际执行后的位姿
                'left': np.concatenate([left_pos,left_quat]), 
                'right': np.concatenate([right_pos, right_quat]), 
                'middle': np.concatenate([middle_pos, middle_quat])
            },
            'images': images,
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        # 初始化机械臂的关节角度和控制器，
        # 只改 qpos，不改 ctrl，电机会瞬间到设定的qpos,造成巨大的动量冲击
        # 只改 ctrl，不改 qpos，机械臂会自动根据ctrl慢慢更新到新位置
        self._physics.reset()
        self._physics.bind(self._left_joints).qpos = LEFT_ARM_POSE # 初始化左臂的关节角度
        self._physics.bind(self._left_gripper_joints).qpos = self.left_gripper_unnorm_fn(1) # 夹爪张开到最大
        self._physics.bind(self._right_joints).qpos = RIGHT_ARM_POSE
        self._physics.bind(self._right_gripper_joints).qpos = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_joints).qpos = MIDDLE_ARM_POSE
        self._physics.bind(self._left_actuators).ctrl = LEFT_ARM_POSE # 初始化左臂的控制器，和左臂的关节角度一样
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1) #初始化左臂的夹爪控制器
        self._physics.bind(self._right_actuators).ctrl = RIGHT_ARM_POSE
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_actuators).ctrl = MIDDLE_ARM_POSE

        self._physics.forward() # 强制物理引擎进行一次正向运动学计算

        observation = self.get_obs() #读取当前环境的观测
        info = "Resetting arms..."

        return observation, info   

    def set_qpos(self, qpos: np.ndarray):
        self._physics.data.qpos[:] = qpos
        # forward kinematics
        self._physics.forward()


    def step_joints(self, action: np.ndarray) -> tuple:
        left_joints = action[:6]
        left_gripper = action[6] # val from 0 to 1
        right_joints = action[7:13]
        right_gripper = action[13] # val from 0 to 1
        middle_joints = action[14:21]

        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints
        self._physics.bind(self._middle_actuators).ctrl = middle_joints

        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        # step physics
        self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        
        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = ""

        return observation, reward, terminated, truncated, info



    # 控制末端执行器位姿，还需要配合控制器（DiffIK/GradIK）求解关节角度
    def step(self, action: np.ndarray) -> tuple:
        left_target = action[:7] # 位置+四元数
        left_gripper = action[7] # val from 0 to 1
        right_target = action[8:15]
        right_gripper = action[15] # val from 0 to 1
        middle_target = action[16:23]

        self._physics.bind(self._left_actuators[:6]).ctrl = self._left_controller.run(
            self._physics.bind(self._left_joints).qpos[:6],
            left_target[:3],
            left_target[3:]
        )
        self._physics.bind(self._right_actuators[:6]).ctrl = self._right_controller.run(
            self._physics.bind(self._right_joints).qpos[:6],
            right_target[:3],
            right_target[3:]
        )
        self._physics.bind(self._middle_actuators).ctrl = self._middle_controller.run(
            self._physics.bind(self._middle_joints).qpos,
            middle_target[:3],
            middle_target[3:]
        )

        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1-left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1-right_gripper)

        # step physics
        self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        
        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = ""

        return observation, reward, terminated, truncated, info

    def render_viewer(self) -> np.ndarray:
        if self._viewer is None:
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
                show_left_ui=True,
                show_right_ui=True,
            )
        # render viewer
        self._viewer.sync()

    # 将中间臂隐藏（移出视角外）
    def hide_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = np.array([0, -2, 0])

    def show_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = self._middle_base_link_init_pos


    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()


class InsertPegEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_insert_peg.xml')
        super().__init__(xml, cameras)

        self.max_reward = 4 

        self._peg_joint = self._mjcf_root.find('joint', 'peg_joint') #在MJCF模型中找到名为'peg_joint'的关节，并将其保存在self._peg_joint中，方便后续操作
        self._hole_joint = self._mjcf_root.find('joint', 'hole_joint') #在MJCF模型中找到名为'hole_joint'的关节，并将其保存在self._hole_joint中，方便后续操作

    def get_reward(self):

        touch_left_gripper = False #左侧夹持器接触
        touch_right_gripper = False #右侧夹持器接触
        peg_touch_table = False #插销接触桌子
        hole_touch_table = False #孔接触桌子
        peg_touch_hole = False #插销接触孔
        pin_touched = False #插销接触销钉

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs: #遍历所有接触对，检查是否满足特定的接触条件，以确定当前的奖励等级
            if geom1 == "peg" and geom2.startswith("right"): 
                touch_right_gripper = True
            
            if geom1.startswith("hole-") and geom2.startswith("left"): 
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "peg":
                peg_touch_table = True

            if geom1 == "table" and geom2.startswith("hole-"): #如果接触对中一个是"table"（桌子）另一个是以"hole-"开头的字符串（表示孔），则将hole_touch_table设置为True，表示孔已经成功接触到桌子。
                hole_touch_table = True

            if geom1 == "peg" and geom2.startswith("hole-"): #如果接触对中一个是"peg"（插销）另一个是以"hole-"开头的字符串（表示孔），则将peg_touch_hole设置为True，表示插销已经成功接触到孔，这通常是任务完成的关键条件之一。
                peg_touch_hole = True

            if geom1 == "peg" and geom2 == "pin": #如果接触对中一个是"peg"（插销）另一个是"pin"（销钉），则将pin_touched设置为True，表示插销已经成功接触到销钉，这通常是任务完成的关键条件之一。
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not hole_touch_table): # grasp both
            reward = 2
        if peg_touch_hole and (not peg_touch_table) and (not hole_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

    #
    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics 
        # 插销的位置
        x_range = [0.1, 0.2] #插销在x轴上的随机范围
        y_range = [-0.1, 0.1] #插销在y轴上的随机范围
        z_range = [0.01, 0.01] #插销在z轴上的随机范围
        ranges = np.vstack([x_range, y_range, z_range]) #将x_range、y_range和z_range堆叠成一个3行2列的数组，每行对应一个轴的范围，第一列是最小值，第二列是最大值
        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1]) #在每个轴的指定范围内随机生成一个位置，ranges[:, 0]表示每个轴的最小值，ranges[:, 1]表示每个轴的最大值，生成的peg_position是一个包含x、y、z坐标的数组，表示插销的初始位置
        peg_quat = np.array([1, 0, 0, 0]) #插销的初始姿态，使用四元数表示，这里表示没有旋转，即插销的坐标轴与世界坐标轴对齐

        #孔的位置
        x_range = [-0.1, -0.2]
        y_range = [-0.1, 0.1]
        z_range = [0.021, 0.021]
        ranges = np.vstack([x_range, y_range, z_range])
        hole_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        hole_quat = np.array([1, 0, 0, 0])

        self._physics.bind(self._peg_joint).qpos = np.concatenate([peg_position, peg_quat]) #将随机生成的peg位置和姿态拼接，并设置到物理引擎中对应的peg关节上，初始化peg的位置和姿态
        self._physics.bind(self._hole_joint).qpos = np.concatenate([hole_position, hole_quat]) #将随机生成的hole位置和姿态拼接，并设置到物理引擎中对应的hole关节上，初始化hole的位置和姿态

        self._physics.forward()


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info #返回初始观察和信息字符串
    
class SlotInsertionEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_slot_insertion.xml')
        super().__init__(xml, cameras)

        self.max_reward = 4

        self._slot_joint = self._mjcf_root.find('joint', 'slot_joint')
        self._stick_joint = self._mjcf_root.find('joint', 'stick_joint')

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [-0.05, 0.05]
        y_range = [0.1, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        slot_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        slot_quat = np.array([1, 0, 0, 0])


        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.08, 0.08]
        y_range = [-0.1, 0.0]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        stick_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        stick_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._slot_joint).qpos = np.concatenate([slot_position, slot_quat])
        self._physics.bind(self._stick_joint).qpos = np.concatenate([stick_position, stick_quat])

        self._physics.forward()


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        stick_touch_table = False
        stick_touch_slot = False
        pins_touch = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "stick" and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1 == "stick" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "stick":
                stick_touch_table = True

            if geom1 == "stick" and geom2.startswith("slot-"):
                stick_touch_slot = True

            if geom1 == "pin-stick" and geom2 == "pin-slot":
                pins_touch = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not stick_touch_table): # grasp stick
            reward = 2
        if stick_touch_slot and (not stick_touch_table): # peg and socket touching
            reward = 3
        if pins_touch: # successful insertion
            reward = 4
        return reward
    

class SewNeedleEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_sew_needle.xml')
        super().__init__(xml, cameras)

        self.max_reward = 5

        self._needle_joint = self._mjcf_root.find('joint', 'needle_joint')
        self._wall_joint = self._mjcf_root.find('joint', 'wall_joint')

        self._threaded_needle = False

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [0.15, 0.2]
        y_range = [-.025,0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        needle_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        needle_quat = np.array([1, 0, 0, 0])


        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.025, 0.025]
        y_range = [-.025,0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        wall_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        wall_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._needle_joint).qpos = np.concatenate([needle_position, needle_quat])
        self._physics.bind(self._wall_joint).qpos = np.concatenate([wall_position, wall_quat])

        self._physics.forward()

        self._threaded_needle = False


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        needle_touch_table = False
        needle_touch_wall = False
        pins_touch = False
        needle_touch_pin = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "needle" and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1 == "needle" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "needle":
                needle_touch_table = True

            if geom1 == "needle" and geom2.startswith("wall-"):
                needle_touch_wall = True

            if geom1 == "pin-needle" and geom2 == "pin-wall":
                self._threaded_needle = True
                pins_touch = True

            if geom1 == "needle" and geom2 == "pin-wall":
                needle_touch_pin = True

        reward = 0
        if touch_right_gripper: # touch needle
            reward = 1
        if touch_right_gripper and (not needle_touch_table): # grasp needle
            reward = 2
        if needle_touch_wall and (not needle_touch_table): # peg and socket touching
            reward = 3
        if self._threaded_needle: # needle threaded
            reward = 4
        if touch_left_gripper and (not touch_right_gripper) and (not needle_touch_table) and (not needle_touch_pin) and self._threaded_needle: # grasped needle on other side
            reward = 5
        return reward
    

class TubeTransferEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_tube_transfer.xml')
        super().__init__(xml, cameras)

        self.max_reward = 3

        self._ball_joint = self._mjcf_root.find('joint', 'ball_joint')
        self._tube1_joint = self._mjcf_root.find('joint', 'tube1_joint')
        self._tube2_joint = self._mjcf_root.find('joint', 'tube2_joint')


    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [0.05, 0.1]
        y_range = [-0.05, 0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        ball_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        ball_quat = np.array([1, 0, 0, 0])
        tube1_position = ball_position
        tube1_quat = ball_quat

        x_range = [-.1, -0.05]
        y_range = [-0.05, 0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        tube2_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        tube2_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._ball_joint).qpos = np.concatenate([ball_position, ball_quat])
        self._physics.bind(self._tube1_joint).qpos = np.concatenate([tube1_position, tube1_quat])
        self._physics.bind(self._tube2_joint).qpos = np.concatenate([tube2_position, tube2_quat])

        self._physics.forward()


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        tube1_touch_table = False
        tube2_touch_table = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("tube1-") and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1.startswith("tube2-") and geom2.startswith("left"): 
                touch_left_gripper = True

            if geom1 == "table" and geom2.startswith("tube1-"):
                tube1_touch_table = True

            if geom1 == "table" and geom2.startswith("tube2-"):
                tube2_touch_table = True

            if geom1 == "ball" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not tube1_touch_table) and (not tube2_touch_table): # grasp both
            reward = 2
        if pin_touched:
            reward = 3
        return reward
    
# 
class HookPackageEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_hook_package.xml')
        super().__init__(xml, cameras)

        self.max_reward = 4
        # 1. 寻址与缓存 (发生在 __init__ 中，只执行一次)
        # 在整棵 XML 树 (_mjcf_root) 中，寻找类型为 'joint'，名字叫 'package_joint' 的节点
        self._package_joint = self._mjcf_root.find('joint', 'package_joint')
        self._hook_joint = self._mjcf_root.find('joint', 'hook_joint')

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        # 挂钩初始化
        x_range = [-0.1, 0.1]
        y_range = [.3, .3]
        z_range = [0.2, 0.3]
        ranges = np.vstack([x_range, y_range, z_range])
        hook_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        hook_quat = np.array([1, 0, 0, 0])

        # 包裹初始化
        x_range = [-.1, 0.1]
        y_range = [0, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        package_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        package_quat = np.array([1, 0, 0, 0]) 

        # 2. 绑定与覆写 (发生在 reset 中，被高频调用)
        # 直接拿出刚才缓存的句柄，绑定到底层内存，瞬间重置包裹的 XYZ 坐标和姿态
        self._physics.bind(self._hook_joint).qpos = np.concatenate([hook_position, hook_quat])
        self._physics.bind(self._package_joint).qpos = np.concatenate([package_position, package_quat])

        self._physics.forward()

        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    
    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        package_touch_table = False
        package_touch_hook = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("package-") and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1.startswith("package-") and geom2.startswith("left"): 
                touch_left_gripper = True

            if geom1 == "table" and geom2.startswith("package-"):
                package_touch_table = True

            if geom1 == "hook" and geom2.startswith("package-"):
                package_touch_hook = True

            if geom1 == "pin-package" and geom2 == "pin-hook":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not package_touch_table): # grasp both
            reward = 2
        if package_touch_hook and (not package_touch_table):
            reward = 3
        if pin_touched:
            reward = 4
        return reward


if __name__ == '__main__':
    # setup the environment
    env = make_sim_env('sim_hook_package', cameras=[])
    observation, info = env.reset(seed=42)

    init_action = np.concatenate([
        observation['poses']['left'], # 左臂的位姿
        np.array([0.03]), # 夹持器的状态
        observation['poses']['right'],
        np.array([0.03]),
        observation['poses']['middle'],
    ])
    action = init_action
    
    i = 0
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render_viewer()

        # Check if the episode is over (terminated) or max steps reached (truncated)
        if terminated or truncated:
            # If the episode ends or is truncated, reset the environment
            observation, info = env.reset()

        # print("Step time:", time.time() - step_start) # 打印每一步

        if i % 100 == 0:
            env.reset()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

        i += 1
