'''
录制模拟环境的数据
'''
import numpy as np
from constants import SIM_DT, SIM_TASK_CONFIGS
import asyncio
from webrtc_headset import WebRTCHeadset
# HeadsetFullControl控制三臂，HeadsetControl控制中间单臂，需要配合遥操使用，可以采集力控数据
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback
import argparse
from sim_env import make_sim_env
import time
import os
from tqdm import tqdm
import h5py

def send_popup_message(headset, message, duration=3.0):
    feedback = HeadsetFeedback()
    feedback.info = message
    headset.send_feedback(feedback)
    time.sleep(duration)

# 初始化环境并执行一步，以确保环境和头戴设备的状态同步
def init_step(env):
    ts, info = env.reset()
    # 将初始观测中的左右臂和中间臂的位姿拼接成一个动作向量，并在每个臂的末尾添加一个0.0，表示夹持器的状态（0.0表示未夹持），然后执行这个动作以确保环境和头戴设备的状态同步
    action = np.concatenate([
        ts['poses']['left'],
        np.array([0.0]), # 夹持器的状态
        ts['poses']['right'],
        np.array([0.0]),
        ts['poses']['middle'],
    ])
    env.step(action)

#重置环境并发送反馈给头戴设备，提示用户环境正在重置
def reset_env(env, headset):
    print("Resetting the environment...")
     # reset the environment
    feedback = HeadsetFeedback() #创建一个HeadsetFeedback对象，用于向头戴设备发送反馈信息
    feedback.info = "Resetting the environment..." #设置反馈信息，提示用户环境正在重置
    headset.send_feedback(feedback) #将反馈信息发送到头戴设备
    ts, info = env.reset() #重置环境，获取初始观测和信息
    return ts #返回初始观测

# 执行一个轨迹，返回轨迹回放数据、动作回放数据和一个布尔值表示轨迹是否成功完成，如果轨迹未成功完成（例如用户中途终止），则继续下一次循环等待用户开始下一个轨迹
def run_episode(env, headset, episode_len, episode_idx):
    headset_control = HeadsetControl() #创建HeadsetControl对象，处理来自头戴设备的数据并生成控制命令和反馈
    feedback = HeadsetFeedback() #创建一个HeadsetFeedback对象，用于向头戴设备发送反馈信息
    headset_control.reset() #重置头戴设备控制器的状态，以准备开始一个新的轨迹
    action = np.zeros(23) #初始化一个全零的动作向量

    # wait for user to start the episode
    print("Waiting for user to start the episode...")
    while True:
        start_time = time.time()
        
        ts = env.get_obs() # 获取当前环境的观测，包括左右臂和中间臂的位姿，以及相机图像等信息

        headset_data = headset.receive_data() # 从头戴设备接收数据
        if headset_data is not None:
            # get the action and feedback from the headset control
            # 获取VR设备的控制命令（动作）和mujoco中的反馈信息（例如对齐状态、同步状态等）
            action, feedback = headset_control.run(
                headset_data,  #头戴设备数据（包括手持器数据）
                ts['poses']['left'], #环境中左臂的位姿 
                ts['poses']['right'],  #环境中右臂的位姿
                ts['poses']['middle']   #环境中中间臂的位姿
            )
            # break if the user holds the right button
            # 有手柄按键按下且处于同步状态，则接管机械臂控制权
            if headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.left_out_of_sync == False and feedback.right_out_of_sync == False:
                headset_control.start(
                    headset_data, 
                    ts['poses']['middle']
                )
                break #跳出while 循环，开始执行轨迹

        feedback.info = f"Align and hold A to start the episode {episode_idx}."
        headset.send_feedback(feedback)
        
        # send initial image to headset
        # 获取环境中zed相机的图像，并将其分割成左右两部分，分别发送给头戴设备的左右眼显示屏
        zed_img = ts["images"]["zed_cam"] 
        right_img = zed_img[:, zed_img.shape[1]//2:, :]
        left_img = zed_img[:, :zed_img.shape[1]//2, :]
        headset.send_images(left_img, right_img)

        time_until_next_step = SIM_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step)) # 保持与仿真环境的时间步长同步，确保固定频率

    # run the episode
    print(f"Starting episode {episode_idx}...")
    episode_replay = [ts]
    action_replay = []
    for step_idx in tqdm(range(episode_len)):
        step_start = time.time()

        # Take a step in the environment using the chosen action
        ts, reward, terminated, truncated, info = env.step(action)
        episode_replay.append(ts) # 保存当前时间步的观测数据
        action_replay.append(action)  # 保存当前时间步的动作数据

        # Check if the episode is terminated 
        if terminated: # 如果环境返回的terminated标志为True，表示任务失败，返回空数据
            return [], [], False

        # Receive data from the headset
        headset_data = headset.receive_data()
        if headset_data is not None:
            action, feedback = headset_control.run(
                headset_data, 
                ts['poses']['left'], #左右臂是为了判断VR和mujoco位姿是否对齐
                ts['poses']['right'], 
                ts['poses']['middle'] #左右臂的基座标，将VR的手柄和头显的位姿增值叠加到mid_arm以获取左右臂位姿
            )
            if headset_data.r_button_one == False: # 松开按键立刻停止轨迹录制，返回空数据
                send_popup_message(headset, f"Episode {episode_idx} terminated by user. {info}", 3.0)
                return [], [], False  

        feedback.info = f"Episode {episode_idx}, Timestep: {str(step_idx).zfill(len(str(episode_len)))}/{episode_len}\n{info}"
        headset.send_feedback(feedback) 
        
        # send initial image to headset
        zed_img = ts["images"]["zed_cam"]
        right_img = zed_img[:, zed_img.shape[1]//2:, :]
        left_img = zed_img[:, :zed_img.shape[1]//2, :]
        headset.send_images(left_img, right_img)

        print(f"Step time: {time.time() - step_start}s")

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))        
    
    return episode_replay, action_replay, True


def confirm_episode(headset, episode_idx):
    # wait for user to redo or do next episode
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset() # 清空上一轮录制时残留的按键状态

    print("Waiting for user to redo or do next episode...")
    while True:
        start_time = time.time()

        headset_data = headset.receive_data() # 从 VR 设备拉取最新的传感器和按键数据
        if headset_data is not None:
            if headset_data.l_button_one == True: # 如果左手柄的 X 按钮被按下
                return True # 返回 True 表示要进行下一轮录制
            elif headset_data.l_button_two == True: # 如果左手柄的 Y 按钮被按下
                return False # 返回 False 表示要重新录制
              
        feedback.info = f"Episode {episode_idx} completed. Press X to start next episode or Y to redo."
        headset.send_feedback(feedback) # 将反馈信息发送到 VR 设备
                
        time_until_next_step = SIM_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step)) # 固定频率


def save_episode(headset, episode_replay, action_replay, camera_names, dataset_dir, episode_idx):
    feedback = HeadsetFeedback()
    feedback.info = f"Saving episode {episode_idx}..."
    headset.send_feedback(feedback)

    # convert data to dataset format
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/all_qpos': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(episode_replay) i.e. time steps: max_timesteps
    max_timesteps = len(episode_replay)
    while episode_replay:  # 
        """
        #从episode_replay列表中依次取出每个时间步的数据并存储到data_dict字典中对应的键下，以便后续保存到HDF5文件中。
        同时，如果camera_names列表中包含相机名称，还会将对应相机的图像数据也存储到data_dict中。
        """
        ts = episode_replay.pop(0)  # 从episode_replay列表中依次取出每个时间步的观测数据ts
        data_dict['/observations/qpos'].append(ts['joints']['position']) # 关节位置
        data_dict['/observations/qvel'].append(ts['joints']['velocity']) # 关节速度
        data_dict['/observations/all_qpos'].append(ts['qpos']) # 完整的关节位置
        data_dict['/action'].append(ts['control']) # 动作
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])

    # HDF5
    # HDF5 文件在创建数据集时，必须提前知道数据的“形状（Shape）
    try:
        if len(camera_names) > 0: # 获取相机图像的形状
            image_shapes = {cam_name: data_dict[f'/observations/images/{cam_name}'][0].shape for cam_name in camera_names}
        qpos_len = len(data_dict['/observations/qpos'][0]) # 获取关节位置的长度
        qvel_len = len(data_dict['/observations/qvel'][0]) # 获取关节速度的长度
        all_qpos_len = len(data_dict['/observations/all_qpos'][0]) # 获取完整的关节位置的长度
        action_len = len(data_dict['/action'][0]) # 获取动作的长度
    except IndexError:
        print('Empty episode, skipping...')
        return False
    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    # 设置HDF5 文件的读写缓存为 1024^2*2=2MB
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True # 标记是否是仿真数据
        obs = root.create_group('observations') #创建observations分组，用于存储的观测数据
        if len(camera_names) > 0:
            image = obs.create_group('images') # observations/images/
            for cam_name in camera_names:
                # 创建相机图像的数据集, 把每一帧图片作为一个独立的数据块（Chunk）存储,后续训练时可以直接帧读取
                _ = image.create_dataset(cam_name, (max_timesteps, *image_shapes[cam_name]), dtype='uint8',
                                            chunks=(1, *image_shapes[cam_name]), ) 
        # compression='gzip',compression_opts=2,) #这是gzip压缩，可以减小hdf5文件的大小,但是会增加读取时间，增加cup负担
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        data_qpos = obs.create_dataset('qpos', (max_timesteps, qpos_len))
        data_all_qpos = obs.create_dataset('all_qpos', (max_timesteps, all_qpos_len))
        data_qvel = obs.create_dataset('qvel', (max_timesteps, qvel_len))
        data_action = root.create_dataset('action', (max_timesteps, action_len))

        for name, array in data_dict.items():
            root[name][...] = array # 将数据存储到HDF5文件中
    print(f'Saving: {time.time() - t0:.1f} secs\n') 
    return True

def collect_data(args):
    dataset_dir = SIM_TASK_CONFIGS[args["task_name"]]["dataset_dir"] #数据集目录，../data_collection_scripts/data/(task_name决定)
    num_episodes = SIM_TASK_CONFIGS[args["task_name"]]["num_episodes"] #轨迹数量，50
    episode_len = SIM_TASK_CONFIGS[args["task_name"]]["episode_len"]  #轨迹长度  300/400
    camera_names = SIM_TASK_CONFIGS[args["task_name"]]["camera_names"] #任务对应使用的相机名称列表
    record_video = args["record_video"]  #是否记录视频
    start_episode_idx = args["episode_idx"] #轨迹索引，即从第几条轨迹开始录制

    # TODO
    # record_video = False # 目前视频录制功能不完善，先默认关闭，后续完善后再开放

    # create the dataset directory if it does not exist
    if not os.path.isdir(dataset_dir): #如果数据集目录不存在
        os.makedirs(dataset_dir, exist_ok=True) #创建数据集目录

    # setup the headset control
    # 设置头戴设备控制
    headset = WebRTCHeadset() #创建WebRTCHeadset对象，连接到头戴设备
    headset.run_in_thread() #在一个单独的线程中运行头戴设备的通信循环，以便它可以同时接收数据和发送反馈
    headset_control = HeadsetControl() #创建HeadsetControl对象，处理来自头戴设备的数据并生成控制命令和反馈

    # setup the environment
    if record_video:
        print("Recording video is enabled.")
        env = make_sim_env(args["task_name"], cameras=camera_names) #根据任务名称和相机名称列表创建仿真环境
    else:
        print("Recording video is disabled.")
        env = make_sim_env(args["task_name"], cameras=['zed_cam'])
    
    init_step(env) #初始化环境并执行一步，以确保环境和头戴设备的状态同步

    episode_idx = start_episode_idx #从指定的轨迹索引开始录制数据
    while episode_idx < num_episodes:
        reset_env(env, headset) #重置环境并发送反馈给头戴设备，提示用户环境正在重置

        # run the episode
        # 执行一个轨迹，返回轨迹回放数据、动作回放数据和一个布尔值表示轨迹是否成功完成，如果轨迹未成功完成（例如用户中途终止），则继续下一次循环等待用户开始下一个轨迹
        episode_replay, action_replay, ok = run_episode(env, headset, episode_len, episode_idx)

        if not ok: # 如果轨迹未成功完成
            continue # 继续下一次循环

        # confirm if the episode is to be saved
        ok = confirm_episode(headset, episode_idx)

        if not ok: # 如果用户不想保存轨迹
            continue # 继续下一次循环

        # save the episode
        if record_video:
            ok = save_episode(headset, episode_replay, action_replay, camera_names, dataset_dir, episode_idx)
        else:
            ok = save_episode(headset, episode_replay, action_replay, [], dataset_dir, episode_idx)

        if not ok: # 如果保存失败
            continue # 继续下一次循环

        episode_idx += 1
'''
录制第 0 个视频，命令行输入：
python collect_data.py --task_name sim_tube_transfer_3arms --episode_idx 0 
'''
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Headset control") #创建一个ArgumentParser对象，用于从命令行解析参数
    parser.add_argument("--task_name", type=str, default="sim_insert_peg")  #任务名称
    parser.add_argument("--episode_idx", type=int, default=0) #轨迹索引
    parser.add_argument("--record_video", type=bool, default=False) #是否记录视频
    args = parser.parse_args()
    args = vars(args)

    try:
        collect_data(args)
    except KeyboardInterrupt: # 捕获键盘中断 ctrl+c
        print("Shutting down...")
        os._exit(42) # 退出程序
