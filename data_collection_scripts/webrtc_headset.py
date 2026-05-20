'''
WebRTC：Web Real-Time Communication，网页实时通信一套开放标准，让浏览器 / APP 之间直接点对点传视频、语音、数据，不用经过中间服务器转发大流量。
基于 WebRTC 的双向通信模块，主要用于将机器人的双目摄像头画面实时传输到 VR 头显中，同时接收 VR 头显传来的操作指令（头部姿态、手柄位置、按键操作等）。
实现低延迟的远程遥控
'''

from google.cloud import firestore
import json
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCRtpSender
)
from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np
import queue
import time
from headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates
import os
import threading
import cv2

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

class BufferVideoStreamTrack(VideoStreamTrack):
    def __init__(self, buffer_size=1, image_format="rgb24", max_fps=60):
        super().__init__()
        self.queue = queue.Queue(maxsize=buffer_size)
        self.image_format = image_format
        self.last_frame = None
        self.max_fps = max_fps
        self.last_send_time = time.time()

    # 从队列中获取最新的视频帧，如果队列为空则返回上一帧的副本
    async def get_frame(self) -> np.ndarray:
        while True:
            try:
                frame = self.queue.get_nowait()
                self.last_frame = frame
                return frame
            except queue.Empty:
                if self.last_frame is not None:
                    return self.last_frame.copy()
                await asyncio.sleep(0) # 交出控制权，等待下一次事件循环
        

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = await self.get_frame()
        # convert to gray scale
        frame = VideoFrame.from_ndarray(frame, format=self.image_format)
        frame.pts = pts
        frame.time_base = time_base

        # limit fps
        elapsed_time = time.time() - self.last_send_time
        await asyncio.sleep(max(1/self.max_fps - elapsed_time, 0))
        self.last_send_time = time.time()

        return frame

    # 将机器人摄像头捕获的图像帧添加到视频轨道的缓冲区中，以便通过 WebRTC 连接发送到 VR 头显
    def add_frame(self, frame):
        # try to put but if pull pop the oldest frame
        try:
            self.queue.put_nowait(frame) # 将图像帧放进队列，如果队列已满则抛出 queue.Full 异常
        except queue.Full:
            try:
                self.queue.get_nowait() # 从队列中移除最旧的帧，为新的帧腾出空间
                self.queue.put_nowait(frame)
            except (queue.Empty, queue.Full): # 
                pass

#
class WebRTCHeadset:
    def __init__(
        self,
        serviceAccountKeyFile='serviceAccountKey.json', # Google Firestore 的服务账号密钥文件路径
        signalingSettingsFile='signalingSettings.json', # # WebRTC 信令及服务器配置文件路径
        video_buffer_size=1, # 视频帧缓冲区大小，默认为1表示只保留最新的一帧
        data_buffer_size=1, # 数据缓冲区大小，默认为1表示只保留最新的一条数据
        send_data_freq=10,  # 发送数据的频率，单位为Hz，默认为10表示每秒发送10条数据
    ):        
        # create firestore client
        # 连接 Firestore (信令服务器)
        with open(serviceAccountKeyFile) as f:
            serviceAccountKey = json.load(f) #读取google云密钥文件，获取访问 Firestore 所需的认证信息
        # 使用密钥实例化 Firestore 客户端，用于后续与数据库通信
        self.db = firestore.Client.from_service_account_info(serviceAccountKey)

        # load signaling settings
        with open(signalingSettingsFile) as f:
            signalingSettings = json.load(f) #读取信令设置文件，获取 WebRTC 连接所需的配置信息，包括机器人ID、密码以及 TURN 服务器的相关信息
        self.robotId = signalingSettings['robotID'] #获取机器人ID，用于在 Firestore 中标识当前机器人的信令数据
        self.password = signalingSettings['password'] # 连接密码，客户端与机器人建立连接
        self.turn_server_url = signalingSettings['turn_server_url'] # TURN 服务器的连接地址(URL)，用于在 WebRTC 连接中进行 NAT 穿透，确保即使在复杂网络环境下也能建立连接
        self.turn_server_username = signalingSettings['turn_server_username'] # TURN 服务器的用户名，用于认证和授权访问 TURN 服务器
        self.turn_server_password = signalingSettings['turn_server_password'] # TURN 服务器的密码

        # create peer connection
        # 配置 STUN 和 TURN 服务器以支持 NAT 穿透，确保在各种网络环境下都能建立 WebRTC 连接
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                # 优先使用Google 提供的公共 STUN 服务器，帮助客户端获取其公共 IP 地址和端口，实现直接连接
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                # 配置自己的 TURN 服务器，确保直连失败也能进行通信
                RTCIceServer(self.turn_server_url, self.turn_server_username, self.turn_server_password)
            ])
        )

        # vars for video and data
        self.channel = None # WebRTC 数据通道，用于发送和接收控制数据（例如头部姿态、手柄位置、按键操作等）
        self.left_video_track = None # WebRTC 视频轨道，用于发送左目摄像头的视频帧到 VR 头显
        self.right_video_track = None # WebRTC 视频轨道，用于发送右目摄像头的视频帧到 VR 头显
        self.video_buffer_size = video_buffer_size # 视频帧缓冲区大小，较大的缓冲区可以减少丢帧但增加延迟，较小的缓冲区可以减少延迟但可能增加丢帧
        self.data_buffer_size = data_buffer_size # 数据缓冲区大小，较大的缓冲区可以减少数据丢失但增加延迟，较小的缓冲区可以减少延迟但可能增加数据丢失
        self.send_data_freq = send_data_freq # 发送数据的频率，较高的频率可以提供更实时的控制但增加网络负载，较低的频率可以减少网络负载但可能导致控制不够及时
        # 使用队列来存储视频帧和控制数据，确保线程安全的数据传递和缓冲机制，避免在多线程环境下出现数据竞争和不一致的情况
        self.receive_data_queue = queue.Queue(maxsize=data_buffer_size) # 接收数据的队列，用于存储从 VR 头显接收到的控制数据（头部，手柄的姿态和按键状态）
        self.send_data_queue = queue.Queue(maxsize=data_buffer_size) # 发送数据的队列，用于存储要发送到 VR 头显的控制数据 （机器人摄像头画面）
        self.thread = None # 用于运行 WebRTC 连接的线程，确保 WebRTC 连接的异步操作不会阻塞主线程，允许同时处理其他任务（例如机器人控制、数据记录等）
        self.event_loop = None # WebRTC 连接的事件循环，负责处理 WebRTC 连接的异步事件和回调函数，确保 WebRTC 连接的正常运行和数据传输

    # 启动 WebRTC 连接，创建一个新的线程来运行 WebRTC 连接的事件循环，确保 WebRTC 连接的异步操作不会阻塞主线程，允许同时处理其他任务（例如机器人控制、数据记录等）
    async def channel_send_loop(self):
        last_data = None # 记录上一次成功发送的数据，以便在当前数据发送失败时重试发送
        while True:
            start_time = time.time()
            
            try:
                if self.channel is not None and self.channel.readyState == "open":
                    data = self.send_data_queue.get_nowait() # 从发送数据的队列中获取要发送的数据，如果队列为空则抛出 queue.Empty 异常
                    data = json.dumps(data) # 将数据转换为 JSON 格式的字符串，以便通过 WebRTC 数据通道发送
                    last_data = data  # 更新缓存
                    self.channel.send(data) # 发送给 VR 头显
            except Exception as e:
                # 如果发送数据失败
                try:
                    if last_data is not None:
                        self.channel.send(last_data) # 重试发送缓存的数据
                except Exception as e: 
                    print(f"Failed to send data: {e}")

            elapsed_time = time.time() - start_time # 计算发送耗时
            await asyncio.sleep(max(1/self.send_data_freq - elapsed_time, 0)) # 根据设定的发送频率调整发送间隔，确保按照指定的频率发送数据，同时避免过度占用 CPU 资源

    
    def receive_data(self) -> HeadsetData: #  -> HeadsetData 表示返回值类型
        try:
            data = self.receive_data_queue.get_nowait() # 从接收数据的队列中获取最新的数据，如果队列为空则抛出 queue.Empty 异常
            return data
        except queue.Empty:
            return None
    
    # 将机器人摄像头捕获的图像帧添加到视频轨道的缓冲区中，以便通过 WebRTC 连接发送到 VR 头显，确保图像帧的实时传输和显示，同时处理可能出现的缓冲区满的情况，避免丢帧或过度占用内存
    def send_images(self, left_image: np.ndarray, right_image: np.ndarray):
        try:
            if self.left_video_track is not None:
                self.left_video_track.add_frame(left_image) #TODO copy image? 把机器人左眼摄像头的一帧画面，塞进 WebRTC 视频轨道，让它实时发给 VR 头显
            if self.right_video_track is not None:
                self.right_video_track.add_frame(right_image) #TODO copy image?
        except Exception as e:
            print(f"Failed to send image: {e}")

    # 发送数据到VR头显
    def send_feedback(self, data: HeadsetFeedback):
        data = {
            # 头部、左右臂是否失去同步标志位
            'headOutOfSync': data.head_out_of_sync, 
            'leftOutOfSync': data.left_out_of_sync,
            'rightOutOfSync': data.right_out_of_sync,
            'info': data.info, 
            # 左右中臂的位姿
            'leftArmPosition': data.left_arm_position.tolist(),
            'leftArmRotation': data.left_arm_rotation.tolist(),
            'rightArmPosition': data.right_arm_position.tolist(),
            'rightArmRotation': data.right_arm_rotation.tolist(),
            'middleArmPosition': data.middle_arm_position.tolist(),
            'middleArmRotation': data.middle_arm_rotation.tolist(),
        }
        try:
            self.send_data_queue.put_nowait(data) # 将数据放进发送列队，如果队列已满则抛出 queue.Full 异常
        except queue.Full:
            try:
                self.send_data_queue.get_nowait() # 从发送队列中移除最旧的数据，为新的数据腾出空间
                self.send_data_queue.put_nowait(data) # 再次尝试将数据放进发送队列
            except (queue.Empty, queue.Full):
                pass

    def on_message(self, message):
        try:
            headset_data = HeadsetData() # 实例化HeadsetData对象，用于存储从 VR 头显接收到的控制数据（头部，手柄的姿态和按键状态）
            data = json.loads(message) # 将接收到的消息从 JSON 格式的字符串解析为 Python 字典对象，以便后续处理和使用
        except json.JSONDecodeError: # 如果消息不是有效的 JSON 格式，抛出 JSONDecodeError 异常，捕获该异常并打印错误信息，避免程序崩溃
            print("WebRTC: JSON decode error")
            return

        try:
            headset_data.h_pos[0] = data['HPosition']['x']
            headset_data.h_pos[1] = data['HPosition']['y']
            headset_data.h_pos[2] = data['HPosition']['z']
            headset_data.h_quat[0] = data['HRotation']['x']
            headset_data.h_quat[1] = data['HRotation']['y']
            headset_data.h_quat[2] = data['HRotation']['z']
            headset_data.h_quat[3] = data['HRotation']['w']
            headset_data.l_pos[0] = data['LPosition']['x']
            headset_data.l_pos[1] = data['LPosition']['y']
            headset_data.l_pos[2] = data['LPosition']['z']
            headset_data.l_quat[0] = data['LRotation']['x']
            headset_data.l_quat[1] = data['LRotation']['y']
            headset_data.l_quat[2] = data['LRotation']['z']
            headset_data.l_quat[3] = data['LRotation']['w']
            headset_data.l_thumbstick_x = data['LThumbstick']['x'] # 左手柄的拇指摇杆在 x 轴上的位置，通常范围在 -1 到 1 之间，表示摇杆向左或向右的程度
            headset_data.l_thumbstick_y = data['LThumbstick']['y'] # 左手柄的拇指摇杆在 y 轴上的位置，通常范围在 -1 到 1 之间，表示摇杆向前或向后的程度
            headset_data.l_index_trigger = data['LIndexTrigger'] # 左手柄的食指触发器的值，通常范围在 0 到 1 之间，表示触发器被按下的程度
            headset_data.l_hand_trigger = data['LHandTrigger'] # 左手柄的握持触发器的值，通常范围在 0 到 1 之间，表示握持触发器被按下的程度
            headset_data.l_button_one = data['LButtonOne'] # 左手柄的按钮一的状态，布尔值，表示按钮是否被按下
            headset_data.l_button_two = data['LButtonTwo'] # 左手柄的按钮二的状态，布尔值，表示按钮是否被按下
            headset_data.l_button_thumbstick = data['LButtonThumbstick'] # 左手柄的拇指按键的状态，布尔值，表示拇指按键是否被按下
            headset_data.r_pos[0] = data['RPosition']['x']
            headset_data.r_pos[1] = data['RPosition']['y']
            headset_data.r_pos[2] = data['RPosition']['z']
            headset_data.r_quat[0] = data['RRotation']['x']
            headset_data.r_quat[1] = data['RRotation']['y']
            headset_data.r_quat[2] = data['RRotation']['z']
            headset_data.r_quat[3] = data['RRotation']['w']
            headset_data.r_thumbstick_x = data['RThumbstick']['x']
            headset_data.r_thumbstick_y = data['RThumbstick']['y']
            headset_data.r_index_trigger = data['RIndexTrigger']
            headset_data.r_hand_trigger = data['RHandTrigger']
            headset_data.r_button_one = data['RButtonOne']
            headset_data.r_button_two = data['RButtonTwo']
            headset_data.r_button_thumbstick = data['RButtonThumbstick']
            # 将头部、左手柄、右手柄的位姿从左手坐标系转换为右手坐标系，以便在机器人世界中使用
            headset_data.h_pos, headset_data.h_quat = convert_left_to_right_coordinates(headset_data.h_pos, headset_data.h_quat)
            headset_data.l_pos, headset_data.l_quat = convert_left_to_right_coordinates(headset_data.l_pos, headset_data.l_quat)
            headset_data.r_pos, headset_data.r_quat = convert_left_to_right_coordinates(headset_data.r_pos, headset_data.r_quat)
        except KeyError:
            print("[RobotWebRTC] Key error") 
            return

        try:
            self.receive_data_queue.put_nowait(headset_data) # 将接收到的数据放进接收队列，如果队列已满则抛出 queue.Full 异常
        except queue.Full:
            try:
                self.receive_data_queue.get_nowait() # 从接收队列中移除最旧的数据，为新的数据腾出空间
                self.receive_data_queue.put_nowait(headset_data) # 再次尝试将数据放进接收队列
            except (queue.Empty, queue.Full):
                pass

    # 运行 WebRTC 连接的主要逻辑，包括创建数据通道、视频轨道，生成并发送 offer，等待并处理 answer，以及处理连接状态变化等，确保 WebRTC 连接的正常建立和维护，同时处理可能出现的异常情况，保证连接的稳定性和可靠性
    async def run_offer(self):
        # create data channel
        self.channel = self.pc.createDataChannel("control") # 创建一条名叫 control 的数据通道

        # 注册一个事件监听器，当数据通道打开时触发，打印 "Data channel is open." 的消息，表示数据通道已经成功建立并准备好进行数据传输
        @self.channel.on("open") 
        def on_open():
            print("Data channel is open.")
        self.channel.on("message", self.on_message)       

        # create video track
        # 创建两个视频轨道，分别用于发送左目和右目摄像头的视频帧到 VR 头显
        self.left_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        self.left_video_sender = self.pc.addTrack(self.left_video_track)
        force_codec(self.pc, self.left_video_sender, 'video/VP8') # 强制使用 VP8 编解码器进行视频传输，确保视频的兼容性和性能，同时避免在不同设备和浏览器之间出现编解码器不兼容的问题

        # create video track
        self.right_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        self.right_video_sender = self.pc.addTrack(self.right_video_track)
        force_codec(self.pc, self.right_video_sender, 'video/VP8')


        # create offer and place in firestore     
        print("WebRTC: Running offer...")  
        await self.pc.setLocalDescription(await self.pc.createOffer()) # 创建一个 WebRTC offer，并将其设置为本地描述，准备发送给 VR 头显
        call_doc = self.db.collection(self.password).document(self.robotId) # 在 Firestore 中创建一个文档，使用之前写的password 和 robotId，用于存储当前机器人的信令数据（offer 和 answer）
        call_doc.set( # 将 offer 的 SDP 和类型存储在 Firestore 文档中，供 VR 头显读取和响应
            {
                'sdp': self.pc.localDescription.sdp, # offer 的 Session Description Protocol (SDP) 内容，包含了媒体协商的信息，例如支持的编解码器、网络信息等
                'type': self.pc.localDescription.type  # offer 的类型，通常为 "offer"，表示这是一个 WebRTC offer，用于发起连接请求
            }
        )

        # wait for answer from firestore
        # 监听 Firestore 文档的变化，等待 VR 头显将 answer 的 SDP 和类型写入文档，当检测到文档中出现 answer 时，获取 answer 的 SDP 和类型，并将其设置为远程描述，完成 WebRTC 连接的建立
        data = None
        def answer_callback(doc_snapshot, changes, read_time):
            nonlocal data
            for doc in doc_snapshot:
                if self.pc.remoteDescription is None and doc.to_dict()['type'] == 'answer':
                    data = doc.to_dict()
        doc_watch = call_doc.on_snapshot(answer_callback) 
        print('WebRTC: Waiting for answer...')
        while data is None:
            await asyncio.sleep(1/30) # 每1/30秒检查一次 Firestore 文档是否有 answer
        print('WebRTC: Answer received.')
        doc_watch.unsubscribe() # 停止监听 Firestore 文档的变化，避免在连接建立后继续监听文档，节省资源和避免不必要的回调触发

        # set remote description from answer
        # 将 answer 的 SDP 和类型设置为远程描述，完成 WebRTC 连接的建立，使得双方可以开始进行媒体和数据的传输
        await self.pc.setRemoteDescription(RTCSessionDescription(
            sdp=data['sdp'],
            type=data['type']
        ))

        # delete firestore call document
        # 在连接建立后，删除 Firestore 中用于信令交换的文档，清理资源并避免其他客户端误读到过期的信令数据
        call_doc = self.db.collection(self.password).document(self.robotId)
        call_doc.delete()

        # add event listener for connection close
        # 注册一个事件监听器，当 WebRTC 连接的 ICE 连接状态发生变化时触发，如果连接状态变为 "closed"，则打印 "WebRTC: Connection closed, restarting..." 的消息，并调用 restart_connection() 方法重新启动连接，确保在连接意外关闭时能够自动恢复连接，保持系统的稳定性和可靠性
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if self.pc.iceConnectionState == "closed":
                print("WebRTC: Connection closed, restarting...")
                await self.restart_connection()

    async def restart_connection(self): 
        # close current peer connection
        await self.pc.close() # 关闭当前的 WebRTC 连接，释放相关资源，准备重新建立连接

        # create new peer connection
        # 重新创建一个新的 WebRTC 连接，配置相同的 STUN 和 TURN 服务器，以便在连接关闭后能够快速恢复连接，继续进行媒体和数据的传输
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                RTCIceServer(self.turn_server_url, self.turn_server_username, self.turn_server_password)
            ])
        )

        # run offer again
        await self.run_offer() # 重新运行 offer 的逻辑，创建数据通道、视频轨道，生成并发送 offer，等待并处理 answer，以及处理连接状态变化等，确保 WebRTC 连接能够成功重新建立，并继续进行媒体和数据的传输

    def run_in_thread(self):
        def run(loop: asyncio.AbstractEventLoop):  # 
            asyncio.set_event_loop(loop) # 将当前线程的事件循环设置为传入的 loop 参数，确保在当前线程中使用该事件循环来处理 WebRTC 连接的异步事件和回调函数
            loop.run_until_complete(self.run_offer())   # 等待握手完成，建立 WebRTC 连接
            loop.create_task(self.channel_send_loop()) # 握手成功后，定期通过 WebRTC 通道将数据发送到 VR 头显
            loop.run_forever() # 启动事件循环，开始处理 WebRTC 连接的异步事件和回调函数，保持连接的正常运行和数据传输，直到调用 loop.stop() 来停止事件循环

        self.event_loop = asyncio.new_event_loop() # 创建一个新的事件循环，确保 WebRTC 连接的异步操作在独立的事件循环中运行，避免与主线程的事件循环冲突
        self.thread = threading.Thread(target=run, args=(self.event_loop,)) # 创建一个新的线程，目标函数为 run，传入事件循环作为参数，确保 WebRTC 连接的异步操作在独立的线程中运行，避免阻塞主线程
        self.thread.start()

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.event_loop.stop() # 停止事件循环
            self.thread.join() # 等待线程结束，释放相关资源

if __name__ == "__main__":
    import time # 确保导入了 time 模块
    try:
        headset = WebRTCHeadset() # 创建实例对象
        headset.run_in_thread() # 启动 WebRTC 连接

        # 假设我们想以大约 30 FPS 的频率进行循环测试
        loop_rate = 30
        sleep_duration = 1.0 / loop_rate

        while True:
            loop_start = time.time()

            data = headset.receive_data() # 从接收队列中获取最新的数据，如果队列为空则返回 None，确保能够及时获取从 VR 头显接收到的控制数据（头部，手柄的姿态和按键状态），并处理可能出现的队列空的情况，避免程序崩溃
            if data is not None:
                print(f"Received data: {data.h_pos}, {data.h_quat}") # 打印接收到的数据中的头部位置和旋转信息，验证数据的正确接收和解析

            feedback = HeadsetFeedback() # 创建对象，用于存储要发送到 VR 头显的反馈信息，包括头部、左右臂是否失去同步标志位，左右中臂的位姿，以及其他信息
            feedback.info = f"Hello from python: {time.time()}"
            headset.send_feedback(feedback) # 将反馈信息放进发送队列，如果队列已满则移除最旧的数据并重试，确保能够及时将反馈信息发送到 VR 头显，同时处理可能出现的队列满的情况，避免程序崩溃

            # headset.send_image(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)) # 发送一帧随机生成的图像数据到 VR 头显，验证视频传输的功能和性能，确保图像帧能够正确地通过 WebRTC 视频轨道发送到 VR 头显，并处理可能出现的异常情况，避免程序崩溃

            # 生成一帧随机的雪花噪点图
            random_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
            # 把这张图同时发给左眼和右眼
            headset.send_images(random_img, random_img)

            # 4. 控制循环频率，避免 CPU 100% 占用
            elapsed = time.time() - loop_start
            if elapsed < sleep_duration:
                time.sleep(sleep_duration - elapsed)
    except KeyboardInterrupt: # 捕获键盘中断信号（例如按下 Ctrl+C），当用户希望终止程序时，执行以下代码来安全地关闭 WebRTC 连接并退出程序
        print("Shutting down...")
        headset.close() # 建议加上这句，确保线程和事件循环被优雅关闭
        os._exit(42)
