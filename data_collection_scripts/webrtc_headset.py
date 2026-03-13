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


    async def get_frame(self) -> np.ndarray:
        while True:
            try:
                frame = self.queue.get_nowait()
                self.last_frame = frame
                return frame
            except queue.Empty:
                if self.last_frame is not None:
                    return self.last_frame.copy()
                await asyncio.sleep(0)
        

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

    def add_frame(self, frame):
        # try to put but if pull pop the oldest frame
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
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
                self.left_video_track.add_frame(left_image) #TODO copy image?
            if self.right_video_track is not None:
                self.right_video_track.add_frame(right_image) #TODO copy image?
        except Exception as e:
            print(f"Failed to send image: {e}")

    def send_feedback(self, data: HeadsetFeedback):
        data = {
            'headOutOfSync': data.head_out_of_sync,
            'leftOutOfSync': data.left_out_of_sync,
            'rightOutOfSync': data.right_out_of_sync,
            'info': data.info,
            'leftArmPosition': data.left_arm_position.tolist(),
            'leftArmRotation': data.left_arm_rotation.tolist(),
            'rightArmPosition': data.right_arm_position.tolist(),
            'rightArmRotation': data.right_arm_rotation.tolist(),
            'middleArmPosition': data.middle_arm_position.tolist(),
            'middleArmRotation': data.middle_arm_rotation.tolist(),
        }
        try:
            self.send_data_queue.put_nowait(data)
        except queue.Full:
            try:
                self.send_data_queue.get_nowait()
                self.send_data_queue.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass

    def on_message(self, message):
        try:
            headset_data = HeadsetData()
            data = json.loads(message)
        except json.JSONDecodeError:
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
            headset_data.l_thumbstick_x = data['LThumbstick']['x']
            headset_data.l_thumbstick_y = data['LThumbstick']['y']
            headset_data.l_index_trigger = data['LIndexTrigger']
            headset_data.l_hand_trigger = data['LHandTrigger']
            headset_data.l_button_one = data['LButtonOne']
            headset_data.l_button_two = data['LButtonTwo']
            headset_data.l_button_thumbstick = data['LButtonThumbstick']
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
            headset_data.h_pos, headset_data.h_quat = convert_left_to_right_coordinates(headset_data.h_pos, headset_data.h_quat)
            headset_data.l_pos, headset_data.l_quat = convert_left_to_right_coordinates(headset_data.l_pos, headset_data.l_quat)
            headset_data.r_pos, headset_data.r_quat = convert_left_to_right_coordinates(headset_data.r_pos, headset_data.r_quat)
        except KeyError:
            print("[RobotWebRTC] Key error") 
            return

        try:
            self.receive_data_queue.put_nowait(headset_data)
        except queue.Full:
            try:
                self.receive_data_queue.get_nowait()
                self.receive_data_queue.put_nowait(headset_data)
            except (queue.Empty, queue.Full):
                pass

    async def run_offer(self):
        # create data channel
        self.channel = self.pc.createDataChannel("control")
        @self.channel.on("open")
        def on_open():
            print("Data channel is open.")
        self.channel.on("message", self.on_message)       

        # create video track
        self.left_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        self.left_video_sender = self.pc.addTrack(self.left_video_track)
        force_codec(self.pc, self.left_video_sender, 'video/VP8')

        # create video track
        self.right_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        self.right_video_sender = self.pc.addTrack(self.right_video_track)
        force_codec(self.pc, self.right_video_sender, 'video/VP8')


        # create offer and place in firestore     
        print("WebRTC: Running offer...")  
        await self.pc.setLocalDescription(await self.pc.createOffer())
        call_doc = self.db.collection(self.password).document(self.robotId)
        call_doc.set(
            {
                'sdp': self.pc.localDescription.sdp,
                'type': self.pc.localDescription.type
            }
        )

        # wait for answer from firestore
        data = None
        def answer_callback(doc_snapshot, changes, read_time):
            nonlocal data
            for doc in doc_snapshot:
                if self.pc.remoteDescription is None and doc.to_dict()['type'] == 'answer':
                    data = doc.to_dict()
        doc_watch = call_doc.on_snapshot(answer_callback)
        print('WebRTC: Waiting for answer...')
        while data is None:
            await asyncio.sleep(1/30)
        print('WebRTC: Answer received.')
        doc_watch.unsubscribe()

        # set remote description from answer
        await self.pc.setRemoteDescription(RTCSessionDescription(
            sdp=data['sdp'],
            type=data['type']
        ))

        # delete firestore call document
        call_doc = self.db.collection(self.password).document(self.robotId)
        call_doc.delete()

        # add event listener for connection close
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if self.pc.iceConnectionState == "closed":
                print("WebRTC: Connection closed, restarting...")
                await self.restart_connection()

    async def restart_connection(self):
        # close current peer connection
        await self.pc.close()

        # create new peer connection
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                RTCIceServer(self.turn_server_url, self.turn_server_username, self.turn_server_password)
            ])
        )

        # run offer again
        await self.run_offer() 

    def run_in_thread(self):
        def run(loop: asyncio.AbstractEventLoop):  
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_offer())  
            loop.create_task(self.channel_send_loop())
            loop.run_forever()

        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run, args=(self.event_loop,))
        self.thread.start()

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.event_loop.stop()
            self.thread.join()

if __name__ == "__main__":
    try:
        headset = WebRTCHeadset()
        headset.run_in_thread()
        
        while True:
            data = headset.receive_data()
            if data is not None:
                print(f"Received data: {data.h_pos}, {data.h_quat}")

            feedback = HeadsetFeedback()
            feedback.info = f"Hello from python: {time.time()}"
            headset.send_feedback(feedback)

            headset.send_image(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)
