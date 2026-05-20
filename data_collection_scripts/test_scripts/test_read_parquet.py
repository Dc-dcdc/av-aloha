import pandas as pd
import cv2
import os

# 1. 读取 parquet
df = pd.read_parquet("src/av-aloha/data_collection_scripts/data/sim_insert_peg/train-00000-of-00001.parquet")

# 2. 提取帧指针信息
camera_column = 'observation.images.zed_cam_left'
pointer = df[camera_column].iloc[0]

print("Parquet 里存的数据其实是：", pointer)

# 3. 顺藤摸瓜，去读视频文件的分辨率
# 注意：这里的路径可能需要根据你实际的文件夹结构做一下拼接
video_path = os.path.join("/home/dc/下载/observation.images.zed_cam_left_episode_000000.mp4")

cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"顺藤摸瓜找到的视频分辨率为: 宽度={width}, 高度={height}")