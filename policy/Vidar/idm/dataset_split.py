import os
import random
import torch
import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision
from transformers import AutoImageProcessor
from PIL import Image, ImageOps
import torch.nn as nn
import imageio.v3 as iio
from io import BytesIO
import av
import bisect

from tqdm import tqdm
from idm.preprocessor import DinoPreprocessor, segment_robot_arms  # Import the preprocessor


class DataSet(Dataset):
    """Dataset for inverse dynamics model

    Contents of Dataset:  If args.load_mp4==True, the data_file_path can be an mp4 file or an qpos file. 
    like:  {task_name}/f'episode_{episode_idx}.mp4', {task_name}/f'episode_{episode_idx}_qpos.pt', where:
    - the mp4 is saved as cv2.VideoWriter({task_name}/f'episode_{episode_idx}.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), fps=30, (width=640, height=720)). The length of the video is episode_len.
    - the 14-dim qpos of each trajectory is in the form of torch.zeros([episode_len, dim=14]).dtype(float32).cpu(), 
    saved with torch.save()
    - the json file is saved as {task_name}.json', which contains the information of the trajectory, but the caption
    is "Random Wing", not specfically processed yet (no use for IDM, but useful for video generation model).
    """
    
    def __init__(self, args, dataset_path, disable_pbar=False, type="train", preprocessor=None):
        self.data = []
        self.dataset_path = dataset_path
        self.type = type
        self.height = 720  # 480 + 240
        self.width = 640        
        self.video_bytes = []  # 改为存储原始字节而不是打开的容器
        self.keyframe_indices = []
        self.video_metadata = []
        self.qpos_data = []
        self.video_lengths = []
        self.preprocessor = preprocessor
        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        for task_name in os.listdir(dataset_path):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                continue
            for file_name in tqdm(os.listdir(task_path), desc=f"Loading videos from {task_name}", disable=disable_pbar):
                if file_name.endswith('.mp4'):
                    episode_idx = file_name.split('_')[1].split('.')[0]
                    video_path = os.path.join(task_path, file_name)
                    qpos_path = os.path.join(task_path, f'episode_{episode_idx}_qpos.pt')
                    
                    if not os.path.exists(qpos_path):
                        print(f"Skipping {video_path} - no matching qpos file")
                        continue
                        
                    # Check video length without keeping capture open
                    cap = cv2.VideoCapture(video_path)
                    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Load qpos
                    qpos = torch.load(qpos_path)
                    
                    if video_length < 30:
                        print(f"Skipping {video_path} - too short")
                        continue
                        
                    # 修改视频加载方式
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()

                    # 将容器初始化移到此处保证元数据同步
                    with av.open(BytesIO(video_bytes)) as container:
                        stream = container.streams.video[0]

                        # 构建关键帧索引
                        keyframes = []
                        for packet in container.demux(stream):
                            if packet.is_keyframe:
                                keyframes.append(packet.pts)

                        # 存储元数据
                        pts_per_frame = int((1 / stream.average_rate) / stream.time_base)
                        self.video_metadata.append({
                            'fps': float(stream.average_rate),
                            'width': stream.width,
                            'height': stream.height,
                            'time_base': stream.time_base,
                            'duration': stream.duration,
                            'pts_per_frame': pts_per_frame
                        })
                        self.keyframe_indices.append(keyframes)

                    # 确保所有列表同步添加
                    self.video_bytes.append(video_bytes)
                    self.qpos_data.append(qpos)
                    self.video_lengths.append(video_length)
        
        self.data_begin = np.cumsum([0] + self.video_lengths[:-1])
        self.data_end = np.cumsum(self.video_lengths)
        # print("self.data_begin, self.data_end", self.data_begin, self.data_end)
        # print("data len", self.__len__())

    def __len__(self):
        return self.data_end[-1]

    def __getitem__(self, idx):
        # 修改边界检查逻辑
        video_idx = np.searchsorted(self.data_end, idx, side='right')
        if video_idx < 0 or video_idx >= len(self.video_bytes):
            raise IndexError(f"Index {idx} out of bounds")
        
        # 确保所有元数据存在
        if (video_idx >= len(self.keyframe_indices) or
            video_idx >= len(self.video_metadata)):
            raise RuntimeError(f"Metadata missing for video {video_idx}")
        
        local_idx = idx - self.data_begin[video_idx]
        
        # 每次使用时从内存字节创建新容器
        video_bytes = BytesIO(self.video_bytes[video_idx])
        with av.open(video_bytes, format='mp4') as container:
            stream = container.streams.video[0]

            # 优化点1：使用预计算的每帧pts增量
            target_pts = local_idx * self.video_metadata[video_idx]['pts_per_frame']

            # 优化点2：二分查找加速关键帧定位
            keyframes = self.keyframe_indices[video_idx]
            index = bisect.bisect_right(keyframes, target_pts) - 1
            nearest_keyframe = keyframes[max(index, 0)]  # 确保不越界

            # 优化点3：精确跳转并限制解码范围
            container.seek(nearest_keyframe, stream=stream, backward=False, any_frame=False)

            # 直接定位到目标帧附近
            for frame in container.decode(stream):
                if frame.pts >= target_pts:
                    frame_array = frame.to_ndarray(format='rgb24')
                    break
            else:
                raise RuntimeError("Frame not found")

        pos = self.qpos_data[video_idx][local_idx]
        image = Image.fromarray(frame_array)
        
        # flip:
        # if random.random() < 0.5:
        #     image, pos = self.preprocessor.handle_flip(image, pos)
        
        # ---------------- split and preprocess image --------------------
        # 新增区域分割处理
        
        np_image = np.array(image)  # [H, W, C]
        h, w = np_image.shape[:2]
        
        # 分割机械臂区域
        _, arm_boxes = segment_robot_arms(np_image)  # arm_boxes就是绝对坐标
        left_split = arm_boxes['left_split'] / w
        right_split = arm_boxes['right_split'] / w
        arm_split = arm_boxes['arm_gripper_split'] / h
        grip_split = arm_boxes['gripper_split'] / w 
        
        
        processed_image = self.preprocessor.process_image(image)  # torch.Tensor[3, H, W], 0~1
        
        new_h, new_w = processed_image.shape[1:]
        # 先preprocess, 再apply noise 
        # 生成随机噪声 np:[0~255], [h, w, 3], torch:[0~1], [3, h, w] (指new_h, new_w)
        noise = torch.rand(3, new_h, new_w)
        # noise过一下dino的normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        noise = (noise - mean) / std
        
        left_split = int(left_split * new_w)
        right_split = int(right_split * new_w)
        arm_split = int(arm_split * new_h)
        grip_split = int(grip_split * new_w)
        
        region_batches = []
        
        # 保存processed_image
        # if idx < 10:
        #     Image.fromarray((processed_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f"output/processed_image_{idx}.png")
            
        # 分割四个区域, 将区域外部分替换为噪声
        # for i in range(4):
        #     mask = torch.zeros_like(processed_image)  # torch.Tensor[3, H, W]
        #     if i == 0: mask[:, :arm_split, :left_split] = 1  # 左上
        #     elif i == 1: mask[:, arm_split:, :grip_split] = 1  # 左下
        #     elif i == 2: mask[:, :arm_split, right_split:] = 1  # 右上
        #     elif i == 3: mask[:, arm_split:, grip_split:] = 1  # 右下
            
        #     # 合并原图和噪声
        #     # import pdb; pdb.set_trace()
        #     region_batches.append((processed_image * mask) + (noise * (1 - mask)))  # torch.Tensor[3, H, W]
        
        # 分割四个区域，将每个区域放缩到原始大小
        for i in range(4):
            # 定义四个区域的坐标 (x1, y1, x2, y2)
            if i == 0:   # 左上
                region = processed_image[:, :arm_split, :left_split]
            elif i == 1: # 左下
                region = processed_image[:, arm_split:, :grip_split]
            elif i == 2: # 右上
                region = processed_image[:, :arm_split, right_split:]
            elif i == 3: # 右下
                region = processed_image[:, arm_split:, grip_split:]
            
            # 使用插值方法将区域放缩到 new_h x new_w
            # 需要先扩展维度以符合F.interpolate的输入要求 [B,C,H,W]
            region = region.unsqueeze(0)  # [1,3,h,w]
            resized_region = torch.nn.functional.interpolate(
                region, 
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            region_batches.append(resized_region.squeeze(0))  # 去掉批次维度 [3,H,W]
        
        # 按idx保存一些图片, 要转成Image的格式
        # if idx < 10:
        #     if not os.path.exists("output"):
        #         os.makedirs("output")
        #     for i in range(4):
        #         Image.fromarray((region_batches[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f"output/region_{idx}_{i}.png")
        
        region_batches = torch.stack(region_batches)  # torch.Tensor[4, 3, H, W]
        return pos, region_batches

