# -- coding: UTF-8
import os
import numpy as np
import torch
import torchvision
import argparse
import dm_env
import requests
import collections
from collections import deque
import logging
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from base64 import b64decode
import threading
import cv2
import signal
import subprocess
import multiprocessing
from tqdm import tqdm

from idm.idm import *
from utils.inference.process import process_image


start_flag = False
exit_flag = False
num_fails = 0
logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, help='Save directory.', default='output/grm')
    parser.add_argument('--task_json', type=str, help='Task json.', default='tasks/train/pick_the_red_apple_using_left_arm.json')
    parser.add_argument('--prompt_idx', type=int, help='Prompt index.', default=0)
    parser.add_argument('--inference_prefix', type=str, help='Inference prefix.', default="0")
    parser.add_argument('--video_file', type=str, help='Video path.', default="video.mp4")
    parser.add_argument('--model_name', type=str, help='Model name.', default="mask")
    parser.add_argument('--load_from', type=str, help='Model name.', default="mask")

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_front', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_side'], required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_left_cmd_topic', action='store', type=str, help='master_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--master_arm_right_cmd_topic', action='store', type=str, help='master_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)

    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true', help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                            # default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)
                            default=[0.012, 0.012, 0.012, 0.02, 0.02, 0.02, 0.5], required=False)
                            # default=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.1], required=False)

    parser.add_argument('--use_image_front', action='store_true', help='use_front_image',
                        default=False, required=False)
    parser.add_argument('--use_image_left', action='store_true', help='use_left_image',
                        default=False, required=False)
    parser.add_argument('--use_image_right', action='store_true', help='use_right_image',
                        default=False, required=False)
    parser.add_argument('--use_image_high', action='store_true', help='use_high_image',
                        default=False, required=False)
    parser.add_argument('--use_image_side', action='store_true', help='use_side_image',
                        default=False, required=False)

    parser.add_argument('--use_master_left', action='store_true', help='use_master_left',
                        default=False, required=False)
    parser.add_argument('--use_master_right', action='store_true', help='use_master_right',
                        default=False, required=False)

    parser.add_argument('--use_puppet_left', action='store_true', help='use_puppet_left',
                        default=False, required=False)
    parser.add_argument('--use_puppet_right', action='store_true', help='use_puppet_right',
                        default=False, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=30, required=False)

    parser.add_argument('--use_keyboard_end', action='store_true', help='use_keyboard_end',
                        default=False, required=False)
    parser.add_argument('--qpos_path', action='store', type=str, help='qpos_path',
                        default='left_qpos_for_random_collect.pt', required=False)
    infer_args = parser.parse_args()
    return infer_args


def signal_exit(*infer_args):
    global exit_flag
    exit_flag = True


def save_worker(ffmpeg_cmd, images):
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    logger.info(f"Saved video length: {len(images)}")
    for image in images:
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def save_data(infer_args, timesteps, save_dir):
    camera_used = {infer_args.camera_names[0]: infer_args.use_image_front, infer_args.camera_names[1]: infer_args.use_image_left, infer_args.camera_names[2]: infer_args.use_image_right, infer_args.camera_names[3]: infer_args.use_image_high, infer_args.camera_names[4]: infer_args.use_image_side}
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
    }

    for cam_name in infer_args.camera_names:
        if camera_used[cam_name]:
            data_dict[f'/observations/images/{cam_name}'] = []

    if len(timesteps) == 0:
        logger.warning(f"No valid data")
        return
    actions = np.array([ts.observation['qpos'] for ts in timesteps])
    moved_frames = np.where(np.linalg.norm(np.array(actions[1:]) - np.array(actions[:-1]), axis=1) > 0.01)[0]
    if len(moved_frames) == 0:
        logger.warning(f"No valid data")
        return
    data_size = moved_frames[-1] - moved_frames[0]
    logger.info(f"Saved length: {data_size}")
    qpos = []
    for frame_idx in range(moved_frames[0], moved_frames[-1]):
        ts = timesteps[frame_idx]
        qpos.append(ts.observation['qpos'])
        for cam_name in infer_args.camera_names:
            if camera_used[cam_name]:
                data_dict[f'/observations/images/{cam_name}'].append(cv2.imencode('.jpg', ts.observation['images'][cam_name], [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes())
                
    width, height = 640, 480
    fps = 30
    workers = []
    for cam_name in infer_args.camera_names:
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',                    # 覆盖输出文件
            '-f', 'rawvideo',        # 输入格式为原始视频
            '-vcodec', 'rawvideo',   # 输入编码器为原始视频
            '-s', f'{width}x{height}',  # 分辨率
            '-pix_fmt', 'bgr24',     # OpenCV 默认像素格式
            '-r', str(fps),          # 帧率
            '-i', '-',               # 从标准输入读取数据
            '-c:v', 'libx264',       # 使用 H.264 编码
            '-preset', 'fast',       # 编码速度与质量平衡
            '-crf', '10',            # 质量参数（0-51，值越小质量越高）
            # '-preset', 'ultrafast',       # 编码速度与质量平衡
            # '-crf', '0',            # 质量参数（0-51，值越小质量越高）
            '-threads', '1',
            '-pix_fmt', 'yuv420p',   # 兼容性像素格式
            '-loglevel', 'error',
            os.path.join(save_dir, cam_name + '.mp4')
        ]
        # save_worker(ffmpeg_cmd, data_dict[f'/observations/images/{cam_name}'])
        workers.append(multiprocessing.Process(target=save_worker, args=(ffmpeg_cmd, data_dict[f'/observations/images/{cam_name}'])))

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    qpos_path = os.path.join(save_dir, 'qpos.pt')

    # Save qpos tensor
    qpos_tensor = torch.tensor(np.array(qpos), dtype=torch.float32)
    torch.save(qpos_tensor, qpos_path)



class IDM:

    def __init__(self, model_name, load_from, video_path, save_dir):
        self.model_name = model_name
        self.load_from = load_from
        self.video_path = video_path
        self.save_dir = save_dir
        
        

def inference_idm(model_name, load_from, video_path, save_dir):
    dinov2_processor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((518, 518)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    net = IDM(model_name=model_name, output_dim=14).cuda() # , dinov2_name="facebook/dinov2-with-registers-base", freeze_dinov2=False
    if os.path.isfile(load_from):
        with torch.cuda.stream(torch.cuda.Stream()):
            loaded_dict = torch.load(load_from, weights_only=False,mmap=True)
            net.load_state_dict(loaded_dict["model_state_dict"])
    else:
        logger.warning("Please give a valid checkpoint.")
        exit(0)
    net.eval()

    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.tensor(frame, device="cuda"))
    video.release()

    actions = []
    with torch.no_grad():
        for i, image in tqdm(enumerate(frames)):
            image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 518, 518]
            inputs = dinov2_processor(image / 255)
            if 'split' in model_name:
                # inputs = process_image(inputs.permute(1, 2, 0)).unsqueeze(1)
                # process_image() needs [H, W, C] as input
                processed_image = process_image(inputs.squeeze(0).permute(1, 2, 0)).unsqueeze(1)  # [4, 1, 3, H, W]
            else:
                processed_image = inputs
            output = net(processed_image, return_mask=True) #, return_mask=True
            if isinstance(output, tuple):
                output, mask = output
            else:
                mask = None
            action = output[0]
            actions.append(action)
            if i % 10 == 0:
                sample_image = inputs[0].detach().cpu().numpy()
                sample_image = np.transpose(sample_image, (1, 2, 0))
                sample_image *= np.array([0.229, 0.224, 0.225])
                sample_image += np.array([0.485, 0.456, 0.406])
                sample_image = np.clip(sample_image, 0, 1)
                sample_image = (sample_image * 255).astype(np.uint8)[:, :, [2, 1, 0]]
                cv2.imwrite(os.path.join(save_dir, f'sample_image_{i}.png'), sample_image)
                if mask is not None:
                    sample_mask = mask[0].detach().cpu().numpy()
                    sample_mask = np.transpose(sample_mask, (1, 2, 0))
                    sample_mask = np.where(sample_mask >= 0.5, sample_image, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f'sample_mask_{i}.png'), sample_mask)
    actions = torch.stack(actions, dim=0).cpu().numpy()
    return actions


def modify_actions(actions):
    # for dim in [6, 13]:
    for dim in [6, 13]:
        actions[:, dim] -= 0.4 * (1 - np.argsort(actions[:, dim]) / len(actions[:, dim]))
        # if actions[:, dim] < 0.5:
            # actions[:, dim] = 0
        gripper = actions[:, dim]
        mask = gripper < 0.5
        gripper[mask] = 0  # Set values less than 0.5 to 0
        actions[:, dim] = gripper 
    return actions


def main():
    infer_args = get_arguments()
    save_dir = os.path.join(infer_args.save_dir, infer_args.task_json.split("/")[-1].split(".json")[0], infer_args.inference_prefix)
    os.makedirs(save_dir, exist_ok=True)

    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(save_dir, "log_idm.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info(infer_args)
    actions = inference_idm(infer_args.model_name, infer_args.load_from, os.path.join(save_dir, infer_args.video_file), save_dir)
    actions = modify_actions(actions)



if __name__ == '__main__':
    main()
