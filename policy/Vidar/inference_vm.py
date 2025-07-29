# -- coding: UTF-8
import numpy as np
import json
import requests
import cv2
import urllib3
from base64 import b64encode, b64decode
import os
import multiprocessing
import argparse
import subprocess
import logging

from utils.inference.process import concatenate_images
from utils.inference.select_video_api import process_responses
from utils.inference.configs import *


logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, help='Save directory.', default='output/grm')
    parser.add_argument('--task_json', type=str, help='Task json.', default='tasks/train/pick_the_red_apple_using_left_arm.json')
    parser.add_argument('--prompt_idx', type=int, help='Prompt index.', default=0)
    parser.add_argument('--inference_prefix', type=str, help='Inference prefix.', default="0")
    parser.add_argument('--tts', action='store_true', help='Test-time scaling.')
    parser.add_argument('--ports', nargs='+', default=[23990], type=int, help='Ports')
    parser.add_argument('--eval_ports', nargs='+', default=[], type=int, help='Eval ports') # 23933, 23934
    infer_args = parser.parse_args()
    return infer_args


def save_video(ffmpeg_cmd, images):
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    for image in images:
        img = cv2.imdecode(np.frombuffer(b64decode(image), np.uint8), cv2.IMREAD_COLOR)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def save_videos(videos, width, height, fps=8):
    workers = []
    for k, v in videos.items():
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
            '-preset', 'veryslow',       # 编码速度与质量平衡
            '-crf', '10',            # 质量参数（0-51，值越小质量越高）
            # '-preset', 'ultrafast',       # 编码速度与质量平衡
            # '-crf', '0',            # 质量参数（0-51，值越小质量越高）
            '-threads', '1',
            '-pix_fmt', 'yuv420p',   # 兼容性像素格式
            '-loglevel', 'error',
            k
        ]
        # save_worker(ffmpeg_cmd, data_dict[f'/observations/images/{cam_name}'])
        workers.append(multiprocessing.Process(target=save_video, args=(ffmpeg_cmd, v)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def get_cam_frame(camera='high'):
    match camera:
        case "high":
            port = 23000
        case "left":
            port = 23001
        case "right":
            port = 23002
    frame = requests.get(f'http://localhost:{port}').json()
    frame['image'] = np.frombuffer(b64decode(frame['image']), dtype=np.uint8).reshape(480, 640, 3)
    frame['depth'] = np.frombuffer(b64decode(frame['depth']), dtype=np.uint16).reshape(480, 640)
    return frame['image'], frame['depth'], frame['timestamp']


def get_obs():
    cam_high, _, _ = get_cam_frame('high')
    cam_left, _, _ = get_cam_frame('left')
    cam_right, _, _ = get_cam_frame('right')
    return concatenate_images(cam_high, cam_left, cam_right)


def worker(port, headers, data, verify):
    logger.info(f"Waiting for response from port {port}")
    # response = requests.post(f"https://172.16.209.181:{port}", headers=headers, data=json.dumps(data), verify=verify).json()
    response = requests.post(f"https://172.16.209.181:{port}", headers=headers, data=json.dumps(data), verify=verify).json()    # fy wlan port
    # response = requests.post(f"https://172.16.204.182:{port}", headers=headers, data=json.dumps(data), verify=verify).json()  # thk port
    # response = requests.post(f"https://localhost:{port}", headers=headers, data=json.dumps(data), verify=verify).json()  # hunyuan
    logger.info(f"Response from port {port} got")
    assert len(response) > 0, "password error"
    return response


def inference_vm(save_dir, ports, prompt, tts):
    logger.info(f"Get a frame for video inference...")
    obs = get_obs()
    cv2.imwrite(os.path.join(save_dir, f"image.jpg"), obs)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    headers = {
        "Content-Type": "application/json",
    }
    seeds = [1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241][:len(ports)]  # NOTE: num of ports should be less than or equal to 8
    pool = multiprocessing.Pool(len(ports))
    jobs = []
    for port, seed in zip(ports, seeds):
        data = {"prompt": prompt, "img": b64encode(cv2.imencode(".jpg", obs, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes()).decode("utf-8"), "seed": seed, "password": "r49h8fieuwK"}
        jobs.append(pool.apply_async(worker, (port, headers, data, False)))
    pool.close()
    pool.join()
    responses = []
    for job in jobs:
        response = job.get()
        responses.append(response)

    videos = {}
    sample_image = responses[0][0]
    height, width, _ = cv2.imdecode(np.frombuffer(b64decode(sample_image), np.uint8), cv2.IMREAD_COLOR).shape
    for i, port in enumerate(ports):
        videos[os.path.join(save_dir, f"{port}.mp4")] = responses[i]
    if tts:
        video_index = process_responses(prompt, responses)
        logger.info(f"TTS selected video index: {video_index}, port: {ports[video_index]}")
        videos[os.path.join(save_dir, f"tts.mp4")] = responses[video_index]
    elif len(ports) == 1:
        videos[os.path.join(save_dir, f"tts.mp4")] = responses[0]
    save_videos(videos, width, height, fps=8)


def main():
    infer_args = get_arguments()
    save_dir = os.path.join(infer_args.save_dir, infer_args.task_json.split("/")[-1].split(".json")[0], infer_args.inference_prefix)
    os.makedirs(save_dir, exist_ok=True)

    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(save_dir, "log_vm.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info(infer_args)
    tts = infer_args.tts
    if len(infer_args.ports) == 1:
        tts = False
        logger.warning("Test-time scaling is not supported for single port")

    system_prompt = "The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
    with open(infer_args.task_json, 'r') as f:
        task = json.load(f)
    prompt = system_prompt + task[infer_args.prompt_idx]['caption']
    with open(os.path.join(save_dir, "prompt.txt"), 'w') as f:
        f.write(prompt)

    if len(infer_args.eval_ports) > 0 and infer_args.save_dir == "output/grm":
        print("Starting evaluation process...")
        p = multiprocessing.Process(target=inference_vm, args=(save_dir, infer_args.eval_ports, prompt, False))
        p.start()
    inference_vm(save_dir, infer_args.ports, prompt, tts)
    if len(infer_args.eval_ports) > 0 and infer_args.save_dir == "output/grm":
        p.join()


if __name__ == '__main__':
    main()
