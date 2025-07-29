# -- coding: UTF-8
import numpy as np
import json
import requests
import cv2
import urllib3
from base64 import b64encode, b64decode
import os
import multiprocessing
import subprocess
import logging
import torch
import torchvision
from datetime import datetime

# from .utils.inference.process import process_image
from .utils.inference.select_video_api import process_responses
from .utils.inference.configs import *
from .idm.idm import IDM


logger = logging.getLogger(__name__)


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
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-preset', 'veryslow',
            '-crf', '10', '-threads', '1', '-pix_fmt', 'yuv420p',
            '-loglevel', 'error', k
        ]
        workers.append(multiprocessing.Process(target=save_video, args=(ffmpeg_cmd, v)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def worker(port, headers, data, verify):
    logger.info(f"Waiting for response from port {port}")
    response = requests.post(f"https://172.16.209.181:{port}", headers=headers, data=json.dumps(data), verify=verify).json()
    logger.info(f"Response from port {port} got")
    assert len(response) > 0, "password error"
    return response


def get_unique_filepath(filepath):
    """
    Generates a unique file path by appending an index if the file already exists.
    e.g. /path/to/video.mp4 -> /path/to/video_1.mp4
    """
    if not os.path.exists(filepath):
        return filepath
    
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    
    i = 1
    while True:
        new_name = f"{name}_{i}{ext}"
        new_filepath = os.path.join(directory, new_name)
        if not os.path.exists(new_filepath):
            return new_filepath
        i += 1


class Vidar:
    def __init__(self, usr_args=None):
        if usr_args is None:
            usr_args = {}
        # VM (Video Model) arguments
        self.ports = usr_args.get('ports', [23990])
        self.tts = usr_args.get('tts', False)
        self.save_dir = usr_args.get('save_dir', 'output/grm_demo')
        
        # IDM (Inverse Dynamics Model) arguments
        self.idm_model_name = usr_args.get('model_name', 'mask')
        self.idm_load_from = usr_args.get('load_from', None)

        self.obs_cache = None
        self.prompt = None
        
        self.timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        os.makedirs(self.save_dir, exist_ok=True)
        self._setup_logger()
        # Initialize the IDM model for video-to-action translation
        self._initialize_idm()

    def _setup_logger(self):
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(os.path.join(self.save_dir, "log_vidar_policy.txt"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        if not logger.hasHandlers():
            logger.addHandler(handler)
            logger.addHandler(console)

    def _initialize_idm(self):
        """Initializes the IDM model and processor."""
        logger.info("Initializing IDM model...")
        self.dinov2_processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((518, 518)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.net = IDM(model_name=self.idm_model_name, output_dim=14).cuda()
        if self.idm_load_from and os.path.isfile(self.idm_load_from):
            with torch.cuda.stream(torch.cuda.Stream()):
                loaded_dict = torch.load(self.idm_load_from, weights_only=False, map_location='cuda:0')
                self.net.load_state_dict(loaded_dict["model_state_dict"])
            logger.info(f"IDM model loaded from {self.idm_load_from}")
        else:
            raise FileNotFoundError(f"Cannot find IDM checkpoint at '{self.idm_load_from}'. Please provide a valid path via 'idm_load_from' argument.")
        self.net.eval()

    def reset(self):
        """Resets the internal state of the model."""
        self.obs_cache = None
        self.prompt = None
        logger.info("Vidar model has been reset.")

    def update_obs(self, obs):
        """Updates the model with the latest observation."""
        self.obs_cache = obs

    def set_instruction(self, instruction):
        """Sets the task instruction for the policy."""
        system_prompt = "The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
        self.prompt = system_prompt + instruction

    def set_task_name(self, task_name):
        """Sets the task name for the policy."""
        self.task_name = task_name

    def _generate_video_policy(self):
        """Generates a video policy and returns the file path."""
        if self.obs_cache is None:
            raise ValueError("Observation cache is empty. Call update_obs() first.")
        if not self.prompt:
            raise ValueError("Prompt is not set. Call set_instruction() first.")
        if not hasattr(self, 'task_name') or not self.task_name:
            raise ValueError("Task name is not set. Call set_task_name() first.")
        
        logger.info(f"Generating video policy for prompt: '{self.prompt}'")
        
        # Create a directory for the current task inside save_dir
        task_video_dir = os.path.join(self.save_dir, self.timestamp_str, self.task_name)
        os.makedirs(task_video_dir, exist_ok=True)
        
        headers = {"Content-Type": "application/json"}
        seeds = [1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241][:len(self.ports)]
        pool = multiprocessing.Pool(len(self.ports))
        jobs = []

        for port, seed in zip(self.ports, seeds):
            data = {
                "prompt": self.prompt, 
                "img": b64encode(cv2.imencode(".jpg", self.obs_cache, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes()).decode("utf-8"), 
                "seed": seed, 
                "password": "r49h8fieuwK"
            }
            jobs.append(pool.apply_async(worker, (port, headers, data, False)))
        
        pool.close()
        pool.join()

        responses = [job.get() for job in jobs]
        videos = {}
        sample_image = responses[0][0]
        height, width, _ = cv2.imdecode(np.frombuffer(b64decode(sample_image), np.uint8), cv2.IMREAD_COLOR).shape
        
        port_to_unique_path = {}
        for i, port in enumerate(self.ports):
            base_path = os.path.join(task_video_dir, f"{port}.mp4")
            unique_path = get_unique_filepath(base_path)
            videos[unique_path] = responses[i]
            port_to_unique_path[port] = unique_path

        policy_video_path = None
        if self.tts:
            video_index = process_responses(self.prompt, responses)
            logger.info(f"TTS selected video index: {video_index}, port: {self.ports[video_index]}")
            
            # Create a uniquely named "tts.mp4" file
            tts_base_path = os.path.join(task_video_dir, "tts.mp4")
            policy_video_path = get_unique_filepath(tts_base_path)
            videos[policy_video_path] = responses[video_index]
        elif len(self.ports) == 1:
            policy_video_path = port_to_unique_path[self.ports[0]]
        
        save_videos(videos, width, height, fps=8)
        
        if policy_video_path is None:
            logger.error("No policy video was selected. TTS is off and multiple ports were used. Falling back to the first port.")
            policy_video_path = port_to_unique_path[self.ports[0]]

        logger.info(f"Policy video generated at: {policy_video_path}")
        return policy_video_path

    def _video_to_actions(self, video_path):
        """Converts a video file into a sequence of actions using the IDM."""
        logger.info(f"Converting video policy '{video_path}' to actions...")
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
            for image in frames:
                image = image.permute(2, 0, 1).unsqueeze(0)
                inputs = self.dinov2_processor(image / 255)
                if 'split' in self.idm_model_name:
                    processed_image = process_image(inputs.squeeze(0).permute(1, 2, 0)).unsqueeze(1)
                else:
                    processed_image = inputs
                output = self.net(processed_image, return_mask=False)
                if isinstance(output, tuple):
                    output, _ = output
                action = output[0]
                actions.append(action)
        
        actions = torch.stack(actions, dim=0).cpu().numpy()
        logger.info(f"Generated {len(actions)} actions from video.")
        return actions

    def _modify_actions(self, actions):
        """Applies post-processing to the generated actions."""
        logger.info("Applying post-processing to generated actions...")
        for dim in [6, 13]:
            actions[:, dim] -= 0.4 * (1 - np.argsort(actions[:, dim]) / len(actions[:, dim]))
            gripper = actions[:, dim]
            mask = gripper < 0.5
            gripper[mask] = 0
            actions[:, dim] = gripper
        return actions

    def get_action(self):
        """
        Full pipeline: generates a video policy and converts it to a sequence of executable actions.
        """
        video_path = self._generate_video_policy()
        raw_actions = self._video_to_actions(video_path)
        modified_actions = self._modify_actions(raw_actions)
        return modified_actions
