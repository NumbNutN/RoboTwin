#!/usr/bin/python3
# -- coding: UTF-8
"""
PI0 model wrapper that connects to a remote WebSocket inference server.

Server side: uv run scripts/serve_policy.py --policy.config <config> --policy.dir <ckpt_dir> --port 8000
Client side: This module connects to the server via WebSocket for inference.
"""
import numpy as np
from openpi_client import websocket_client_policy


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
                 server_host="localhost", server_port=8000):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pi0_step = pi0_step
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None

        # Connect to remote inference server
        print(f"Connecting to inference server at {server_host}:{server_port}...")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=server_host, port=server_port
        )
        print(f"Connected! Server metadata: {self.policy.get_server_metadata()}")

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
