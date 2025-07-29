import threading
import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class SharedConfig:
    add_caption_aloha = """The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following manipulation task: """
    task_config = {'camera_names': ['cam_front', 'cam_left_wrist', 'cam_right_wrist', 'cam_high']}
    train_mean = torch.tensor([-0.26866713, 0.83559588, 0.69520934, -0.29099351, 0.18849116, -0.01014598,
                            1.41953145, 0.35073715, 1.05651613, 0.8930193, -0.37493264, -0.18510782,
                            -0.0272574, 1.35274259]).cuda()
    train_std = torch.tensor([0.25945241, 0.65903812, 0.52147207, 0.42150272, 0.32029947, 0.28452226,
                            1.78270006, 0.29091741, 0.67675932, 0.58250554, 0.42399049, 0.28697442,
                            0.31100304, 1.67651926]).cuda()


inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

collect_data_thread = None
collect_data_lock = threading.Lock()
collect_data_actions = None
collect_data_timestep = None

dinov2_processor = None

# exit_flag = False