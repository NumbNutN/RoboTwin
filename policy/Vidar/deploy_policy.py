# import packages and module here
import numpy as np
import cv2

def encode_obs(observation):  # Post-Process Observation
    obs = observation
    head_rgbs = np.array(obs["head_camera"]["rgb"])
    left_rgbs = np.array(obs["left_camera"]["rgb"])
    right_rgbs = np.array(obs["right_camera"]["rgb"])
        
    if head_rgbs.shape == left_rgbs.shape == right_rgbs.shape and len(head_rgbs) > 0:
        combined_images = []
        num_frames = len(head_rgbs)
        
        for i in range(num_frames):
            head_img = head_rgbs[i]
            left_img = left_rgbs[i]
            right_img = right_rgbs[i]
            
            h, w, _ = head_img.shape
            
            new_h, new_w = h // 2, w // 2
            
            left_resized = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            right_resized = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            bottom_row = np.concatenate([left_resized, right_resized], axis=1)
            
            if bottom_row.shape[1] != w:
                bottom_row = cv2.resize(bottom_row, (w, new_h), interpolation=cv2.INTER_AREA)
            
            final_h = h + new_h
            combined_img = np.zeros((final_h, w, 3), dtype=head_img.dtype)
            
            combined_img[:h, :w] = head_img
            combined_img[h:, :w] = bottom_row
    return combined_img


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    Your_Model = None
    # ...
    return Your_Model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    if len(
            model.obs_cache
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
