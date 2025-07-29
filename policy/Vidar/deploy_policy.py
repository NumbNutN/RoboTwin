# import packages and module here
import numpy as np
import cv2
from .inference_vm import Vidar # Import the Vidar class

def encode_obs(observation):  # Post-Process Observation
    obs = observation
    head_rgbs = np.array(obs["head_camera"]["rgb"])
    left_rgbs = np.array(obs["left_camera"]["rgb"])
    right_rgbs = np.array(obs["right_camera"]["rgb"])
        
    # This logic assumes a single frame observation, not a sequence.
    # The original logic seemed to expect a sequence (len(head_rgbs)).
    # We'll process a single frame observation dict.
    head_img = head_rgbs
    left_img = left_rgbs
    right_img = right_rgbs
    
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
    """Initializes and returns the Vidar policy model."""
    model = Vidar(usr_args=usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """
    Vidar generates a video policy, which is then passed to another module 
    (simulated here) to get executable actions.
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # Set instruction and update observation for the model
    model.set_instruction(instruction)
    model.update_obs(obs)

    # Vidar model generates a video policy (file path)
    policy_video_path = model.get_policy()
    
    # --- Action Generation Simulation ---
    # In a real scenario, `policy_video_path` would be passed to another
    # module that translates the video into a sequence of actions.
    # Here, we simulate this by returning a placeholder action sequence.
    print(f"Policy video generated at: {policy_video_path}")
    print("Simulating action generation from video...")
    
    # Placeholder: generate a dummy sequence of 20 actions
    num_actions = 20 
    # Each action is a 14-dim vector for [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    simulated_actions = np.random.rand(num_actions, 14) * 0.1 

    # --- Action Execution Loop ---
    for action in simulated_actions:
        # Execute each step of the action
        TASK_ENV.take_action(action, action_type='qpos')
        # We don't need to get observation and update the model inside the loop
        # because the policy was already generated for the whole task.
        # If your actual action generation is step-by-step, you would need to
        # get observations here.

    print("Finished executing simulated actions.")


def reset_model(model):  
    """Cleans the model cache at the beginning of every evaluation episode."""
    model.reset()
