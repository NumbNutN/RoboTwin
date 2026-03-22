"""
Convert RoboTwin collected data to LeRobot format for OpenPi training.

Usage:
    cd policy/pi05
    uv run scripts/convert_robotwin_to_lerobot.py \
        --task_name stack_blocks_two \
        --setting demo_clean \
        --expert_data_num 200

The resulting dataset will be saved to $LEROBOT_HOME/<repo_name>.
Use the repo_name as --data.repo-id when training.
"""

import argparse
import json
import os
import shutil

import cv2
import h5py
import numpy as np
try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME, LeRobotDataset


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def main():
    parser = argparse.ArgumentParser(description="Convert RoboTwin data to LeRobot format")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--expert_data_num", type=int, required=True)
    parser.add_argument("--repo_name", type=str, default=None,
                        help="Output repo name. Defaults to <task_name>-<setting>-<num>")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 480],
                        help="Width Height for resized images")
    args = parser.parse_args()

    if args.repo_name is None:
        args.repo_name = f"{args.task_name}-{args.setting}-{args.expert_data_num}"

    data_dir = os.path.join("../../data", args.task_name, args.setting)
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        exit(1)

    # Clean up any existing dataset
    output_path = LEROBOT_HOME / args.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Determine state/action dimensions from first episode
    sample_path = os.path.join(data_dir, "data", "episode0.hdf5")
    left_gripper, left_arm, right_gripper, right_arm, _ = load_hdf5(sample_path)
    state_dim = left_arm.shape[1] + 1 + right_arm.shape[1] + 1  # arm + gripper for each side
    img_w, img_h = args.image_size

    print(f"State dim: {state_dim}, Image size: {img_w}x{img_h}")
    print(f"Output: {output_path}")

    # Create LeRobot dataset
    # Feature keys must match what the training config's repack_transforms expect:
    #   "observation.images.cam_high"        -> images.cam_high
    #   "observation.images.cam_left_wrist"  -> images.cam_left_wrist
    #   "observation.images.cam_right_wrist" -> images.cam_right_wrist
    #   "observation.state"                  -> state
    #   "action"                             -> actions
    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type="aloha",
        fps=args.fps,
        features={
            "observation.images.cam_high": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for i in range(args.expert_data_num):
        hdf5_path = os.path.join(data_dir, "data", f"episode{i}.hdf5")
        instr_path = os.path.join(data_dir, "instructions", f"episode{i}.json")

        left_gripper, left_arm, right_gripper, right_arm, image_dict = load_hdf5(hdf5_path)
        num_steps = left_gripper.shape[0]

        # Load instruction as task description
        with open(instr_path, "r") as f:
            instr_dict = json.load(f)
        task_description = instr_dict.get("seen", [instr_dict.get("unseen", [""])[0]])[0]

        # Build states: [left_arm, left_gripper, right_arm, right_gripper]
        states = []
        for j in range(num_steps):
            state = np.concatenate([
                left_arm[j], [left_gripper[j]],
                right_arm[j], [right_gripper[j]],
            ]).astype(np.float32)
            states.append(state)

        # Add frames: state[t] paired with action[t] = state[t+1]
        for j in range(num_steps - 1):
            # Decode compressed images
            cam_high_bits = image_dict["head_camera"][j]
            cam_high = cv2.imdecode(np.frombuffer(cam_high_bits, np.uint8), cv2.IMREAD_COLOR)
            cam_high = cv2.cvtColor(cv2.resize(cam_high, (img_w, img_h)), cv2.COLOR_BGR2RGB)

            cam_left_bits = image_dict["left_camera"][j]
            cam_left = cv2.imdecode(np.frombuffer(cam_left_bits, np.uint8), cv2.IMREAD_COLOR)
            cam_left = cv2.cvtColor(cv2.resize(cam_left, (img_w, img_h)), cv2.COLOR_BGR2RGB)

            cam_right_bits = image_dict["right_camera"][j]
            cam_right = cv2.imdecode(np.frombuffer(cam_right_bits, np.uint8), cv2.IMREAD_COLOR)
            cam_right = cv2.cvtColor(cv2.resize(cam_right, (img_w, img_h)), cv2.COLOR_BGR2RGB)

            dataset.add_frame({
                "observation.images.cam_high": cam_high,
                "observation.images.cam_left_wrist": cam_left,
                "observation.images.cam_right_wrist": cam_right,
                "observation.state": states[j],
                "action": states[j + 1],
                "task": task_description,
            })

        dataset.save_episode()
        print(f"Episode {i}/{args.expert_data_num} done ({num_steps - 1} frames)")

    # Note: consolidate() is not available in this version of lerobot.
    # Stats will be computed by openpi's compute_norm_stats.py.
    print(f"\nDataset saved to: {output_path}")
    print(f"Use --data.repo-id={args.repo_name} when training")


if __name__ == "__main__":
    main()
