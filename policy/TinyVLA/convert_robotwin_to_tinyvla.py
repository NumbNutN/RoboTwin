"""
Convert RoboTwin HDF5 data to TinyVLA-compatible HDF5 format.

RoboTwin format:
    /joint_action/vector            (N, 14)
    /observation/{cam}/rgb          compressed JPEG bytes
    /endpose/...
    + instructions/*.json           {"seen": [...], "unseen": [...]}

TinyVLA expected format:
    /action                         (N, 14)
    /observations/qpos              (N, 14)
    /observations/qvel              (N, 14)
    /observations/images/cam_high   (N, 480, 640, 3) uint8
    /observations/images/cam_left_wrist
    /observations/images/cam_right_wrist
    /language_raw                   UTF-8 string
    attrs['compress'] = True/False

Usage:
    python convert_robotwin_to_tinyvla.py <task_name>

Example:
    python convert_robotwin_to_tinyvla.py stack_blocks_two
"""

import os
import sys
import json
import glob
import h5py
import cv2
import numpy as np
from tqdm import tqdm


# Camera name mapping: RoboTwin -> TinyVLA
CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

# Target image size (TinyVLA expects 480x640 before its own resize to 448x448)
TARGET_H, TARGET_W = 480, 640


def decode_image(raw_bytes):
    """Decode compressed image bytes to numpy array, resize to target size."""
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != TARGET_H or img.shape[1] != TARGET_W:
        img = cv2.resize(img, (TARGET_W, TARGET_H))
    return img


def convert_episode(src_path, dst_path, instruction, compress=True):
    """Convert a single episode HDF5 file."""
    with h5py.File(src_path, 'r') as src:
        episode_len = src['joint_action/vector'].shape[0]
        action = src['joint_action/vector'][()]  # (N, 14)

        # qpos: use current joint state (same as action for RoboTwin)
        # For frame t, qpos is the joint state at that frame.
        # We use joint_action/vector shifted by 1 as a proxy (state before action).
        # First frame qpos = zeros, rest = previous action.
        qpos = np.zeros_like(action)
        qpos[1:] = action[:-1]

        # qvel: finite difference approximation
        qvel = np.zeros_like(action)
        qvel[1:] = action[1:] - action[:-1]

        # Decode images
        images = {}
        for robotwin_cam, tinyvla_cam in CAMERA_MAP.items():
            cam_path = f'observation/{robotwin_cam}/rgb'
            if cam_path not in src:
                raise KeyError(f"Camera {robotwin_cam} not found in {src_path}")
            raw_imgs = src[cam_path][()]
            decoded = []
            for i in range(episode_len):
                decoded.append(decode_image(raw_imgs[i]))
            images[tinyvla_cam] = np.stack(decoded, axis=0)  # (N, H, W, 3)

    # Write TinyVLA format
    with h5py.File(dst_path, 'w') as dst:
        dst.attrs['compress'] = compress
        dst.create_dataset('action', data=action.astype(np.float32))
        dst.create_dataset('language_raw', data=instruction.encode('utf-8'))

        obs_grp = dst.create_group('observations')
        obs_grp.create_dataset('qpos', data=qpos.astype(np.float32))
        obs_grp.create_dataset('qvel', data=qvel.astype(np.float32))

        img_grp = obs_grp.create_group('images')
        for cam_name, img_array in images.items():
            if compress:
                # Store as compressed JPEG bytes (same as RoboTwin)
                compressed = []
                for i in range(len(img_array)):
                    _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img_array[i], cv2.COLOR_RGB2BGR))
                    compressed.append(encoded.tobytes())
                max_len = max(len(c) for c in compressed)
                dt = h5py.special_dtype(vlen=np.uint8)
                ds = img_grp.create_dataset(cam_name, shape=(len(compressed),), dtype=dt)
                for i, c in enumerate(compressed):
                    ds[i] = np.frombuffer(c, dtype=np.uint8)
            else:
                img_grp.create_dataset(cam_name, data=img_array, dtype=np.uint8)

    return episode_len


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_robotwin_to_tinyvla.py <task_name> [data_root]")
        print("Example: python convert_robotwin_to_tinyvla.py stack_blocks_two")
        sys.exit(1)

    task_name = sys.argv[1]
    data_root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'
    )

    task_dir = os.path.join(data_root, task_name, 'demo_clean')
    src_data_dir = os.path.join(task_dir, 'data')
    instructions_dir = os.path.join(task_dir, 'instructions')
    dst_data_dir = os.path.join(task_dir, 'tinyvla_data')

    if not os.path.isdir(src_data_dir):
        print(f"Source data dir not found: {src_data_dir}")
        sys.exit(1)

    os.makedirs(dst_data_dir, exist_ok=True)

    # Find all episodes
    src_files = sorted(glob.glob(os.path.join(src_data_dir, 'episode*.hdf5')),
                       key=lambda x: int(os.path.basename(x).replace('episode', '').replace('.hdf5', '')))

    print(f"Task: {task_name}")
    print(f"Source: {src_data_dir} ({len(src_files)} episodes)")
    print(f"Output: {dst_data_dir}")

    episode_lens = []
    for src_file in tqdm(src_files, desc="Converting"):
        basename = os.path.basename(src_file)
        ep_name = basename.replace('.hdf5', '')
        dst_file = os.path.join(dst_data_dir, basename)

        # Load instruction
        inst_file = os.path.join(instructions_dir, f'{ep_name}.json')
        if os.path.exists(inst_file):
            with open(inst_file) as f:
                inst_data = json.load(f)
            # Randomly pick one from "seen" instructions
            instructions = inst_data.get('seen', inst_data.get('unseen', []))
            instruction = instructions[0] if instructions else f"{task_name}"
        else:
            instruction = task_name.replace('_', ' ')

        ep_len = convert_episode(src_file, dst_file, instruction, compress=True)
        episode_lens.append(ep_len)

    print(f"\nDone! Converted {len(src_files)} episodes")
    print(f"Episode lengths: min={min(episode_lens)}, max={max(episode_lens)}, mean={np.mean(episode_lens):.0f}")
    print(f"\nUpdate constants.py with:")
    print(f'  "dataset_dir": [DATA_DIR + "/{task_name}/demo_clean/tinyvla_data"],')
    print(f'  "episode_len": {max(episode_lens)},')


if __name__ == '__main__':
    main()
