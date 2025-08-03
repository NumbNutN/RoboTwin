import os
import sys
# Add project root to Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import json
import shutil
import numpy as np
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import re
import time
import ffmpeg
import random

# Assuming envs.utils.parse_hdf5 and api are available in the python path
from envs.utils.parse_hdf5 import read_hdf5
from api import generate_caption_with_concatenated_images


# --- Prompts for Caption Generation (from prepare_data4vidar.py) ---

PROMPT_DICT = {
    # bimanual tasks
    "grab_roller": "grab the roller on the table.",
    "handover_block": "using left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad.",
    "hanging_mug": "using left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.",
    "lift_pot": "lift the pot.",
    "pick_diverse_bottles":"pick up one bottle with one arm, and pick up another bottle with the other arm.",
    "blocks_ranking_rgb":"place the red block, green block, and blue block in the order of red, green, and blue from left to right, placing in a row.",
    "blocks_ranking_size":"move the blocks to the center of the table, and arrange them from largest to smallest, from left to right.",
    "pick_dual_bottles":"pick up two bottles with one arm, and pick up another bottle with the other arm.",
    "place_bread_skillet":"if there is one bread on the table, grab the bread and put it in the skillet, if there are two breads on the table, simultaneously grab up two breads and put them in the skillet.",
    "place_burger_fries":"pick the hamburg and frenchfries and put them onto the tray.",
    "place_can_basket":"Use one arm to pick up the can and another arm place it in the basket.",
    "place_cans_plasticbox":"pick and place cans into plasticbox.",
    "place_dual_shoes":"pick up the two shoes on the table and put them in the shoebox, with the shoe tip pointing to the left.",
    "place_object_basket":"one arm to grab the target object and put it in the basket, then use the other arm to grab the basket, and finally move the basket slightly away.",
    "put_bottles_dustbin":"grab the bottles and put them into the dustbin to the left of the table.",
    "put_object_cabinet":"use one arm to open the cabinet's drawer, and use another arm to put the object on the table to the drawer.",
    "scan_object":"use one arm to pick the scanner and use the other arm to pick the object, and use the scanner to scan the object.",
    "stack_bowls_two":"stack the two bowls on top of each other.",
    "stack_bowls_three":"stack the three bowls on top of each other.",
    "stack_blocks_two":"move the red blocks to the center of the table, and using another arm to stack the geen block on the red block.",
    "stack_blocks_three":"move the blocks to the center of the table, and stack the blue block on the green block, and the green block on the red block"
}

SINGLE_ARM_DICT = {
    # unimanual tasks
    "handover_mic": "grasp the microphone on the table and handover it to the other arm.",
    "move_can_pot": "there is a can and a pot on the table, pick up the can and move it to beside the pot.",
    "move_stapler_pad": "move the stapler to a colored mat.",
    "open_laptop": "open the laptop.",
    "place_a2b_left":"place object A on the left of object B.",
    "turn_switch":"click the switch.",
    "adjust_bottle":"pick up the bottle on the table headup.",
    "beat_block_hammer":"take the yellow and black hammer grip and strike the block.",
    "click_alarmclock":"click the alarm clock's center of the top side button on the table.",
    "click_bell":"click the bell's top center on the table.",
    "dump_bin_bigbin":"grab the small bin and pour the balls into the big bin.",
    "move_playingcard_away":"pick up the playing card and move it away from the table. For example, if the playing card is on the outward side of the table, you should move it further outward side of the table.",
    "open_microwave":"open the microwave.",
    "place_a2b_right":"place object A on the right of object B.",
    "place_container_plate":"place the container onto the plate.",
    "place_empty_cup":"place the empty cup on the coaster.",
    "place_fan":"grab the fan and place it on a colored mat, and make sure the fan is facing the robot.",
    "place_mouse_pad":"grab the mouse and place it on a colored mat.",
    "place_object_scale":"grab the object and put it on the scale.",
    "place_object_stand":"place the object on the stand.",
    "place_phone_stand":"pick up the phone and put it on the phone stand.",
    "place_shoe":"grab the shoe from the table and place it on the mat.",
    "press_stapler":"press the stapler.",
    "rotate_qrcode":"catch the qrcode board on the table, pick it up and rotate to let the qrcode face towards the robot.",
    "shake_bottle_horizontally":"shake the bottle horizontally.",
    "shake_bottle":"shake the bottle.",
    "stamp_seal":"grab the stamp and stamp onto the specific color mat.",
    "move_pillbottle_pad":"pick the pillbottle and place it onto the pad."
}


def generate_caption(task_name, qpos_path, episode_idx, caption_source, source_instruction_path):
    """Generates a caption for a given task and episode based on the specified source."""

    if caption_source == 'api':
        # TODO: This part requires cv2 to read frames for the API.
        # This is a placeholder as we removed the direct cv2 dependency for simplicity.
        # To enable this, video reading logic would be needed here.
        print("API captioning is selected but video frame extraction is not implemented in this version. Skipping.")
        # Example of how it would work if cv2 was used:
        # frames = cv2.imread(...)
        # num_images_for_api = 6
        # select_every = len(frames) // num_images_for_api if len(frames) > num_images_for_api else 1
        # images_for_api = [cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes() for image in frames[::select_every]]
        # caption = generate_caption_with_concatenated_images(images_for_api, re.sub(r'^\d+_|_\d+$', '', task_name).replace('_', ' '))[0]
        return "Caption from API (not implemented)"

    if caption_source == 'instruction_json':
        if not source_instruction_path or not os.path.exists(source_instruction_path):
            print(f"  - Warning: Instruction JSON not found at {source_instruction_path}. Cannot generate caption.")
            return ""
        try:
            with open(source_instruction_path, 'r') as f:
                data = json.load(f)
            seen_captions = data.get("seen")
            if seen_captions and isinstance(seen_captions, list) and len(seen_captions) > 0:
                caption = random.choice(seen_captions)
                return caption
            else:
                print(f"  - Warning: 'seen' key is missing, empty, or not a list in {source_instruction_path}.")
                return ""
        except Exception as e:
            print(f"  - Warning: Could not read or parse instruction JSON {source_instruction_path}: {e}")
            return ""

    if caption_source == 'prompt_dict':
        # --- Generate caption from local dictionaries ---
        if task_name in PROMPT_DICT:
            return "using both arms, " + PROMPT_DICT[task_name]
        
        if task_name in SINGLE_ARM_DICT:
            base_caption = SINGLE_ARM_DICT[task_name]
            try:
                qpos_data = torch.load(qpos_path, map_location='cpu')
                data_subset = qpos_data[:30]  # Analyze first 30 frames
                left_std = data_subset[:, :7].std(dim=0).sum().item()
                right_std = data_subset[:, 7:].std(dim=0).sum().item()
                prefix = "using left arm, " if left_std > right_std else "using right arm, "
                return prefix + base_caption
            except Exception as e:
                print(f"  - Warning: Could not process qpos for {task_name} ep {episode_idx} to determine arm. {e}")
                return base_caption  # Fallback to base caption

        # Special handling for specific tasks
        if task_name == "place_bread_basket":
            base_caption  = PROMPT_DICT.get(task_name)
            try:
                qpos_data = torch.load(qpos_path, map_location='cpu')
                data_subset = qpos_data[:30]
                left_std = data_subset[:, :7].std(dim=0).sum().item()
                right_std = data_subset[:, 7:].std(dim=0).sum().item()
                MOVEMENT_THRESHOLD = 0.1
                is_dual_arm = (left_std > MOVEMENT_THRESHOLD and right_std > MOVEMENT_THRESHOLD)
                if is_dual_arm:
                    return "using both arms, simultaneously grab up two breads and put them in the basket."
                else:
                    prefix = "using left arm, " if left_std > right_std else "using right arm, "
                    return prefix + "grab the bread and put it in the basket."
            except Exception as e:
                print(f"  - Warning: Qpos processing failed for {task_name} ep {episode_idx}. {e}")
                return base_caption # Fallback

    print(f"  - Warning: No caption rule found for task '{task_name}'.")
    return None # Return None instead of raising an error


def update_captions_for_task(output_dir, task_name, caption_source, task_data_path):
    """
    Updates only the captions in an existing task's JSON file.
    """
    output_task_dir = os.path.join(output_dir, task_name)
    output_json_path = os.path.join(output_dir, f"{task_name}.json")

    if not os.path.exists(output_json_path):
        print(f"Skipping '{task_name}': --caption-only specified but no existing JSON found at {output_json_path}.")
        return

    print(f"\nUpdating captions only for task: {task_name}")
    try:
        with open(output_json_path, 'r') as f:
            task_json_data = json.load(f)
    except Exception as e:
        print(f"  - Error reading JSON file {output_json_path}: {e}")
        return
    
    updated_json_data = []
    for item in tqdm(task_json_data, desc=f"  - Episodes in {task_name}"):
        video_path_in_json = item.get('video_path')
        if not video_path_in_json:
            updated_json_data.append(item) # Keep item as is if no path
            continue

        base_name = os.path.basename(video_path_in_json).rsplit('.', 1)[0] # e.g., "episode_123"
        try:
            episode_idx = int(base_name.replace("episode_", ""))
            source_episode_name = f"episode{episode_idx}" # Original source file name doesn't have underscore
        except ValueError:
            print(f"  - Warning: Could not parse episode index from '{base_name}'. Skipping caption update for this entry.")
            updated_json_data.append(item)
            continue
        
        qpos_path = os.path.join(output_task_dir, f"{base_name}_qpos.pt")
        source_instruction_path = os.path.join(task_data_path, "instructions", f"{source_episode_name}.json")

        new_caption = generate_caption(task_name, qpos_path, episode_idx, caption_source, source_instruction_path)
        
        if new_caption is not None:
            item['caption'] = new_caption
        else:
            print(f"  - Caption generation failed for {base_name}, keeping old caption.")
        
        updated_json_data.append(item)

    with open(output_json_path, "w") as f:
        json.dump(updated_json_data, f, indent=4)
    print(f"  - Updated JSON captions at: {output_json_path}")


def process_task(task_data_path, output_dir, task_name, caption_source):
    """
    Processes a single task: extracts actions, copies videos, generates captions,
    and creates a JSON index.
    """
    hdf5_dir = os.path.join(task_data_path, "data")
    video_dir = os.path.join(task_data_path, "video")

    if not (os.path.isdir(hdf5_dir) and os.path.isdir(video_dir)):
        print(f"Skipping '{task_name}': does not contain 'data' and 'video' subdirectories.")
        return

    print(f"\nProcessing task: {task_name}")

    output_task_dir = os.path.join(output_dir, task_name)
    os.makedirs(output_task_dir, exist_ok=True)
    
    task_json_data = []

    try:
        hdf5_files_to_process = [f for f in os.listdir(hdf5_dir) if f.startswith("episode") and f.endswith(".hdf5")]
        hdf5_files = sorted(hdf5_files_to_process, key=lambda x: int(x.replace("episode", "").replace(".hdf5", "")))
    except (ValueError, FileNotFoundError):
        print(f"  - Warning: Could not find or sort files in {hdf5_dir}. Skipping task.")
        return

    for hdf5_filename in tqdm(hdf5_files, desc=f"  - Episodes in {task_name}"):
        episode_name = hdf5_filename.split('.')[0]
        episode_idx = int(episode_name.replace("episode", ""))

        # 1. Unify episode naming scheme to episode_{idx}
        output_episode_base_name = f"episode_{episode_idx}"

        # 2. Process HDF5 to create .pt file
        hdf5_file_path = os.path.join(hdf5_dir, hdf5_filename)
        output_pt_path = os.path.join(output_task_dir, f"{output_episode_base_name}_qpos.pt")
        try:
            data_dict = read_hdf5(hdf5_file_path)
            if 'joint_action' in data_dict and 'vector' in data_dict['joint_action']:
                action_vector_np = data_dict['joint_action']['vector']
                action_vector_pt = torch.from_numpy(action_vector_np)
                torch.save(action_vector_pt, output_pt_path)
            else:
                print(f"  - Warning: 'joint_action' or 'vector' not in {hdf5_filename}. Skipping episode.")
                continue
        except Exception as e:
            print(f"  - Error processing HDF5 file {hdf5_filename}: {e}")
            continue

        # 3. Copy video file
        source_video_path = os.path.join(video_dir, f"{episode_name}.mp4")
        dest_video_path = os.path.join(output_task_dir, f"{output_episode_base_name}.mp4")
        if os.path.exists(source_video_path):
            shutil.copy(source_video_path, dest_video_path)
        else:
            print(f"  - Warning: Video file not found at {source_video_path}. Cannot copy.")
            continue # If video is missing, no point in continuing for this episode

        # 4. Generate caption
        source_instruction_path = os.path.join(task_data_path, "instructions", f"{episode_name}.json")
        caption = generate_caption(task_name, output_pt_path, episode_idx, caption_source, source_instruction_path)

        # 5. Get video metadata and add entry to JSON
        try:
            probe = ffmpeg.probe(dest_video_path)
            video_info = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if video_info:
                duration = float(video_info.get('duration', -1))
                width = int(video_info.get('width', -1))
                height = int(video_info.get('height', -1))
            else:
                duration, width, height = -1, -1, -1
        except Exception as e:
            print(f"  - Warning: Could not probe video file {dest_video_path}: {e}")
            duration, width, height = -1, -1, -1

        relative_video_path = os.path.join(task_name, f"{output_episode_base_name}.mp4")
        task_json_data.append({
            "video_path": relative_video_path,
            "caption": caption,
            "width": width,
            "height": height,
            "time": duration
        })

    # 6. Write the JSON file for the entire task
    if task_json_data:
        output_json_path = os.path.join(output_dir, f"{task_name}.json")
        with open(output_json_path, "w") as f:
            json.dump(task_json_data, f, indent=4)
        print(f"  - Created JSON index at: {output_json_path}")


def check_integrity(dest_dataset_path):
    """
    Checks the integrity of the processed dataset by comparing JSON entries with files.
    """
    print("\n--- Running Integrity Check ---")
    total_time_seconds = 0
    for json_filename in sorted(os.listdir(dest_dataset_path)):
        if not json_filename.endswith('.json'):
            continue
        
        task_name = json_filename.rsplit('.', 1)[0]
        task_dir = os.path.join(dest_dataset_path, task_name)
        json_path = os.path.join(dest_dataset_path, json_filename)

        if not os.path.isdir(task_dir):
            print(f"Warning: Missing task directory for {json_filename}: {task_dir}")
            continue

        with open(json_path, 'r') as f:
            info_list = json.load(f)
        
        num_info = len(info_list)
        
        qpos_in_json = {
            info['video_path'].split('/')[-1].replace('.mp4', '_qpos.pt')
            for info in info_list
        }
        
        actual_qpos_files = {f for f in os.listdir(task_dir) if f.endswith('_qpos.pt')}
        
        if len(qpos_in_json) != len(actual_qpos_files):
            print(f"Mismatch in {task_name}: JSON has {len(qpos_in_json)} entries, but directory has {len(actual_qpos_files)} .pt files.")
        
        missing_files = qpos_in_json - actual_qpos_files
        if missing_files:
            print(f"  - Missing .pt files in {task_name} dir: {missing_files}")
            
        extra_files = actual_qpos_files - qpos_in_json
        if extra_files:
             print(f"  - Extra .pt files in {task_name} dir not in JSON: {extra_files}")

    print("--- Integrity Check Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Process and prepare a robotics dataset with captions.")
    parser.add_argument("src_dir", type=str, help="Source directory for the dataset (the root).")
    parser.add_argument("dst_dir", type=str, help="Destination directory for the processed data.")
    parser.add_argument("task_config", type=str, help="Task configuration subdirectory name (e.g., 'expert_demos').")
    parser.add_argument(
        "--caption-source",
        type=str,
        default="prompt_dict",
        choices=["prompt_dict", "instruction_json", "api"],
        help="Source for generating captions: 'prompt_dict' (local templates), 'instruction_json' (from source files), or 'api'."
    )
    parser.add_argument("--check-integrity", action="store_true", help="Run an integrity check after processing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed data and re-process all tasks.")
    parser.add_argument("--caption-only", action="store_true", help="Only update captions in existing JSON files, do not process files.")
    args = parser.parse_args()

    if args.caption_source == 'api':
        if 'OPENAI_API_BASE' not in os.environ or 'OPENAI_API_KEY' not in os.environ:
            print("Warning: --caption-source='api' is set, but OPENAI_API_BASE or OPENAI_API_KEY environment variables are not found.")
    
    if args.caption_only:
        print("--- Running in caption-only mode ---")
        if not os.path.isdir(args.src_dir):
             print(f"Error: Source directory '{args.src_dir}' not found, which may be required for caption generation.")
             return
        if not os.path.isdir(args.dst_dir):
             print(f"Error: Destination directory '{args.dst_dir}' not found. Nothing to update.")
             return

        for filename in sorted(os.listdir(args.dst_dir)):
            if not filename.endswith('.json'):
                continue
            
            task_name = filename.rsplit('.', 1)[0]
            task_data_path = os.path.join(args.src_dir, task_name, args.task_config)

            if not os.path.isdir(os.path.join(args.dst_dir, task_name)):
                print(f"Warning: Corresponding task data folder '{task_name}' not found in destination. Skipping.")
                continue
            if not os.path.isdir(task_data_path) and args.caption_source == 'instruction_json':
                print(f"Warning: Corresponding source data folder not found at '{task_data_path}'. Skipping caption update for '{task_name}'.")
                continue

            update_captions_for_task(args.dst_dir, task_name, args.caption_source, task_data_path)
    else:
        # Full processing mode
        dataset_dir = args.src_dir
        output_base_dir = args.dst_dir
        
        if not os.path.isdir(dataset_dir):
            print(f"Error: Source dataset directory not found at {dataset_dir}")
            return
            
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Starting dataset processing for task config: '{args.task_config}'")
        print(f"Source root: {dataset_dir}")
        print(f"Outputting to: {output_base_dir}")

        # Iterate over each high-level task folder in the dataset directory.
        task_names = sorted(os.listdir(dataset_dir))
        for task_name in task_names:
            task_data_path = os.path.join(dataset_dir, task_name, args.task_config)
            
            # Skip already processed tasks unless --overwrite is specified
            output_json_path = os.path.join(output_base_dir, f"{task_name}.json")
            if not args.overwrite and os.path.exists(output_json_path):
                print(f"Skipping task '{task_name}': Already processed. Use --overwrite to re-process.")
                continue
            
            if os.path.isdir(task_data_path):
                process_task(task_data_path, output_base_dir, task_name, args.caption_source)

        if args.check_integrity:
            check_integrity(output_base_dir)
    
    print("\nDataset processing finished.")


if __name__ == "__main__":
    main() 