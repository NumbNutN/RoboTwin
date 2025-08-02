import os
import json
import shutil
import numpy as np
import torch
import argparse
from envs.utils.parse_hdf5 import read_hdf5
from tqdm import tqdm

def process_task(task_data_path, output_dir, task_name):
    """
    Processes a single task directory: extracts action vectors, copies videos,
    and returns a list of dictionaries for the JSON index.
    """
    hdf5_dir = os.path.join(task_data_path, "data")
    video_dir = os.path.join(task_data_path, "video")

    if not (os.path.isdir(hdf5_dir) and os.path.isdir(video_dir)):
        print(f"Skipping '{task_name}' as it does not contain 'data' and 'video' subdirectories.")
        return []

    print(f"\nProcessing task: {task_name}")

    # --- Setup Output Directories ---
    output_task_dir = os.path.join(output_dir, task_name)
    os.makedirs(output_task_dir, exist_ok=True)
    output_json_path = f"{output_task_dir}.json"

    json_data = []
    
    try:
        all_files_in_dir = os.listdir(hdf5_dir)
        # First, filter out irrelevant files like 'index.html'
        hdf5_files_to_process = [f for f in all_files_in_dir if f.startswith("episode") and f.endswith(".hdf5")]
        # Now, sort the filtered list
        hdf5_files = sorted(hdf5_files_to_process, key=lambda x: int(x.replace("episode", "").replace(".hdf5", "")))
    except (ValueError, FileNotFoundError):
        print(f"  - Warning: Could not find or sort files in {hdf5_dir}. Skipping task.")
        return []

    for hdf5_filename in tqdm(hdf5_files, desc=f"  - Episodes in {task_name}"):
        if not hdf5_filename.endswith(".hdf5"):
            continue

        episode_name = hdf5_filename.split('.')[0]
        
        # --- 1. Process HDF5 to create .pt file ---
        hdf5_file_path = os.path.join(hdf5_dir, hdf5_filename)
        data_dict = read_hdf5(hdf5_file_path)
        
        if 'joint_action' in data_dict and 'vector' in data_dict['joint_action']:
            action_vector_np = data_dict['joint_action']['vector']
            action_vector_pt = torch.from_numpy(action_vector_np)
            
            output_pt_path = os.path.join(output_task_dir, f"{episode_name}_qpos.pt")
            torch.save(action_vector_pt, output_pt_path)
        else:
            print(f"  - Warning: 'joint_action' or 'vector' not found in {hdf5_filename}. Skipping .pt creation.")
            continue

        # --- 2. Copy video file ---
        source_video_path = os.path.join(video_dir, f"{episode_name}.mp4")
        dest_video_path = os.path.join(output_task_dir, f"{episode_name}.mp4")
        if os.path.exists(source_video_path):
            shutil.copy(source_video_path, dest_video_path)
        else:
            print(f"  - Warning: Video file not found at {source_video_path}. Cannot copy.")

        # --- 3. Add entry to JSON data ---
        relative_video_path = os.path.join(task_name, f"{episode_name}.mp4")
        json_data.append({"video_path": relative_video_path})

    # --- 4. Write the JSON file ---
    if json_data:
        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"  - Created JSON index at: {output_json_path}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Process a robotics dataset.")
    parser.add_argument("src_dir", type=str, help="Source directory for the dataset (the root).")
    parser.add_argument("dst_dir", type=str, help="Destination directory for the processed data.")
    parser.add_argument("task_config", type=str, help="Task configuration subdirectory name (e.g., 'expert_demos').")
    args = parser.parse_args()

    dataset_dir = args.src_dir
    output_base_dir = os.path.join(args.dst_dir)
    
    
    if not os.path.isdir(dataset_dir):
        print(f"Error: Source dataset directory not found at {dataset_dir}")
        return
        
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Starting dataset processing for task config: '{args.task_config}'")
    print(f"Source root: {dataset_dir}")
    print(f"Outputting to: {output_base_dir}")

    all_json_data = []
    # Iterate over each high-level task folder in the dataset directory.
    for task_name in sorted(os.listdir(dataset_dir)):
        # The path to the specific task config data for this task
        task_data_path = os.path.join(dataset_dir, task_name, args.task_config)
        
        if os.path.isdir(task_data_path):
            # Pass the task_name explicitly
            task_json_data = process_task(task_data_path, output_base_dir, task_name)
            if task_json_data:
                all_json_data.extend(task_json_data)
    
    if all_json_data:
        output_json_path = os.path.join(args.dst_dir, f"{args.task_config}.json")
        with open(output_json_path, "w") as f:
            json.dump(all_json_data, f, indent=4)
        print(f"\nCreated summary JSON index at: {output_json_path}")

    print("\nDataset processing finished.")


if __name__ == "__main__":
    main() 