import cv2
import h5py, pickle
import numpy as np
import os
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)
    
    # print("Keys in data_list['observation']:", data_list["observation"].keys())

    obs = data_list["observation"]
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
            
            combined_images.append(combined_img)
        
        images_to_video(np.array(combined_images), out_path=video_path)
    else:
        images_to_video(head_rgbs, out_path=video_path)


    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)
