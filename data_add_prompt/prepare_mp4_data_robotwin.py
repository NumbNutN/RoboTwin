import cv2
import os
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import ffmpeg

import time

import re
from api import generate_caption_with_concatenated_images

from multiprocessing import Pool
import subprocess
from datetime import datetime

import shutil

PROMPT_DICT = {

    # fix joint
    "handover_block": "Using both arms, using left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad.",
    "hanging_mug": "Using both arms, using left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.",
    "lift_pot": "Using both arms, lift the pot.",
    "pick_diverse_bottles":"Using both arms, pick up one bottle with one arm, and pick up another bottle with the other arm.",
    "stack_blocks_two":"Using both arms, there are two blocks on the table, the color of the blocks is red, green. Move the blocks to the center of the table, and stack the geen block on the red block.",
    
    # left/right arm
    "grab_roller": "grab the roller on the table.",
    "handover_mic": "grasp the microphone on the table and handover it to the other arm.",
    "move_can_pot": "there is a can and a pot on the table, pick up the can and move it to beside the pot.",
    "move_stapler_pad": "move the stapler to a colored mat.",
    "open_laptop": "open the laptop.",
    "place_a2b_left":"place object A on the left of object B.",
    "turn_switch":"click the switch."

    # special
    "place_bread_basket":"If there is one bread on the table, grab the bread and put it in the basket, if there are two breads on the table, using both arms, simultaneously grab up two breads and put them in the basket.",
}


def get_prompt(task_name):
    return f'{task_name[0].lower()}{task_name[1:]}'


def read_video(file_name, width=640, height=480, resize_width=None, resize_height=None):
    process1 = (
        ffmpeg.input(file_name).output('pipe:', format='rawvideo', pix_fmt='bgr24', threads=1, v='fatal').run_async(pipe_stdout=True)
    )
    frames = []
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        image = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        if resize_width and resize_height:
            image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        frames.append(image)
    return frames


def write_video(images, save_path, width=640, height=720, fps=30):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',                       # 覆盖输出文件
        '-v', 'fatal',              # 日志级别
        '-f', 'rawvideo',           # 输入格式为原始视频
        '-vcodec', 'rawvideo',      # 输入编码器为原始视频
        '-threads', '1',            # 线程数
        '-s', f'{width}x{height}',  # 分辨率
        '-pix_fmt', 'bgr24',        # OpenCV 默认像素格式
        '-r', str(fps),             # 帧率
        '-i', '-',                  # 从标准输入读取数据
        '-c:v', 'libx264',          # 使用 H.264 编码
        '-preset', 'veryslow',      # 编码速度与质量平衡
        '-crf', '10',               # 质量参数（0-51，值越小质量越高）
        '-pix_fmt', 'yuv420p',      # 兼容性像素格式
        save_path
    ]
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        for image in images:
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception as e:
        print(f'{save_path}: {e}')


def rearrange_video_views(task_name, video_paths, dest_data_file_path, caption, fps=30, episode_idx=0):
    print(f"Rearranging {task_name} episode {episode_idx} videos to {dest_data_file_path}")
    
    video = read_video(video_paths,640,720)
    frames = []
    for frame in video:
        frames.append(frame)
    write_video(frames, dest_data_file_path, 640, 720, fps)
    
    num_images_for_api = 6
    select_every = len(frames) // num_images_for_api
    images_for_api = [cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes() for image in frames[::select_every]]
    fail=0
    while True:
        try:
            caption = generate_caption_with_concatenated_images(images_for_api, re.sub(r'^\d+_|_\d+$', '', task_name).replace('_', ' '))[0]
            break
        except Exception as e:
            import openai
            print(f"error occur: {e}")
            print(f"current openai request base url is: {openai.base_url}")
            fail += 1
            print(f"fail time {fail} in task {task_name} episode {episode_idx}")
            time.sleep(30*fail)
    
    info = {'video_path': dest_data_file_path, 'caption': caption, 'width': 640, 'height': 720, 'time': len(frames) / fps}
    # with open(os.path.join('assets/test_video_rearranged', f'{task_name}.json'), 'w') as f:
    #     json.dump(info, f, indent=4)
    print(f"{task_name} episode {episode_idx} videos to {dest_data_file_path} rearranged")
    return task_name, info


def rearrange_video_dataset(source_dataset_path, dest_dataset_path, fps=30):
    print(f"Rearrange {source_dataset_path} to {dest_dataset_path}")
    task_episode_info = defaultdict(list)

    os.makedirs(dest_dataset_path, exist_ok=True)
    
    pool = Pool(8)
    jobs = []

    # Level 1: Iterate through each task directory (e.g., 'grab_roller')
    for task_name in os.listdir(source_dataset_path):
        source_task_path = os.path.join(source_dataset_path, task_name)
        
        if not os.path.isdir(source_task_path) or task_name == 'error':
            continue

        dest_task_path = os.path.join(dest_dataset_path, task_name)
        os.makedirs(dest_task_path, exist_ok=True)

        # Get all files inside the source task directory
        all_files_in_task = os.listdir(source_task_path)
        # Filter to get only the video files, which will be our starting point
        video_files = [f for f in all_files_in_task if f.endswith('.mp4')]

        # Level 2: Iterate through the found video files
        for video_filename in video_files:
            # e.g., video_filename is 'episode0.mp4'
            base_name = video_filename.rsplit('.', 1)[0] # base_name is 'episode0'
            
            # From the video's base name, construct the expected qpos filename
            qpos_filename = f"{base_name}_qpos.pt"

            # Check if the corresponding qpos file actually exists
            if qpos_filename in all_files_in_task:
                # We found a matching pair!
                
                # Define full paths for source files
                source_video_path = os.path.join(source_task_path, video_filename)
                source_qpos_path = os.path.join(source_task_path, qpos_filename)

                # Copy the qpos file to the new destination
                shutil.copy(source_qpos_path, dest_task_path)
                
                # Prepare arguments for the processing job
                dest_data_file_path = os.path.join(dest_task_path, video_filename)
                caption = ''
                try:
                    episode_idx = int(base_name.replace('episode', ''))
                    jobs.append(pool.apply_async(rearrange_video_views, args=(task_name, source_video_path, dest_data_file_path, caption, fps, episode_idx)))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse episode index from '{base_name}'. Skipping.")
    pool.close()
    pool.join()
    num_errors = 0
    for job in jobs:
        task_name, info = job.get()
        if task_name != -1 and info != -1:
            task_episode_info[task_name].append(info)
        else:
            num_errors += 1
    print(f'total {len(jobs)}, errors {num_errors}')
    
    for k, v in task_episode_info.items():
        if os.path.exists(os.path.join(dest_dataset_path, f'{k}.json')):
            with open(os.path.join(dest_dataset_path, f'{k}.json'), 'r') as f:
                info = json.load(f)
            v=info+v
        with open(os.path.join(dest_dataset_path, f'{k}.json'), 'w') as f:
            json.dump(v, f, indent=4)


def check_file(dest_dataset_path):
    time = 0
    for task_name in os.listdir(dest_dataset_path):
        if not os.path.isdir(os.path.join(dest_dataset_path, task_name)):
            continue
        with open(os.path.join(dest_dataset_path, f'{task_name}.json'), 'r') as f:
            info = json.load(f)
        num_info = len(info)
        qpos_list = []
        for i in range(len(info)):
            qpos_list.append(info[i]['video_path'].split('/')[-1].split('.')[0]+'_qpos.pt')
            time += info[i]['time']
        # num_info=1
        # qpos_list = []
        # qpos_list.append(info['video_path'].split('/')[-1].split('.')[0]+'_qpos.pt')
        num_qpos = 0
        for qpos_file in os.listdir(os.path.join(dest_dataset_path, task_name)):
            if not qpos_file.endswith('.pt'):
                continue
            if qpos_file not in qpos_list:
                print(f'find {qpos_file} in {os.path.join(dest_dataset_path, task_name)} but not in json')
                continue
            num_qpos+=1
        if num_info != num_qpos:
            print(f'num_info {num_info} num_qpos {num_qpos} in {os.path.join(dest_dataset_path, task_name)}')
    print('total time: ', time)
    
def rearrange_task_json(dest_dataset_path):
    path_prefix = '/mnt/vepfs/base2/share/robot/video_data/test_aloha_3'
    for task_name in os.listdir(dest_dataset_path):
        if task_name == 'move_green_cube_into_green_bowl_using_right_arm':
            continue
        info_new = []
        if not os.path.isdir(os.path.join(dest_dataset_path, task_name)):
            continue
        with open(os.path.join(dest_dataset_path, f'{task_name}.json'), 'r') as f:
            info = json.load(f)
        prefix = ''
        if 'using_right_arm' in task_name:
            prefix = 'using right arm, '
        elif 'using_left_arm' in task_name:
            prefix = 'using left arm, '
        elif 'using_both_arms' in task_name:
            prefix = 'using both arms, '
        info['caption'] = prefix + info['caption'].lower()
        info['video_path'] = path_prefix + info['video_path'].split('assets/test')[-1]
        info_new.append(info)
        with open(os.path.join(dest_dataset_path, f'{task_name}.json'), 'w') as f:
            json.dump(info_new, f, indent=4)


dirs_list = [
    "/home/numbnut/repo/RobotTwin2/processed_data"
]


if __name__ == '__main__':
    # 首先清除所有代理环境变量
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'SOCKS_PROXY', 'socks_proxy']
    for var in proxy_vars:
        if var in os.environ:
            print(f"Clearing proxy environment variable: {var}")
            del os.environ[var]
    
    # 设置 OpenAI 环境变量
    os.environ['DISABLE_PROXY'] = 'true'
    os.environ['OPENAI_API_BASE'] = 'https://pro.xiaoai.plus/v1'
    os.environ['OPENAI_API_KEY'] = 'sk-zV5Are9supT6lXicA9HTRh9LVQ00L1sCPDw7oxMOz3ErsWOY'
    
    # import pathlib
    # for dir in dirs_list:
    #     path_obj = pathlib.Path(dir)
    #     entrys = [entry for entry in path_obj.iterdir() if entry.is_dir()]
    #     dest_entrys = []
    #     for entry in entrys:
    #         # 构造新的目录名
    #         new_name = f"{entry.name}-rearranged"
    #         # 获取父目录，然后连接上新的名称
    #         new_path = entry.parent / new_name
    #         dest_entrys.append(new_path)
        
    #     for entry, dest_entry in zip(entrys,dest_entrys):
    #         rearrange_video_dataset(
    #             str(entry.resolve()),str(dest_entry.resolve()),fps=30
    #         )
    #         check_file(dest_entry.resolve())
    
    for dir in dirs_list:
        dest_dir = dir + "-rearranged"
        rearrange_video_dataset(
            dir,dest_dir,fps=30
        )
        check_file(dest_dir)
    
        # check_file(dest_dir)
    # rearrange_video_dataset(
    #     '/media/user/Dataset/cobot-magic/assets/cube0426', 
    #     # '/media/user/Dataset/cobot-magic/assets/cube0418-rearranged', 
    #     r'assets/cube0426-random-rearranged',
    #     fps=30)
    # rearrange_video_dataset(
    #    r'/media/user/My Passport/IDM_dataset/cube0506', 
    # #    r'/media/user/My\ Passport/IDM_dataset/cube0418-random-rearranged',
    #    r'assets/cube0506-random-rearranged',
    #    fps=30)
    # rearrange_video_dataset(
    #   r'assets/test_video',
    # #    r'/media/user/My\ Passport/IDM_dataset/cube0418-random-rearranged',
    #    r'assets/test_video_rearranged',
    #    fps=30)
    # check_file('assets/cube0418-human-rearranged')
    # check_file('assets/cube0418-random-rearranged')
    # check_file('assets/cube0418-random-rearranged_')
    # check_file('assets/cube0418-random-rearranged2')
    # check_file('assets/cube0426-random-rearranged')
    # check_file('assets/cube0426-random-rearranged2')
    # check_file('assets/cube0430-random-rearranged')
    # check_file('assets/test_video_rearranged')
    # check_file('assets/cube0506-random-rearranged')
    # rearrange_task_json('assets/test_video_rearranged')