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
    for task_name in os.listdir(source_dataset_path):
        if task_name == 'error':
            continue
        os.makedirs(os.path.join(dest_dataset_path, task_name), exist_ok=True)
        if not os.path.isdir(os.path.join(source_dataset_path, task_name)):
            continue
        for episode_dir in os.listdir(os.path.join(source_dataset_path, task_name)):
            # if not qpos_file.endswith('.pt'):
            #     continue
            episode_idx = episode_dir.split('_')[1]
            video_paths = os.path.join(source_dataset_path, task_name,f'episode_{episode_idx}','.mp4')
            if not os.path.isfile(video_paths):
                continue
            file_name = f'episode_{episode_idx}.mp4'
            qpos_file = os.path.join(source_dataset_path, task_name,f'episode_{episode_idx}_qpos.pt')
            os.system(f'cp "{os.path.join(source_dataset_path, task_name, qpos_file)}" "{os.path.join(dest_dataset_path, task_name)}"')
            dest_data_file_path = os.path.join(dest_dataset_path, task_name, file_name)
            caption = ''
            # rearrange_video_views(task_name, video_paths, dest_data_file_path, caption, fps)
            jobs.append(pool.apply_async(rearrange_video_views, args=(task_name, video_paths, dest_data_file_path, caption, fps, episode_idx)))
            # print(f"task_name {task_name} episode_idx {episode_idx} video_paths {video_paths} dest_data_file_path {dest_data_file_path}")
        #     break
        # break
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