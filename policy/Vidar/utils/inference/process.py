import numpy as np
import cv2

def interpolate_action(args, prev_action, cur_action): # 插值
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def encode_and_decode_image(image):
    _, image_bytes = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return image


def get_image(observation):
    """concatenate images and resize to 640x720
    """
    if len(observation['images']) == 0:
        return None
    image_front = observation['images']['cam_front']
    image_left_wrist = observation['images']['cam_left_wrist']
    image_right_wrist = observation['images']['cam_right_wrist']
    image_high = observation['images']['cam_high']
    # encode and decode images
    image_front, image_left_wrist, image_right_wrist, image_high = map(encode_and_decode_image, [image_front, image_left_wrist, image_right_wrist, image_high])
    # image = np.concatenate([cv2.resize(image_high, (320, 240)), cv2.resize(image_left_wrist, (320, 240)), cv2.resize(image_right_wrist, (320, 240))], axis=0)
    image = np.concatenate([image_high, np.concatenate([cv2.resize(image_left_wrist, (320, 240)), cv2.resize(image_right_wrist, (320, 240))], axis=1)], axis=0)
    return image


def concatenate_images(image_high, image_left_wrist, image_right_wrist):
    image_high, image_left_wrist, image_right_wrist = map(encode_and_decode_image, [image_high, image_left_wrist, image_right_wrist])
    image = np.concatenate([image_high, np.concatenate([cv2.resize(image_left_wrist, (320, 240)), cv2.resize(image_right_wrist, (320, 240))], axis=1)], axis=0)
    return image