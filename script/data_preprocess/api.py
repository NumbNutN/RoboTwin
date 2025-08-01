import openai
import os
import cv2
import numpy as np
from openai import OpenAI
from base64 import b64encode
import httpx

# 清除所有可能的代理环境变量
def clear_proxy_env():
    """清除所有代理相关的环境变量"""
    proxy_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
        'ALL_PROXY', 'all_proxy',
        'NO_PROXY', 'no_proxy',
        'SOCKS_PROXY', 'socks_proxy',
        'PROXY', 'proxy'
    ]
    
    for var in proxy_vars:
        if var in os.environ:
            print(f"Clearing proxy environment variable: {var}")
            del os.environ[var]

# 创建支持代理的 httpx 客户端
def create_http_client():
    """创建支持代理的 HTTP 客户端"""
    try:
        # 如果设置了 DISABLE_PROXY，清除所有代理环境变量
        if os.getenv('DISABLE_PROXY', 'false').lower() == 'true':
            print("Proxy disabled by DISABLE_PROXY environment variable")
            clear_proxy_env()
            return httpx.Client()
        
        # 检查是否有代理环境变量
        proxy_url = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY') or os.getenv('http_proxy') or os.getenv('https_proxy')
        
        if proxy_url and proxy_url.startswith('socks'):
            # 如果是 SOCKS 代理，暂时禁用以避免问题
            print(f"Warning: SOCKS proxy detected ({proxy_url}), but disabled to avoid connection issues")
            clear_proxy_env()
            return httpx.Client()
        elif proxy_url:
            # 如果是 HTTP/HTTPS 代理，使用正确的语法
            print(f"Using HTTP proxy: {proxy_url}")
            return httpx.Client(proxy=proxy_url)
        else:
            # 无代理
            return httpx.Client()
    except Exception as e:
        print(f"Warning: Failed to create proxy client: {e}. Using default client.")
        return httpx.Client()

# 创建全局的 OpenAI 客户端
def create_openai_client():
    """创建配置好的 OpenAI 客户端"""
    # 确保环境变量已设置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv('OPENAI_API_BASE')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not base_url:
        raise ValueError("OPENAI_API_BASE environment variable is not set")
    
    print(f"Creating OpenAI client with base_url: {base_url}")
    
    http_client = create_http_client()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client
    )

# 全局客户端实例
_client = None

def get_client():
    """获取或创建 OpenAI 客户端实例"""
    global _client
    if _client is None:
        _client = create_openai_client()
    return _client


def decode_image(image):
    return cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)


def decode_all_images(images):
    return [decode_image(image) for image in images]


def generate_caption(name):
    client = get_client()
    model = "gpt-4o"
    system_prompt = ("You are a skilled robot instruction annotator. Your task is to take a given raw instruction of robot operations, and then expand the original instruction by adding details to make it detailed, rich, and accurate. Your response should be no longer than one sentence and match genuine human instructions. Try to avoid using specific numbers like 'five centimeters' and avoid modifying the position like \"forward, backward, left, right\". You need to include descriptions of the actions of both robot arms. You should only respond with the expanded instruction, and the word limit is 230.")
    dataset_prompt = f"The raw instruction of the task is: {name}" 
    content = []
    content.append({"type": "text", "text": dataset_prompt})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=230)
    return response.choices[0].message.content, model


def generate_caption_with_images(images, name):
    client = get_client()
    encoded_images = [b64encode(image).decode('utf-8') for image in images]
    model_name = "gpt-4o"
    system_prompt = ("You are a skilled robot instruction annotator. Your task is to take a given raw instruction of robot operations, and then expand the original instruction by adding details to make it detailed, rich, and accurate. You can also access images to help you build your answer, and each image is from the camera in the center. Your response should be no longer than one sentence and match genuine human instructions. Try to avoid using specific numbers like 'five centimeters' and avoid modifying the position like \"forward, backward, left, right, rightmost, leftmost\". Try to avoid specifying which particular arm to use for the subtask. Try to avoid using ordinal numbers like \"first, second\". Note that a task may involve operating multiple objects, especially placement tasks. You should only respond with the expanded instruction, and the word limit is 230.")
    dataset_prompt = f"The raw instruction of the task is: {name}"
    content = []
    content.append({"type": "text", "text": dataset_prompt})
    for encoded_image in encoded_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model_name, 
        messages=messages, 
        max_tokens=230)
    return response.choices[0].message.content, model_name


def generate_caption_with_concatenated_images(images, name):
    client = get_client()
    encoded_images = [b64encode(image).decode('utf-8') for image in images]
    model_name = "gpt-4o"
    system_prompt = ("You are a skilled robot instruction annotator. Your task is to take a given raw instruction of robot operations, and then expand the original instruction by adding details to make it detailed, rich, and accurate. You can also access images to help you build your answer, and each image is the concatenation of images from three different cameras: the camera in the center at high altitude, the camera on the left wrist, and the camera on the right wrist. Your response should be no longer than one sentence and match genuine human instructions. Try to avoid using specific numbers like 'five centimeters' and avoid modifying the position like \"forward, backward, left, right, rightmost, leftmost\". Try to avoid specifying which particular arm to use for the subtask. Try to avoid using ordinal numbers like \"first, second\". You can specify objects using colors, shapes, or other attributes, if necessary. You should only respond with the expanded instruction, and the word limit is 230.") # You need to include descriptions of the actions of both robot arms. Specify which arm should perform each action, such as 'left arm' and 'right arm'. 
    dataset_prompt = f"The raw instruction of the task is: {name}"
    content = []
    content.append({"type": "text", "text": dataset_prompt})
    for encoded_image in encoded_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model_name, 
        messages=messages, 
        max_tokens=230)
    return response.choices[0].message.content, model_name


def generate_task_and_prompt(ros_operator):
    client = get_client()
    while True:
        result = ros_operator.get_frame()
        if result[0] is False:
            print(result[1])
            continue
        print("syn success")
        break
    (img_front, img_left, img_right, img_high, img_front_depth, img_left_depth, img_right_depth, img_high_depth, puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result
    image = np.concatenate([np.concatenate([img_high, img_front], axis=1), np.concatenate([img_left, img_right], axis=1)], axis=0)
    encoded_image = b64encode(image).decode('utf-8')
    model_name = "gpt-4o"
    system_prompt = (
        """
**Prompt:**  
Analyze the four sub-images (high center, low center, left wrist, and right wrist cameras) to identify visible objects and their spatial relationships. Generate a feasible task for a dual-arm robot with grippers, ensuring the task:  
1. Uses only objects present in the images.  
2. Leverages both arms for efficient execution.  
3. Avoids assumptions about objects or states not visible in the images.  

Output a task name (a concise phrase) followed by a single sentence describing the task, separated by a period. Do not include any additional text.  

**Thinking Guidance:**  
- Prioritize tasks that require coordination between both arms (e.g., lifting, transferring, or assembling).  
- Ensure the task is physically achievable given the objects' size, shape, and arrangement.  
- Avoid tasks requiring tools or objects not visible in the images.  

Example output:  
Stack red blocks. Use both arms to pick up and stack the red blocks into a stable tower.
        """
        )
    # "You are a skilled robot instruction annotator and you are very good at finding tasks that robotic arms can accomplish within a given scene. Your task is to come up with a task that a robotic arm can accomplish based on the input image, and then expand the task by adding details to make it detailed, rich, and accurate. The input image you get is the concatenation of images from four different cameras: the camera in the center at high altitude, the camera in the center at low altitude, the camera on the left wrist, and the camera on the right wrist. Your response should be a phrase no longer than 8 words and a sentence, separated by a period. The sentence should match genuine human instructions. Try to avoid using specific numbers like 'five centimeters' and avoid modifying the position like \"forward, backward, left, right, rightmost, leftmost\". Try to avoid specifying which particular arm to use for the subtask. Try to avoid using ordinal numbers like \"first, second\". You can specify objects using colors, shapes, or other attributes, if necessary. You should only respond with the phrase and the expanded instruction, and the word limit is 230."
    content = []
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    while True:
        response = client.chat.completions.create(
            model=model_name, 
            messages=messages, 
            max_tokens=230)
        response_content = response.choices[0].message.content
        parts = response_content.split('.', 1)
        task_name = parts[0].strip()
        prompt = parts[1].strip() if len(parts) > 1 else ''
        if prompt=='' or len(task_name.split()) > 8:
            continue
        return task_name.replace(' ', '_'), prompt, encoded_image

def decide_if_task_is_finished_successfully(task_name, prompt, first_frame, ros_operator):
    client = get_client()
    while True:
        result = ros_operator.get_frame()
        if result[0] is False:
            print(result[1])
            continue
        print("syn success")
        break
    (img_front, img_left, img_right, img_high, img_front_depth, img_left_depth, img_right_depth, img_high_depth, puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result
    image = np.concatenate([img_high, img_front, img_left, img_right], axis=0)
    last_frame = b64encode(image).decode('utf-8')
    model_name = "gpt-4o"
    system_prompt = (
        """
**Prompt:**  
Analyze two multi-view image sets (pre-action & post-action). Each set contains four sub-images: high center, low center, left wrist, and right wrist cameras. Compare visual changes across all perspectives using these steps:  
1. Identify key objects/positions mentioned in the instruction.  
2. Check if expected transformations (e.g., object relocation, state change) are consistently visible across relevant camera angles.  
3. Confirm absence of residual states contradicting the goal (e.g., spills remaining, tools not stored).  
Prioritize decisive evidence from the most viewpoint-appropriate camera(s). Ignore irrelevant changes like lighting variations. Output only "True" or "False" followed by a concise rationale under 15 words, separated by a period.  

Example output:  
False. Target object remains unmoved in left wrist view.  
True. All cameras show lid properly sealed.  

**Thinking Guidance:**  
- Cross-validate critical changes in at least two camera angles to avoid parallax illusions.  
- If wrist cameras show conflicting evidence with center views, default to center views unless the task specifically involves arm-mounted tool manipulation.  
- Reject assumptions about intermediate states; rely solely on pre/post visual evidence.
        """
                     ) 
    '''You are very good at identifying the differences between two images, explaining the difference and explaining what happened. Your task is to take two given images, a task name and an expanded instruction of robot operations, and then decide whether the task is finished. Each image is the concatenation of images from four different cameras: the camera in the center at high altitude, the camera in the center at low altitude, the camera on the left wrist, and the camera on the right wrist. For the given two images, one is from the beginning of the task and the other represents the current scene. Please determine whether the task has been completed based on these two images. Your response should be a 'true' or a 'false' and your chain of thought no longer than one sentence. Seperate them with a period. Try to avoid using specific numbers like 'five centimeters' and avoid modifying the position like \"forward, backward, left, right, rightmost, leftmost\". Try to avoid specifying which particular arm to use for the subtask. Try to avoid using ordinal numbers like \"first, second\". You can specify objects using colors, shapes, or other attributes, if necessary. Think twice before giving the answer. You should only respond with 'true' or 'false' and your chain of thought, and the word limit is 230.'''

    dataset_prompt = f"The name of the task is: {task_name}. The prompt of the task is: {prompt}. This is the image pre-action."
    content = []
    content.append({"type": "text", "text": dataset_prompt})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_frame}"}})
    dataset_prompt2 = f"This is the image post-action."
    content2 = []
    content2.append({"type": "text", "text": dataset_prompt2})
    content2.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_frame}"}})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}, {"role": "user", "content": content2}]
    while True:
        response = client.chat.completions.create(
            model=model_name, 
            messages=messages, 
            max_tokens=230)
        response_content = response.choices[0].message.content
        parts = response_content.split('.', 1)
        success_flag = parts[0].strip().lower()
        if success_flag in ['true', 't', 'yes', 'ok', 'y']:
            success_flag = True
        elif success_flag in ['false', 'f', 'no', 'n']:
            success_flag = False
        else:
            continue
        chain_of_thought = parts[1].strip() if len(parts) > 1 else ''
        if chain_of_thought=='' or len(task_name.split()) > 8:
            continue
        return success_flag, chain_of_thought

def define_task_class(task_name, prompt):
    client = get_client()
    model_name = "gpt-4o"
    task_classes = ['articulated manipulations', 'coordination manipulations', 'basic manipulations', ' object interactions', 'precision manipulations', 'scene understanding']
    system_prompt = (f"You are very good at categorizing robot tasks. Remember all tasks are categorized into six types: 1) Articulated Manipulations: Involve opening, closing, turning on, or turning off objects with articulated joints. 2) Coordination Manipulations: Require coordination between robot arms. 3) Basic Manipulations: Include basic skills such as grasping, holding, and placing. 4) Object Interactions: Involve interacting with multiple objects, for example, pushing one cube across another. 5) Precision Manipulations: Necessary when objects are difficult to grasp or target areas are limited, such as pouring liquid into a cup or inserting a battery. 6) Scene Understanding: Major challenges related to understanding the scene, like closing the upper drawer from the right side or placing four large blocks of different colors into corresponding colored boxes. Your task is to take the task name and prompt, and exclusively categorize this task into one of the six categories mentioned above. Think twice before giving the answer. You should only respond with a phrase chosen in {task_classes} and your chain of thought, seperated by a period, and the word limit is 230.")
    dataset_prompt = f"The name of the task is: {task_name}. The prompt of the task is: {prompt}"
    content = []
    content.append({"type": "text", "text": dataset_prompt})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    while True:
        response = client.chat.completions.create(
            model=model_name, 
            messages=messages, 
            max_tokens=230)
        response_content = response.choices[0].message.content
        parts = response_content.split('.', 1)
        task_class = parts[0].lower()
        if task_class in task_classes:
            chain_of_thought = parts[1].strip() if len(parts) > 1 else ''
            if chain_of_thought=='' or len(task_name.split()) > 8:
                continue
            return task_class, chain_of_thought
        else:
            print(f"task class {task_class} is not in task classes {task_classes}")