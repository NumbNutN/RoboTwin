from openai import OpenAI
import openai
import os
import logging
logger = logging.getLogger(__name__)

def env_init():
    os.environ['OPENAI_API_BASE'] = 'https://pro.xiaoai.plus/v1'
    os.environ['OPENAI_API_KEY'] = 'sk-zV5Are9supT6lXicA9HTRh9LVQ00L1sCPDw7oxMOz3ErsWOY'
    os.environ['DISABLE_PROXY'] = 'true'

    # 首先清除所有代理环境变量
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'SOCKS_PROXY', 'socks_proxy']
    for var in proxy_vars:
        if var in os.environ:
            print(f"Clearing proxy environment variable: {var}")
            del os.environ[var]

def get_imgs(responses, n_imgs_per_video, select_every, video_indexes):
    imgs = []
    for video_index in video_indexes:
        response = responses[video_index]
        for i in range(1, n_imgs_per_video + 1):
            imgs.append(response[i * select_every])
    return imgs


def process_responses(prompt, responses):
    n_videos = len(responses)
    if n_videos == 1:
        logger.info("TTS default to 0 due to only one video")
        return 0

    n_imgs_per_video = 5
    select_every = len(responses[0]) // (n_imgs_per_video + 1)
    video_indexes = list(range(n_videos))
    video_index = 0
    max_n_videos_per_round = 2
    while len(video_indexes) > 1:
        if len(video_indexes) <= max_n_videos_per_round:
            video_index = video_indexes[rank_videos(get_imgs(responses, n_imgs_per_video, select_every, video_indexes), len(video_indexes), prompt)]
            break
        video_indexes.append(video_indexes[rank_videos(get_imgs(responses, n_imgs_per_video, select_every, video_indexes[:max_n_videos_per_round]), max_n_videos_per_round, prompt)])
        video_indexes = video_indexes[max_n_videos_per_round:]

    # output = {"imgs": responses[video_index]}
    # for i, response in enumerate(responses):
    #     if i == video_index:
    #         output[f"{i}_selected"] = response
    #         output[f"{i}"] = response
    #     else:
    #         output[f"{i}"] = response
    return video_index


def parse_rank_response(response, n_videos, index):
    try:
        index = int(response[index]) - 1
        if index < 0 or index >= n_videos:
            return -1
        else:
            return index
    except (ValueError, IndexError):
        return -1


def rank_videos(imgs, n_videos, caption):
    openai.base_url = os.getenv('OPENAI_API_BASE')
    # openai.api_key = os.getenv("XIAOAI_API_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # client = OpenAI()
    model_name = "gpt-4o"
    model_kwargs = {
        model_name:{
            'api_key': openai.api_key,
            'base_url': openai.base_url,
        }
    }

    n_imgs_per_video = len(imgs) // n_videos
    system_prompt = (f"You are a skilled robot video ranker. Your task is to identify the index of the video with the highest quality based on the provided image clips and video caption. When evaluating the images, consider both their physical accuracy and how well they align with the video caption. Each image contains three views, and you must assess their consistency, ensuring there are no abrupt appearances or disappearances of objects or color blocks between frames. When determining the index, if there is a tie, output the smallest video index. For example: if video 1 has the highest quality, output 1; if video 2 has the highest quality, output 2; if all the videos have the same quality, output 1.")
    img_seq = "\n".join([f"- **video_{i}**: image_1, image_2, ..., image_{n_imgs_per_video}" for i in range(1, n_videos + 1)])
    video_reference = "all the videos" if n_videos > 2 else "both videos"
    user_prompt = f"""We have {n_videos} videos, each containing {n_imgs_per_video} images, for you to evaluate. The caption for {video_reference} is '{caption}'. The images are arranged in the following sequence:

    {img_seq}

    Please assess the quality of the videos and provide the index of the one with the highest quality, without any explanations.
    """

    content = []
    content.append({"type": "text", "text": user_prompt})
    for img in imgs:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    # response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=230).choices[0].message.content
    with OpenAI(**model_kwargs.get(model_name,{})) as client:
        response = client.chat.completions.create(
            model=model_name, 
            messages=messages, 
            max_tokens=230).choices[0].message.content
    for index in [0, -1, -2]:
        video_index = parse_rank_response(response, n_videos, index)
        if video_index != -1:
            return video_index
    print(f"Failed to parse response: {response}")
    return 0
