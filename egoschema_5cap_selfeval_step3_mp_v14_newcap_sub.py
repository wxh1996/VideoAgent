from openai import OpenAI

client = OpenAI()

import copy
import csv
import json
import logging
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import lmdb
import numpy as np
import redis
import requests

from global_vars import LLM_CACHE_FILE
from utils_clip_xiaohan import frame_retrieval_seg_ego
from utils_general import get_from_cache, save_to_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(
    "egochema_subset_5cap_selfevalCoT_step3_recap_eva448_newcap_v14_allfeat_subset_final.log"
)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))
video_frame_path = "/pasteur/u/xhanwang/VideoAgent/egoschema/val_video_q1_fps1_frames"

redis_cli = redis.Redis(host="localhost", port=6379, db=0)
redis_cli.config_set("save", "60 1")
last_save_timestamp = redis_cli.lastsave()
print("[redis] last_save_timestamp", last_save_timestamp)

llm_cache = redis_cli


def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None



def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_text_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_llm_response(
    system_prompt, prompt, json_format=True, model="gpt-4-1106-preview"
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logger.info("Cache Hit")
        logger.info(cached_value)
        return cached_value

    print("Not hit cache", key)
    input()
    
    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response, llm_cache)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"


def generate_final_answer(question, caption, num_frames):
    # formatted_description = [{"1": "The boy is sitting on the floor in front of a Christmas tree, and he is wrapping a present with wrapping paper. He then puts a bow on the present and sits back down. The video does not provide any information about the boy reaching for or selecting a specific present."}, {"2": "The boy picks up the present and walks away, suggesting that he is likely going to open the present or play with it."}, {"4": "The boy might have moved to the couch to sit down and open the present, or he could have moved there to get a better view of the couch."}, {"6": "The context of the boy playing with the toy on the couch suggests that he is enjoying his present and engaging in imaginative play. The presence of the teddy bear on the couch further supports this idea. The boy's actions of picking up the present and playing with it indicate that he is excited and happy with his new toy."}]
    answer_format = {"final_answer": "xxx"}
    # false_answer_format = {"final answer": "-1"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response, None


def generate_description(question, caption, num_frames):
    # Send the question to GPT-4
    formatted_description = {
        "frame_descriptions": [
            {
                "segment_id": "1/2/3/4",
                "duration": "xxx - xxx",
                "description": "frame of xxx",
            },
            {
                "segment_id": "1/2/3/4",
                "duration": "xxx - xxx",
                "description": "frame of xxx",
            },
            {
                "segment_id": "1/2/3/4",
                "duration": "xxx - xxx",
                "description": "frame of xxx",
            },
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial five frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial five frames.
    To achieve this, we will:
    1. Divide the video into four segments based on the intervals between the initial five frames.
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    ```
    {formatted_description}
    ```
    """
    # You need more information about the video.
    # Please divide the video to 3 parts uniformly. Describe the visual content that can help localize one important frame to answer the question. The generated describtion should only contain short visual description. After that, to obtain more information to answer the question, please generate the coresponding sub-question or instruction for the selected frame. The generated sub-question or instruction will be fed into a vision language model along with the frame to get the desirable information.
    # Just generate one sentence of visual description and one sub-question or instruction for one video snapshot. Return the descriptions and sub-questions in JSON format, {formatted_description}.
    # """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response, None


def generate_description_step(question, caption, num_frames, segment_des, seg_id):
    # Send the question to GPT-4
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {seg_id}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    # You need more information about the video.
    # Please divide the video to 3 parts uniformly. Describe the visual content that can help localize one important frame to answer the question. The generated describtion should only contain short visual description. After that, to obtain more information to answer the question, please generate the coresponding sub-question or instruction for the selected frame. The generated sub-question or instruction will be fed into a vision language model along with the frame to get the desirable information.
    # Just generate one sentence of visual description and one sub-question or instruction for one video snapshot. Return the descriptions and sub-questions in JSON format, {formatted_description}.
    # """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response, None


def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return None, response


def ask_gpt_caption(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    # false_answer_format = {"final answer": "-1"}
    # Send the question to GPT-4
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response


def ask_gpt_caption_step(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    # false_answer_format = {"final answer": "-1"}
    # Send the question to GPT-4
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False)
    return prompt, response






def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption


def run_one_question(video_id, ann, caps, all_answers):
    count_frame = 0
    corr = 0
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    # import pdb; pdb.set_trace()
    num_frames = len(caps)
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    video_caption_new = read_caption(caps, sample_idx)
    previous_prompt, answer = ask_gpt_caption(
        formatted_question, video_caption_new, num_frames
    )
    answer_idx = parse_text_find_number(answer)
    _, confidence = self_eval(previous_prompt, answer)
    confidence = parse_text_find_confidence(confidence)
    count_frame += 5
    # import pdb; pdb.set_trace()

    if confidence < 3:
        logger.info("GPT Not Sure, Do locaization!")
        try:
            duration_des = [
                str(sample_idx[i]) + "-" + str(sample_idx[i + 1])
                for i in range(len(sample_idx) - 1)
            ]
            segment_des = {}
            for seg_id, duration in enumerate(duration_des):
                segment_des[seg_id + 1] = duration
            candiate_descriptions, _ = generate_description_step(
                formatted_question,
                video_caption_new,
                num_frames,
                segment_des,
                seg_id + 2,
            )
            # candiate_descriptions, _ = generate_description(formatted_question, video_caption_new, num_frames)
            # continue
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx, frame_lenth = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            for k, desc in enumerate(
                parsed_candiate_descriptions["frame_descriptions"]
            ):
                desc["selected frame"] = frame_idx[k]
                sample_idx.append(frame_idx[k])
            sample_idx = list(set(sample_idx))
            sample_idx = sorted(sample_idx)
            logger.info(str(sample_idx))
            video_caption_new = read_caption(
                caps, sample_idx
            )

            previous_prompt, answer = ask_gpt_caption_step(
                formatted_question, video_caption_new, num_frames
            )
            answer_idx = parse_text_find_number(answer)
            _, confidence = self_eval(previous_prompt, answer)
            confidence = parse_text_find_confidence(confidence)
            count_frame = len(sample_idx)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            answer, _ = generate_final_answer(
                formatted_question, video_caption_new, num_frames
            )
            answer_idx = parse_text_find_number(answer)
    # import pdb; pdb.set_trace()

    if confidence < 3:
        logger.info("GPT Not Sure, Do locaization!")
        try:
            duration_des = [
                str(sample_idx[i]) + "-" + str(sample_idx[i + 1])
                for i in range(len(sample_idx) - 1)
            ]
            segment_des = {}
            for seg_id, duration in enumerate(duration_des):
                segment_des[seg_id + 1] = duration
            candiate_descriptions, _ = generate_description_step(
                formatted_question,
                video_caption_new,
                num_frames,
                segment_des,
                seg_id + 2,
            )
            # continue
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx, frame_lenth = frame_retrieval_seg_ego(
                parsed_candiate_descriptions["frame_descriptions"], video_id, sample_idx
            )
            for k, desc in enumerate(
                parsed_candiate_descriptions["frame_descriptions"]
            ):
                desc["selected frame"] = frame_idx[k]
                sample_idx.append(frame_idx[k])
            sample_idx = list(set(sample_idx))
            sample_idx = sorted(sample_idx)
            logger.info(str(sample_idx))
            video_caption_new = read_caption(
                caps, sample_idx
            )
            answer, _ = generate_final_answer(
                formatted_question, video_caption_new, num_frames
            )
            answer_idx = parse_text_find_number(answer)
            count_frame = len(sample_idx)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            answer, _ = generate_final_answer(
                formatted_question, video_caption_new, num_frames
            )
            answer_idx = parse_text_find_number(answer)
    if answer_idx == -1:
        logger.info("Answer Index Not Found!")
        answer_idx = random.randint(0, 4)
    # print(answer)
    logger.info(video_id + "/" + str(answer_idx) + "/" + str(ann["truth"]))
    if int(ann["truth"]) == answer_idx:
        corr += 1.0

    all_answers[video_id] = (answer_idx, corr, count_frame)
    # logger.info('acc:' + str(corr/(idx+1)))
    # logger.info('Ave Frame: ' + str(count_frame/(idx+1)))
    # Save the answer to a JSON file
    return corr, count_frame


def main():
    # input_ann_file = '/pasteur/u/xhanwang/VideoAgent/egoschema/fullset_anno.json'
    input_ann_file = "/pasteur/u/xhanwang/VideoAgent_release/egoschema/subset_anno.json"
    all_cap_file = "/pasteur/u/xhanwang/VideoAgent_release/egoschema/lavila_subset.json"
    anns = json.load(open(input_ann_file, "r"))
    all_caps = json.load(open(all_cap_file, "r"))
    all_answers = {}

    tasks = [
        (video_id, anns[video_id], all_caps[video_id], all_answers)
        for video_id in list(anns.keys())
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(
            lambda p: run_one_question(*p), tasks
        )  # Unpack each tuple in the tasks list

    json_file_name = "egochema_subset_5cap_selfevalCoT_step3_recap_eva448_newcap_v14_allfeat_subset_final.json"
    json.dump(all_answers, open(json_file_name, "w"))


if __name__ == "__main__":
    main()
