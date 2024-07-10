import glob
import json
import logging
import os
from typing import List

import lmdb
import numpy as np
import redis
import requests
import torch
import torch.nn.functional as F

from global_vars import CLIP_CACHE_FILE, CLIP_URL
from utils_general import get_from_cache, save_to_cache

# clip_cache = lmdb.open(CLIP_CACHE_FILE, map_size=int(1e11))

redis_cli = redis.Redis(host="localhost", port=6379, db=0)
redis_cli.config_set("save", "60 1")
last_save_timestamp = redis_cli.lastsave()
print("[redis] last_save_timestamp", last_save_timestamp)

clip_cache = redis_cli


def get_embeddings(inputs: List[str], model: str, modality: str) -> np.ndarray:
    input_to_embeddings = {}
    # for inp in inputs:
    #     key = json.dumps([inp, model])
    #     cached_value = get_from_cache(key, clip_cache)
    #     if cached_value is not None:
    #         logging.debug(f"CLIP Cache Hit")
    #         input_to_embeddings[inp] = json.loads(cached_value)

    # uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]
    uncached_inputs = inputs

    if len(uncached_inputs) > 0:
        try:
            response = requests.post(
                CLIP_URL, data={modality: json.dumps(uncached_inputs)}
            ).json()
            # print(type(response["embeddings"]), len(response["embeddings"]))
            for inp, embedding in zip(uncached_inputs, response["embeddings"]):
                input_to_embeddings[inp] = embedding
                # key = json.dumps([inp, model])
                # save_to_cache(key, json.dumps(embedding), clip_cache)
        except Exception as e:
            logging.error(f"CLIP Error: {e}")
            for inp in uncached_inputs:
                input_to_embeddings[inp] = None

    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)


def frame_retrieval(text, video_id):
    video_dir = os.path.join(
        "/pasteur/u/xhanwang/VideoAgent/nextqa/val_video_q1_fps3_frames/", video_id
    )
    frames = sorted(glob.glob(video_dir + "*.jpg"))
    frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings([text], "ViT-bigG-14", "text")
    similarity = text_embedding @ frame_embeddings.T
    scores = similarity.squeeze(0).tolist()
    return scores


def frame_retrieval_all(descriptions, video_id):
    frame_embeddings = np.load(
        os.path.join(
            "/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/ego_features_448",
            video_id + ".npy",
        )
    )
    # frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "EVA-CLIP-8B-plus",
        "text",
    )
    # frames = sorted(glob.glob(video_dir + '/*.jpg'))
    # frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    # text_embedding = get_embeddings([text[str(idx)] for idx, text in enumerate(descriptions)], "ViT-bigG-14", "text")
    # import pdb; pdb.set_trace()
    similarity = text_embedding @ frame_embeddings.T
    # scores = similarity.squeeze(0).tolist()
    return similarity, len(frames)


def frame_retrieval_all_ego(descriptions, video_id, sample_idx):
    frame_embeddings = np.load(
        os.path.join(
            "/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/ego_features_448",
            video_id + ".npy",
        )
    )
    # frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "EVA-CLIP-8B-plus",
        "text",
    )
    # import pdb; pdb.set_trace()
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg_similarity = text_embedding[idx] @ frame_embeddings.T
        seg_frame_idx = seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)
    # similarity = text_embedding @ frame_embeddings.T
    # scores = similarity.squeeze(0).tolist()
    return frame_idx, frame_embeddings.shape[0]


def frame_retrieval_iter(descriptions, video_id):
    video_dir = os.path.join(
        "/pasteur/u/xhanwang/VideoAgent/nextqa/val_video_q1_fps3_frames/", video_id
    )
    frames = sorted(glob.glob(video_dir + "/*.jpg"))
    frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "ViT-bigG-14",
        "text",
    )
    # import pdb; pdb.set_trace()
    similarity = text_embedding @ frame_embeddings.T
    # scores = similarity.squeeze(0).tolist()
    return similarity, len(frames)


def frame_retrieval_seg(descriptions, video_id, sample_idx):
    video_dir = os.path.join(
        "/pasteur/u/xhanwang/VideoAgent/nextqa/val_video_q1_fps3_frames/", video_id
    )
    frames = sorted(glob.glob(video_dir + "/*.jpg"))
    frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "ViT-bigG-14",
        "text",
    )
    # import pdb; pdb.set_trace()
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[
            sample_idx[seg] + 1 : sample_idx[seg + 1]
        ]
        # import pdb; pdb.set_trace()
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)
    # scores = similarity.squeeze(0).tolist()
    return frame_idx, len(frames)


# def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
#     video_dir = os.path.join('/pasteur/u/xhanwang/VideoAgent/egoschema/val_video_q1_fps1_frames', video_id)
#     frames = sorted(glob.glob(video_dir + '/*.jpg'))
#     frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
#     text_embedding = get_embeddings([description["description"] for description in descriptions], "ViT-bigG-14", "text")
#     # import pdb; pdb.set_trace()
#     frame_idx = []
#     for idx, description in enumerate(descriptions):
#         seg = int(description["segment_id"]) - 1
#         seg_frame_embeddings=frame_embeddings[sample_idx[seg]+1:sample_idx[seg+1]]
#         # import pdb; pdb.set_trace()
#         seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
#         seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
#         frame_idx.append(seg_frame_idx)
#     # scores = similarity.squeeze(0).tolist()
#     return frame_idx, len(frames)
def frame_retrieval_seg_nqa(descriptions, video_id, sample_idx):
    # video_dir = os.path.join('/pasteur/u/xhanwang/VideoAgent/egoschema/val_video_q1_fps1_frames', video_id)
    # frames = sorted(glob.glob(video_dir + '/*.jpg'))
    # frame_embeddings = np.load(os.path.join('/pasteur/u/yuhuiz/VideoAgent/ego_features_openclip',video_id+'.npy'))
    # frame_embeddings = np.load(os.path.join('/pasteur/u/yuhuiz/VideoAgent/final/nextqa_features/',video_id+'.npy'))
    frame_embeddings = np.load(
        os.path.join(
            "/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/nextqa_features_448",
            video_id + ".npy",
        )
    )

    # frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "EVA-CLIP-8B-plus",
        "text",
    )
    # import pdb; pdb.set_trace()
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
        # import pdb; pdb.set_trace()
        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)
    # scores = similarity.squeeze(0).tolist()
    return frame_idx, frame_embeddings.shape[0]


def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    # video_dir = os.path.join('/pasteur/u/xhanwang/VideoAgent/egoschema/val_video_q1_fps1_frames', video_id)
    # frames = sorted(glob.glob(video_dir + '/*.jpg'))
    # frame_embeddings = np.load(os.path.join('/pasteur/u/yuhuiz/VideoAgent/ego_features_openclip',video_id+'.npy'))
    # frame_embeddings = np.load(os.path.join('/pasteur/u/xhanwang/VideoAgent/egoschema/egoeva_feature',video_id+'.npy'))
    # frame_embeddings = np.load(os.path.join('/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/ego_features_224',video_id+'.npy'))
    # frame_embeddings = np.load(os.path.join('/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/nextqa_features_448',video_id+'.npy'))
    frame_embeddings = np.load(
        os.path.join(
            "/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/ego_features_448",
            video_id + ".npy",
        )
    )
    # frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "EVA-CLIP-8B-plus",
        "text",
    )
    # import pdb; pdb.set_trace()
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
        # import pdb; pdb.set_trace()
        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)
    # scores = similarity.squeeze(0).tolist()
    return frame_idx, frame_embeddings.shape[0]


# def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
#     # video_dir = os.path.join('/pasteur/u/xhanwang/VideoAgent/egoschema/val_video_q1_fps1_frames', video_id)
#     # frames = sorted(glob.glob(video_dir + '/*.jpg'))
#     frame_embeddings = np.load(os.path.join('/pasteur/u/yuhuiz/VideoAgent/ego_features_openclip',video_id+'.npy'))
#     # frame_embeddings = get_embeddings(frames, "ViT-bigG-14", "image")
#     text_embedding = get_embeddings([description["description"] for description in descriptions], "ViT-bigG-14", "text")
#     # import pdb; pdb.set_trace()
#     frame_idx = []
#     for idx, description in enumerate(descriptions):
#         seg = int(description["segment_id"]) - 1
#         seg_frame_embeddings=frame_embeddings[sample_idx[seg]+1:sample_idx[seg+1]]
#         # import pdb; pdb.set_trace()
#         seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
#         seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
#         frame_idx.append(seg_frame_idx)
#     # scores = similarity.squeeze(0).tolist()
#     return frame_idx, frame_embeddings.shape[0]

if __name__ == "__main__":
    candiate_description = [
        {
            "segment_id": "1",
            "duration": "1-38",
            "description": "frame of C picking up a dog mat",
        },
        {
            "segment_id": "3",
            "duration": "45-86",
            "description": "frame of C putting the dog mat in the sink",
        },
        {
            "segment_id": "5",
            "duration": "90-133",
            "description": "frame of C washing the dog mat with soap and water",
        },
        {
            "segment_id": "6",
            "duration": "133-135",
            "description": "frame of C rinsing the dog mat",
        },
        {
            "segment_id": "8",
            "duration": "142-180",
            "description": "frame of C drying the dog mat off",
        },
    ]
    sample_idx = [1, 38, 45, 86, 90, 133, 135, 142, 180]
    frame_idx, sp = frame_retrieval_seg_ego(
        candiate_description, "3223ece4-dc21-4ca9-8e78-2af8036ec4e8", sample_idx
    )
    print(frame_idx, sp)
    # import pdb; pdb.set_trace()

    # embeddings = get_embeddings(
    #     [""],
    #     "ViT-bigG-14",
    #     "image",
    # )
    # print(embeddings)

    # embeddings = get_embeddings(["shit", "haha", "hello world"], "ViT-bigG-14", "text")
    # print(embeddings)
