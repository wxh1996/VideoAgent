import json
from typing import List

import numpy as np
import requests

CLIP_URL = "http://localhost:8888"


def get_embeddings(inputs: List[str], modality: str, url: str) -> np.ndarray:
    response = requests.post(url, data={modality: json.dumps(inputs)}).json()
    embeddings = response["embeddings"]
    return np.array(embeddings)


def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    frame_embeddings = np.load(
        f"/pasteur/u/yuhuiz/VideoAgent/0_extract_clip_features/ego_features_448/{video_id}.npy"
    )
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions],
        "text",
        CLIP_URL,
    )
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
    return frame_idx


if __name__ == "__main__":
    pass
