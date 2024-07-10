import pickle
from typing import List

import numpy as np

cache_clip = pickle.load(open("cache_clip.pkl", "rb"))


def get_embeddings(inputs: List[str]) -> np.ndarray:
    embeddings = [cache_clip[input] for input in inputs]
    return np.array(embeddings)


def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    frame_embeddings = np.load(f"ego_features_448/{video_id}.npy")
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)

    return frame_idx


if __name__ == "__main__":
    pass
