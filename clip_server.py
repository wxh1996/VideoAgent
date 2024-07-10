import json
import logging
import sys
from typing import List

sys.path.append("/pasteur/u/xhanwang/VideoAgent/eva/EVA/EVA-CLIP-18B/shinji")
# import open_clip
import eva_clip as open_clip
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from tqdm import trange

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

CLIP_MODEL = "EVA-CLIP-8B-plus"
CLIP_DATASET = "eva_clip"
BATCH_SIZE = 32
DEVICE = "cuda"

(model, _, preprocess,) = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_DATASET, force_custom_clip=True
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)


def get_image_embeddings(image_paths: List[str]) -> List[List[float]]:
    for i in trange(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i : i + BATCH_SIZE]
        images = torch.stack(
            [preprocess(Image.open(img).convert("RGB")) for img in batch]
        ).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            image_features = image_features.cpu().numpy()
            if i == 0:
                embeddings = image_features
            else:
                embeddings = np.concatenate((embeddings, image_features))
    return embeddings.tolist()


def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    for i in trange(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        text = tokenizer(batch).to(DEVICE)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            text_features = text_features.cpu().numpy()
            if i == 0:
                embeddings = text_features
            else:
                embeddings = np.concatenate((embeddings, text_features))
    return embeddings.tolist()


@app.route("/", methods=["POST"])
def interact_with_clip():
    logging.info(request.form)
    if "image" in request.form:
        images = json.loads(request.form["image"])
        logging.info(images)
        embeddings = get_image_embeddings(images)

    if "text" in request.form:
        texts = json.loads(request.form["text"])
        logging.info(texts)
        embeddings = get_text_embeddings(texts)

    return jsonify({"embeddings": embeddings})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=int(sys.argv[1]), debug=False)
