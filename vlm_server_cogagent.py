import argparse
import sys
from dataclasses import dataclass

import torch
from flask import Flask, jsonify, request
from PIL import Image

from global_vars import COGVLM_CHECKPOINT_PATH, COGVLM_CODE_PATH

sys.path.append(COGVLM_CODE_PATH)
from utils.models.cogagent_model import CogAgentModel
from sat.model.mixins import CachedAutoregressiveMixin
from utils.utils.chat import chat
from utils.utils.language import llama2_text_processor_inference, llama2_tokenizer
from utils.utils.vision import get_image_processor

app = Flask(__name__)


@dataclass
class ModelConfig:
    max_length: int = 2048
    top_p: float = 0.4
    top_k: int = 1
    temperature: float = 0.8
    english: bool = True
    chinese: bool = False
    version: str = "chat_old"
    from_pretrained: str = COGVLM_CHECKPOINT_PATH
    local_tokenizer: str = "lmsys/vicuna-7b-v1.5"
    no_prompt: bool = False
    fp16: bool = False
    bf16: bool = True
    stream_chat: bool = False


config = ModelConfig()

# Load the model
model, model_args = CogAgentModel.from_pretrained(
    config.from_pretrained,
    args=argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=1,
        model_parallel_size=1,
        mode="inference",
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device="cuda",
        **vars(config)
    ),
)
model = model.eval()

# Load the tokenizer and image processor
tokenizer = llama2_tokenizer(config.local_tokenizer, signal_type=config.version)
image_processor = get_image_processor(model_args.eva_args["image_size"][0])
cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None

model.add_mixin("auto-regressive", CachedAutoregressiveMixin())

text_processor_infer = llama2_text_processor_inference(
    tokenizer, config.max_length, model.image_length
)
print(model_args)
# import pdb; pdb.set_trace()
# cache_image=None
# test_image = '/pasteur/u/xhanwang/VideoAgent/3212132295-Scene-001-02.jpg'
# with torch.no_grad():
#     response, history, cache_image = chat(
#                     test_image,
#                     model,
#                     text_processor_infer,
#                     image_processor,
#                     query='What is the animal in the frame',
#                     history=None,
#                     cross_img_processor=cross_image_processor,
#                     image=cache_image,
#                     max_length=config.max_length,
#                     top_p=config.top_p,
#                     temperature=config.temperature,
#                     top_k=config.top_k,
#                     no_prompt=config.no_prompt,
#                     args=model_args
#                 )
    

@app.route("/", methods=["POST"])
def interact_with_cogvlm():
    # Check for image in the POST request
    if "image" not in request.form:
        return jsonify({"error": "Image not provided"}), 400

    if "text" not in request.form:
        return jsonify({"error": "Text not provided"}), 400
    # print(request.form["image"])
    # raw_image = Image.open(request.form["image"]).convert("RGB")
    cache_image = None
    # print(request.files["image"])
    # import pdb; pdb.set_trace()
    try:
        with torch.no_grad():
            response, history, cache_image = chat(
                request.form["image"],
                model,
                text_processor_infer,
                image_processor,
                query=request.form["text"],
                history=None,
                cross_img_processor=cross_image_processor,
                image=cache_image,
                max_length=config.max_length,
                top_p=config.top_p,
                temperature=config.temperature,
                top_k=config.top_k,
                no_prompt=config.no_prompt,
                args=model_args
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"input": history, "output": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(sys.argv[1]), debug=False)
