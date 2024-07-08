import hashlib
from typing import Dict, List, Optional

import lmdb
from PIL import Image
import redis
import logging


def resize_image(image: Image.Image, size=(256, 256)) -> Image.Image:
    return image.resize(size)


def merge_images_horizontally(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = [resize_image(image) for image in images]
    total_width = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    height = imgs[0].height

    merged = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in imgs:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    return merged


def merge_images_vertically(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = images
    total_height = sum(img.height for img in imgs) + gap * (len(imgs) - 1)
    width = max(img.width for img in imgs)

    merged = Image.new("RGB", (width, total_height))

    y_offset = 0
    for img in imgs:
        merged.paste(img, (0, y_offset))
        y_offset += img.height + gap

    return merged


def save_data_diff_image(dataset1: List[Dict], dataset2: List[Dict], save_path: str):
    # Load images into memory as PIL Image objects
    images_dataset1 = [Image.open(item["path"]) for item in dataset1]
    images_dataset2 = [Image.open(item["path"]) for item in dataset2]

    # Merge images from the same dataset horizontally
    merged_images_dataset1 = merge_images_horizontally(images_dataset1)
    merged_images_dataset2 = merge_images_horizontally(images_dataset2)

    # Merge the resulting images from different datasets vertically
    final_merged_image = merge_images_vertically(
        [merged_images_dataset1, merged_images_dataset2]
    )

    # Save the merged image
    final_merged_image.save(save_path)


# def hash_key(key) -> str:
#     return hashlib.sha256(key.encode()).hexdigest()


def get_from_cache(key: str, redis_env: redis.Redis) -> Optional[str]:
    try:
        # with env.begin(write=False) as txn:
        #     hashed_key = hash_key(key)
        #     value = txn.get(hashed_key.encode())
        value = redis_env.get(key.encode())
        if value:
            # logging.warning(f"Cache Hit: {key} -> {value.decode()}")
            return value.decode()
    except Exception as e:
        logging.warning(f"Error getting from cache: {e}")
    return None


def save_to_cache(key: str, value: str, redis_env: redis.Redis):
    try:
        # with env.begin(write=True) as txn:
        #     hashed_key = hash_key(key)
        #     txn.put(hashed_key.encode(), value.encode())
        redis_env.set(key.encode(), value.encode())
        # logging.warning(f"Cache Saved: {key} -> {value}")
    except Exception as e:
        logging.warning(f"Error saving to cache: {e}")
    return None
