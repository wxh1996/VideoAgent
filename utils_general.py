import logging
import pickle
from typing import Optional

cache_llm = pickle.load(open("cache_llm.pkl", "rb"))


def save_to_cache(key: str, value: str):
    return None


def get_from_cache(key: str) -> Optional[str]:
    try:
        return cache_llm[key.encode()].decode()
    except Exception as e:
        logging.warning(f"Error getting from cache: {e}")
    return None
