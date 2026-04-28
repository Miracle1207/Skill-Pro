# utils/encode.py
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

_ENCODER_MODEL = None
_EMB_CACHE = {}

def init_text_encoder(model_name="/mnt/QTJC/qirui/LM/all-MiniLM-L6-v2", device=None):
    global _ENCODER_MODEL
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用较小但效果平衡的模型，并确保在指定设备上加载
    _ENCODER_MODEL = SentenceTransformer(model_name, device=device)

def encode_text(text, normalize=True):
    """
    改进点：
    1. 自动初始化检查
    2. 增加归一化选项 (normalize=True 使得点积等同于余弦相似度)
    3. 鲁棒性检查
    """
    global _ENCODER_MODEL, _EMB_CACHE
    if _ENCODER_MODEL is None:
        init_text_encoder()

    if text is None or (isinstance(text, str) and len(text.strip()) == 0):
        return np.zeros(_ENCODER_MODEL.get_sentence_embedding_dimension())

    # 如果是列表，递归处理或批量处理
    if isinstance(text, list):
        return _ENCODER_MODEL.encode(text, normalize_embeddings=normalize, convert_to_numpy=True)

    if text in _EMB_CACHE:
        return _EMB_CACHE[text]

    # 计算嵌入，normalize_embeddings 是关键
    emb = _ENCODER_MODEL.encode(text, normalize_embeddings=normalize, convert_to_numpy=True)
    _EMB_CACHE[text] = emb
    return emb