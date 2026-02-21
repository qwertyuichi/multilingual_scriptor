"""Whisper モデルの LRU キャッシュ管理 (faster-whisper 版)。

WhisperModel はロードコストが高いため、サイズデバイスの組み合わせを
LRU で保持して再利用する。
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from faster_whisper import WhisperModel

from core.constants import DEFAULT_MODEL_CACHE_LIMIT
from core.logging_config import get_logger

logger = get_logger(__name__)

# LRU キャッシュ: key = (model_size, device, compute_type)  WhisperModel
_MODEL_CACHE: OrderedDict[tuple, WhisperModel] = OrderedDict()


def _resolve_compute_type(device: str, compute_type: str = "default") -> str:
    """デバイスに応じた compute_type を自動解決する。"""
    if compute_type != "default":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _load_cached_model(
    model_size: str,
    device: str,
    compute_type: str = "default",
) -> WhisperModel:
    """LRU キャッシュから WhisperModel を取得。未キャッシュ時はロードしてキャッシュ。

    Parameters
    ----------
    model_size   : "large-v3" / "turbo" など
    device       : "cuda" / "cpu" / "auto"
    compute_type : "float16" / "int8" / "float32" / "default"
                   "default" の場合 CUDAfloat16、CPUint8 を自動選択。
    """
    resolved_ct = _resolve_compute_type(device, compute_type)
    key = (model_size, device, resolved_ct)

    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        logger.debug(f"[MODEL_CACHE] hit: {key}")
        return _MODEL_CACHE[key]

    # キャッシュ上限を超えたら最古エントリを削除
    while len(_MODEL_CACHE) >= DEFAULT_MODEL_CACHE_LIMIT:
        evict_key, _ = _MODEL_CACHE.popitem(last=False)
        logger.info(f"[MODEL_CACHE] evict: {evict_key}")
        if evict_key[1] == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    logger.info(f"[MODEL_CACHE] loading model size={model_size} device={device} compute_type={resolved_ct}")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=resolved_ct,
    )
    _MODEL_CACHE[key] = model
    logger.info(f"[MODEL_CACHE] loaded: {key}")
    return model
