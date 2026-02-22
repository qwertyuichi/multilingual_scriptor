"""Whisper モデルの LRU キャッシュ管理 (faster-whisper 版)。

WhisperModel はロードコストが高いため、サイズデバイスの組み合わせを
LRU で保持して再利用する。
"""
from __future__ import annotations

from collections import OrderedDict
import gc
import time

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

    logger.info(f"[MODEL_CACHE] loading model size={model_size} device={device} compute_type={resolved_ct}")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=resolved_ct,
    )
    _MODEL_CACHE[key] = model
    logger.info(f"[MODEL_CACHE] loaded: {key}")
    return model


def clear_cache() -> None:
    """Unload and clear all cached models.

    This removes references to WhisperModel instances from the LRU cache
    so their destructors run in the main thread after any worker threads
    have been stopped. Call this during application shutdown after
    ensuring no transcription threads are active.
    """
    keys = list(_MODEL_CACHE.keys())
    if not keys:
        logger.debug("[MODEL_CACHE] clear: cache empty")
        return
    logger.info(f"[MODEL_CACHE] clearing {len(keys)} cached model(s)")

    # Pop and explicitly delete model instances, then force GC to ensure
    # underlying C++ destructors run on the main thread. Add a small sleep
    # between deletions to give native libraries time to release device
    # resources (helps avoid races in ctranslate2 / device allocators).
    for k in keys:
        try:
            model = _MODEL_CACHE.pop(k, None)
            if model is not None:
                try:
                    del model
                except Exception:
                    pass
            logger.debug(f"[MODEL_CACHE] cleared: {k}")
            # let destructor run
            gc.collect()
            time.sleep(0.05)
        except Exception as e:
            logger.exception(f"[MODEL_CACHE] failed to clear {k}: {e}")

    # Final GC pass
    gc.collect()
