"""Whisper モデルの LRU キャッシュ管理。

`_load_cached_model(model_size, device)` でモデルをロードし、
`DEFAULT_MODEL_CACHE_LIMIT` 個を上限として LRU 方式でキャッシュする。
上限を超えた最古エントリは解放し CUDA メモリを確保する。
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import whisper

from constants import DEFAULT_MODEL_CACHE_LIMIT
from logging_config import get_logger

logger = get_logger(__name__)

_MODEL_CACHE_LIMIT = DEFAULT_MODEL_CACHE_LIMIT
_MODEL_CACHE: OrderedDict[tuple[str, str], any] = OrderedDict()


def _load_cached_model(model_size: str, device: str):
    """Whisper モデルを (model_size, device) キーで LRU キャッシュロード。

    - 既存キー: 末尾へ移動して再利用
    - 新規キー: ロード後に追加し、上限超過なら最古エントリを解放して VRAM を空ける
    """
    key = (model_size, device)

    # キャッシュ済み -> LRU 更新して返す
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]

    # 新規ロード
    logger.info(f"[MODEL] loading whisper '{model_size}' on {device} ...")
    m = whisper.load_model(model_size, device=device)
    _MODEL_CACHE[key] = m

    # 上限超過なら最古を解放
    if len(_MODEL_CACHE) > _MODEL_CACHE_LIMIT:
        try:
            old_key, old_model = _MODEL_CACHE.popitem(last=False)
            del old_model
            if str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(f"[MODEL] evicted cache entry: {old_key}")
        except Exception:
            pass

    return m
