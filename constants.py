"""集中定数定義。

GUI/再文字起こしパイプラインで散在していたマジック文字列・数値をここに集約し、
将来の調整時に漏れを防ぐ目的。
"""
from __future__ import annotations

# プレースホルダ表示用テキスト
PLACEHOLDER_PENDING = "[再解析中]"

# ウォッチドッグタイムアウト (ms)
DEFAULT_WATCHDOG_TIMEOUT_MS = 15000

# 分割時の最小セグメント長 (秒)
MIN_SEGMENT_DUR = 0.2

# 部分再文字起こしの最大許容長 (秒)
MAX_RANGE_SEC = 30.0

# モデルキャッシュ上限 (LRU)
DEFAULT_MODEL_CACHE_LIMIT = 2

__all__ = [
    "PLACEHOLDER_PENDING",
    "DEFAULT_WATCHDOG_TIMEOUT_MS",
    "MIN_SEGMENT_DUR",
    "MAX_RANGE_SEC",
    "DEFAULT_MODEL_CACHE_LIMIT",
]
