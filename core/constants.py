"""集中定数定義。

GUI/再文字起こしパイプラインで散在していたマジック文字列・数値をここに集約し、
将来の調整時に漏れを防ぐ目的。
"""
from __future__ import annotations

# 利用可能な Whisper 公式モデル名一覧 (UI コンボボックス用)
WHISPER_MODELS: list[str] = [
    "large-v3",
    "distil-large-v3",
]

# プレースホルダ表示用テキスト
PLACEHOLDER_PENDING = "[再解析中]"

# ウォッチドッグタイムアウト (ms)
DEFAULT_WATCHDOG_TIMEOUT_MS = 15000

# 分割時/手動再分割時の最小セグメント長 (秒)
# 初回フル文字起こしの min_seg_dur (config.toml 内) とは別管理。
# 0.2 だと極端に短い区間が量産され重複テキストが発生しやすいため 0.50 に引き上げ。
MIN_SEGMENT_DUR = 0.50

# 重複テキスト自動マージ用閾値 (デバッグ/品質向上)
# 直前セグメントと JA/RU 両テキストが完全一致し、区間長がこの値未満ならマージ候補
DUP_MERGE_MAX_SEG_DUR = 1.20  # 秒
# 直前セグメント終端と今回セグメント開始のギャップ許容 (通常 0 か極小)
DUP_MERGE_MAX_GAP = 0.30      # 秒

# 部分再文字起こしの最大許容長 (秒)
MAX_RANGE_SEC = 30.0

# モデルキャッシュ上限 (LRU)
DEFAULT_MODEL_CACHE_LIMIT = 2

__all__ = [
    "WHISPER_MODELS",
    "PLACEHOLDER_PENDING",
    "DEFAULT_WATCHDOG_TIMEOUT_MS",
    "MIN_SEGMENT_DUR",
    "DUP_MERGE_MAX_SEG_DUR",
    "DUP_MERGE_MAX_GAP",
    "MAX_RANGE_SEC",
    "DEFAULT_MODEL_CACHE_LIMIT",
]
