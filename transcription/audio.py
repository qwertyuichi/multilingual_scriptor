"""音声処理ユーティリティ群 (Qt 非依存 / faster-whisper 版)。

提供関数:
 - `_extract_audio(video_path, output_audio_path)` : 動画から 16kHz mono WAV を抽出
 - `_weight_lang_probs(all_probs, lang1, lang2, w1, w2)` : 言語確率に重みを適用
"""
from __future__ import annotations

import subprocess

from core.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================
# 音声抽出
# ============================================================

def _extract_audio(video_path: str, output_audio_path: str) -> None:
    """動画から 16kHz mono PCM WAV を抽出。
    ffmpeg エラーは呼び出し側で例外として扱う。
    """
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        '-y', output_audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ============================================================
# 言語判定ユーティリティ
# ============================================================

def _weight_lang_probs(
    all_probs: list[tuple[str, float]] | None,
    lang1: str,
    lang2: str | None,
    w1: float,
    w2: float,
) -> tuple[float, float, bool]:
    """TranscriptionInfo.all_language_probs に重みを適用し確率 (%) と信頼度フラグを返す。

    Parameters
    ----------
    all_probs : faster-whisper が返す [('ja', 0.8), ('ru', 0.1), ...] 形式のリスト
    lang1     : 第1言語コード
    lang2     : 第2言語コード (None = 単言語モード)
    w1        : 第1言語スコア補正係数
    w2        : 第2言語スコア補正係数

    Returns
    -------
    (lang1_pct, lang2_pct, is_confident)
        lang1_pct / lang2_pct : 0-100 の百分率。合計 100 になるよう正規化。
        is_confident : 選択言語の合計確率質量が閾値以上なら True。
                       False のとき選択外言語である可能性が高い。
    """
    if not all_probs:
        return 50.0, 50.0, False
    probs: dict[str, float] = dict(all_probs)
    lang1_raw = probs.get(lang1, 0.0) * w1
    lang2_raw = (probs.get(lang2, 0.0) * w2) if lang2 else 0.0
    total = lang1_raw + lang2_raw
    if total <= 0:
        return 50.0, 50.0, False
    # 生確率の合計が低い = 選択言語がほぼ検出されていない → 信頼度低
    raw_total = probs.get(lang1, 0.0) + (probs.get(lang2, 0.0) if lang2 else 0.0)
    is_confident = raw_total >= 0.15
    if lang2 is None:
        return 100.0, 0.0, is_confident
    return (lang1_raw / total) * 100.0, (lang2_raw / total) * 100.0, is_confident


# スクリプトベースの補強機能は廃止されました。関連のユーティリティは削除されています。
