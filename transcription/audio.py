"""音声処理ユーティリティ群 (Qt 非依存 / faster-whisper 版)。

提供関数:
 - `clean_hallucination(text, max_repeat)` : Whisper ハルシネーション簡易クレンジング
 - `_extract_audio(video_path, output_audio_path)` : 動画から 16kHz mono WAV を抽出
 - `_detect_script_lang(text)` : テキストの Unicode スクリプトから言語を推定
 - `_weight_lang_probs(all_probs, ja_weight, ru_weight)` : 言語確率に重みを適用
"""
from __future__ import annotations

import subprocess

from core.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================
# テキスト後処理
# ============================================================

def clean_hallucination(text: str, max_repeat: int = 8) -> str:
    """出力されたテキストの簡易クレンジング。

    - 同一文字が *max_repeat* を超えて連続する場合は切り詰め
    - 30 文字以上の連続ブロック(空白区切り) があれば先頭を残し警告タグ付与
    """
    if not text:
        return text
    cleaned: list[str] = []
    prev = ''
    count = 0
    for ch in text:
        if ch == prev:
            count += 1
            if count <= max_repeat:
                cleaned.append(ch)
        else:
            prev = ch
            count = 1
            cleaned.append(ch)
    out = ''.join(cleaned)
    if any(len(block) >= 30 for block in out.split()):
        return '[HALLUCINATION?] ' + out[:120]
    return out


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

def _detect_script_lang(text: str) -> str | None:
    """Unicode スクリプト分析で言語を推定する。

    - ひらがな / カタカナ / CJK が多い → 'ja'
    - キリル文字が多い → 'ru'
    - 判定不能 → None (呼び出し側が重みベース判定にフォールバック)
    """
    ja_chars = sum(
        1 for c in text
        if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'
    )
    ru_chars = sum(1 for c in text if '\u0400' <= c <= '\u04ff')

    if ja_chars > 0 and ru_chars == 0:
        return 'ja'
    if ru_chars > 0 and ja_chars == 0:
        return 'ru'
    if ja_chars > ru_chars * 2:
        return 'ja'
    if ru_chars > ja_chars * 2:
        return 'ru'
    return None   # 混在 or 判定不能


def _weight_lang_probs(
    all_probs: list[tuple[str, float]] | None,
    ja_weight: float,
    ru_weight: float,
) -> tuple[float, float]:
    """TranscriptionInfo.all_language_probs に重みを適用し JA/RU 確率 (%) を返す。

    Parameters
    ----------
    all_probs  : faster-whisper が返す [('ja', 0.8), ('ru', 0.1), ...] 形式のリスト
    ja_weight  : 日本語スコア補正係数
    ru_weight  : ロシア語スコア補正係数

    Returns
    -------
    (ja_pct, ru_pct) : 0-100 の百分率。合計 100 になるよう正規化。
    """
    if not all_probs:
        return 50.0, 50.0
    probs: dict[str, float] = dict(all_probs)
    ja_raw = probs.get('ja', 0.0) * ja_weight
    ru_raw = probs.get('ru', 0.0) * ru_weight
    total = ja_raw + ru_raw
    if total <= 0:
        return 50.0, 50.0
    return (ja_raw / total) * 100.0, (ru_raw / total) * 100.0
