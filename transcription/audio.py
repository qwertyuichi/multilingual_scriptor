"""音声処理ユーティリティ群 (Qt 非依存)。

提供関数:
 - `clean_hallucination(text, max_repeat)` : Whisper ハルシネーション簡易クレンジング
 - `_extract_audio(video_path, output_audio_path)` : 動画から 16kHz mono WAV を抽出
 - `_build_hybrid_segments(model, audio_path, min_seg_dur)` : JA/RU 二重走査でセグメント境界を統合
 - `_has_voice(segment, sample_rate, vad_level, return_analysis)` : VAD 音声判定
 - `_detect_lang_probs(model, audio_segment, ja_weight, ru_weight)` : JA/RU 言語確率推定
 - `_transcribe_clip(model, audio_segment, language)` : 1 クリップを Whisper で文字起こし
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import numpy as np
import torch
import whisper
import webrtcvad
from scipy.io import wavfile

from logging_config import get_logger

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
# セグメント境界検出
# ============================================================

def _build_hybrid_segments(model, audio_path: str, min_seg_dur: float = 0.25) -> list[dict]:
    """JA/RU それぞれを一度走らせて start/end 点を統合し最小長フィルタ。

    Whisper 自体のセグメント境界を粗結合する簡易ハイブリッド方式。
    """
    logger.debug("[ADV] Hybrid segmentation ...")
    common_kw = dict(verbose=False, condition_on_previous_text=False,
                     word_timestamps=False, task='transcribe')
    ja_res = model.transcribe(audio_path, language='ja', **common_kw)
    ru_res = model.transcribe(audio_path, language='ru', **common_kw)
    points: set[float] = set()
    for seg in ja_res.get('segments', []):
        points.update({round(float(seg['start']), 2), round(float(seg['end']), 2)})
    for seg in ru_res.get('segments', []):
        points.update({round(float(seg['start']), 2), round(float(seg['end']), 2)})
    pts = sorted(p for p in points if p >= 0)
    merged: list[dict] = []
    for a, b in zip(pts, pts[1:]):
        if (b - a) >= min_seg_dur:
            merged.append({'start': a, 'end': b})
    if not merged:
        return [{'start': s['start'], 'end': s['end']} for s in ja_res.get('segments', [])]
    return merged


# ============================================================
# VAD / 音声判定
# ============================================================

def _has_voice(
    segment: np.ndarray,
    sample_rate: int = 16000,
    vad_level: int = 2,
    return_analysis: bool = False,
):
    """簡易 VAD + 音声フレーム比率の計算。

    return_analysis=False -> bool のみ返す (後方互換)
    return_analysis=True  -> (has_voice: bool, voiced_ratio: float)
    """
    if len(segment) < sample_rate * 0.2:
        return (True, 1.0) if return_analysis else True
    vad = webrtcvad.Vad(vad_level)
    frame_dur_ms = 30
    frame_size = int(sample_rate * frame_dur_ms / 1000)
    pcm16 = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    total = 0
    voiced = 0
    for offset in range(0, len(pcm16), frame_size * 2):
        frame = pcm16[offset: offset + frame_size * 2]
        if len(frame) < frame_size * 2:
            break
        total += 1
        try:
            if vad.is_speech(frame, sample_rate):
                voiced += 1
        except Exception:
            pass
    if total == 0:
        ratio = 0.0
        res = False
    else:
        ratio = voiced / total
        res = voiced > 0
    if return_analysis:
        return res, ratio
    return res


# ============================================================
# 言語確率推定
# ============================================================

def _detect_lang_probs(
    model,
    audio_segment: np.ndarray | torch.Tensor,
    ja_weight: float = 1.0,
    ru_weight: float = 1.0,
) -> tuple[str, float, float]:
    """JA / RU 言語確率 (簡易) を取得し重み補正後に百分率で返す。

    - 例外時はデフォルト 50/50 フォールバック
    - EN 等は無視し 2 クラス正規化

    Returns
    -------
    (lang, ja_prob_pct, ru_prob_pct)
    """
    sr = 16000
    if isinstance(audio_segment, torch.Tensor):
        audio_segment = audio_segment.detach().cpu().numpy()
    audio_segment = np.asarray(audio_segment).flatten().astype(np.float32)

    # 長さ補正 (pad/trim)
    min_len = sr * 2
    max_len = sr * 30
    if audio_segment.size < min_len:
        audio_segment = whisper.pad_or_trim(audio_segment, min_len)
    elif audio_segment.size > max_len:
        audio_segment = whisper.pad_or_trim(audio_segment, max_len)

    # モデルのメル次元数を取得
    try:
        expected_mels = getattr(getattr(model, 'dims', object()), 'n_mels', None)
        if not isinstance(expected_mels, int):
            expected_mels = model.encoder.conv1.weight.shape[1]
        if not isinstance(expected_mels, int):
            expected_mels = 80
    except Exception:
        expected_mels = 80

    def build_mel(arr: np.ndarray):
        try:
            return whisper.log_mel_spectrogram(arr, n_mels=expected_mels).to(model.device)
        except TypeError:
            return whisper.log_mel_spectrogram(arr).to(model.device)
        except AssertionError:
            arr = np.asarray(arr).flatten().astype(np.float32)
            return whisper.log_mel_spectrogram(arr, n_mels=expected_mels).to(model.device)

    mel = build_mel(audio_segment)

    # 言語検出
    try:
        with torch.no_grad():
            _, probs = model.detect_language(mel)
        logger.debug(f"[LANG_PROB] probs={probs}")
    except Exception as e:
        try:
            padded = whisper.pad_or_trim(audio_segment, sr * 30)
            mel2 = build_mel(padded)
            with torch.no_grad():
                _, probs = model.detect_language(mel2)
        except Exception:
            logger.warning(f"[LANG_PROB][EXCEPTION] {e}. fallback probs={{'ja':0.5,'ru':0.5}}")
            probs = {'ja': 0.5, 'ru': 0.5}

    ja_raw = float(probs.get('ja', 0.0))
    ru_raw = float(probs.get('ru', 0.0))
    ja_adj = ja_raw * ja_weight
    ru_adj = ru_raw * ru_weight
    denom = ja_adj + ru_adj
    if denom <= 0:
        return 'ja', 50.0, 50.0
    ja_prob = ja_adj / denom * 100.0
    ru_prob = ru_adj / denom * 100.0
    lang = 'ja' if ja_prob >= ru_prob else 'ru'
    return lang, ja_prob, ru_prob


# ============================================================
# クリップ文字起こし
# ============================================================

def _transcribe_clip(model, audio_segment: np.ndarray, language: str) -> dict:
    """1 クリップを一時 WAV に書き出し Whisper へ渡して文字起こし。

    numpy 配列を直接渡すインターフェースが安定していないため WAV 経由で処理する。
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        wavfile.write(tmp_path, 16000, (audio_segment * 32767).astype(np.int16))
        return model.transcribe(
            tmp_path,
            language=language,
            temperature=0.2,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            verbose=False,
            condition_on_previous_text=False,
            task='transcribe'
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
