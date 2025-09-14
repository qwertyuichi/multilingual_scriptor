"""Transcription processing module.

リファクタリング方針:
 - 機能は現状(高度処理のみ)を維持
 - 冗長ロジックの関数分割
 - 型ヒント / ドキュメント整備
 - コメントは意味が重複するものを簡潔化し、役割が曖昧な箇所は明確化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import logging
import os
import tempfile
import subprocess
from datetime import timedelta
import traceback

import numpy as np
import torch
import whisper
import webrtcvad
from scipy.io import wavfile
from PySide6.QtCore import QThread, Signal

# ============================================================
# ロギング設定
# ============================================================

logger = logging.getLogger(__name__)
# ここではハンドラを追加せず、アプリ起動側(main.py)の root ロガー設定を利用する。
# (以前はここで StreamHandler を追加していたため二重出力になっていた)

# ============================================================
# 文字列 / 後処理ユーティリティ
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

# =============== 高度処理用ユーティリティ (reference/transcription_for_kapra.py から抽出/簡略化) ===============

def _extract_audio(video_path: str, output_audio_path: str) -> None:
    """動画から 16kHz mono PCM wav を抽出。
    ffmpeg エラーは呼び出し側で例外として扱う。"""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        '-y', output_audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

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

def _has_voice(segment: np.ndarray, sample_rate: int = 16000, vad_level: int = 2) -> bool:
    """簡易 VAD。一定長未満は肯定。WebRTC VAD で 30ms フレーム判定。"""
    if len(segment) < sample_rate * 0.2:
        return True
    vad = webrtcvad.Vad(vad_level)
    frame_dur_ms = 30
    frame_size = int(sample_rate * frame_dur_ms / 1000)
    pcm16 = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    for offset in range(0, len(pcm16), frame_size * 2):
        frame = pcm16[offset: offset + frame_size * 2]
        if len(frame) < frame_size * 2:
            break
        if vad.is_speech(frame, sample_rate):
            return True
    return False

def _detect_lang_probs(model, audio_segment: np.ndarray | torch.Tensor,
                       ja_weight: float = 1.0, ru_weight: float = 1.0) -> tuple[str, float, float]:
    """JA / RU 言語確率 (簡易) を取得し重み補正後に百分率で返す。

    例外時はデフォルト 50/50 フォールバック。EN 等は無視し 2 クラス正規化。
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
    # メル次元推定
    try:
        expected_mels = getattr(getattr(model, 'dims', object()), 'n_mels', None)
        if not isinstance(expected_mels, int):
            expected_mels = model.encoder.conv1.weight.shape[1]
        if not isinstance(expected_mels, int):
            expected_mels = 80
    except Exception:
        expected_mels = 80
    # スペクトログラム計算 (互換フォールバック付き)
    def build_mel(arr: np.ndarray):
        try:
            return whisper.log_mel_spectrogram(arr, n_mels=expected_mels).to(model.device)
        except TypeError:
            return whisper.log_mel_spectrogram(arr).to(model.device)
        except AssertionError:
            arr = np.asarray(arr).flatten().astype(np.float32)
            return whisper.log_mel_spectrogram(arr, n_mels=expected_mels).to(model.device)
    mel = build_mel(audio_segment)
    if mel.shape[0] not in (expected_mels, 80):  # 再試行(稀ケース)
        try:
            mel = build_mel(audio_segment)
        except Exception:
            pass
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

def _transcribe_clip(model, audio_segment: np.ndarray, language: str):
    """1 クリップを一時 WAV に書き出し whisper へ渡して文字起こし。

    直接 numpy から与えるインターフェースが安定していないため WAV 経由。
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

def advanced_process_video(
    video_path: str,
    model_size: str = 'large-v3',
    segmentation_model_size: str | None = 'turbo',
    seg_mode: str = 'hybrid',
    device: str | None = None,
    ja_weight: float = 0.80,
    ru_weight: float = 1.25,
    min_seg_dur: float = 0.60,
    ambiguous_threshold: float = 10.0,
    vad_level: int = 2,
    gap_threshold: float = 0.5,
    output_format: str = 'txt',
    srt_max_line: int = 50,
    include_silent: bool = False,
    debug: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """動画を高度ルールで処理し GUI 互換の結果 dict を返す。
    戻り値: {'text': str, 'segments': [{'start','end','text','id'}], 'language': 'mixed'}
    """
    # ログレベル切替 (debug引数) - root ロガーを調整して全体反映
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info('[ADV] loading models ...')  # モデルロードログ
    # デバイス決定 (None の場合は自動)
    selected_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if selected_device == 'cuda' and not torch.cuda.is_available():
        logger.warning('[DEVICE] cuda 選択されたが利用不可のため cpu へフォールバックします')
        selected_device = 'cpu'
    logger.info(f'[DEVICE] using device={selected_device}')
    load_kw = {'device': selected_device}
    if status_callback:
        status_callback('モデル読み込み中...')
    if segmentation_model_size:
        seg_model = whisper.load_model(segmentation_model_size, **load_kw)
        model = whisper.load_model(model_size, **load_kw)
    else:
        model = whisper.load_model(model_size, **load_kw)
        seg_model = model
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
        audio_path = tmp_aud.name
    try:
        if status_callback:
            status_callback('音声抽出中...')
        _extract_audio(video_path, audio_path)
        full_audio = whisper.load_audio(audio_path)
        if status_callback:
            status_callback('セグメント解析中...')
        if seg_mode == 'hybrid':  # JA/RU 二重走査で境界統合
            segs = _build_hybrid_segments(seg_model, audio_path, min_seg_dur=min_seg_dur)
            initial_segments = [{'start': s['start'], 'end': s['end']} for s in segs]
        else:  # Whisper デフォルト分割
            res = seg_model.transcribe(
                audio_path, language=None, verbose=False,
                word_timestamps=False, condition_on_previous_text=False,
                task='transcribe'
            )
            initial_segments = [
                {'start': s['start'], 'end': s['end']} for s in res.get('segments', [])
            ]
        output_lines = []
        gui_segments = []
        srt_entries = []
        jsonl_entries = []
        prev_end = 0.0
        idx = 0
        total_segments = len(initial_segments)
        for seg in initial_segments:
            if cancel_flag and cancel_flag():
                if status_callback:
                    status_callback('キャンセル中...')
                break
            # 進捗を10%→90%で線形更新
            if progress_callback is not None and initial_segments:
                prog = 10 + int(80 * (idx + 1) / len(initial_segments))
                progress_callback(prog)
            if status_callback and total_segments:
                # 過剰な描画を避けるため 50 ステップ or 各1件レベルで更新
                step = max(1, total_segments // 50)
                if (idx % step) == 0:
                    status_callback(f'文字起こし中... ({idx+1}/{total_segments})')
            st = seg['start']; ed = seg['end']  # 秒
            gap = st - prev_end
            if include_silent and gap >= gap_threshold and prev_end>0:
                # テキスト出力用 GAP 行
                output_lines.append(f"[GAP {gap:.2f}s]")
                # GUI 表示用 GAP 疑似セグメント (音声無し区間)
                gui_segments.append({
                    'start': prev_end,
                    'end': st,
                    'text': '[GAP]',
                    'text_ja': '',
                    'text_ru': '',
                    'chosen_language': 'gap',
                    'id': f"gap-{idx}",  # 音声セグメントとは別扱い
                    'ja_prob': 0.0,
                    'ru_prob': 0.0,
                    'gap': True,
                })
            prev_end = ed
            start_sample = max(0, int(st * 16000))
            end_sample = min(len(full_audio), int(ed * 16000))
            if end_sample <= start_sample:
                continue
            clip = full_audio[start_sample:end_sample].astype(np.float32)
            if len(clip) < 16000 * 0.1:  # 100ms 未満はスキップ
                if include_silent:
                    output_lines.append(f"[SKIP short {st:.2f}-{ed:.2f}]")
                continue
            if not _has_voice(clip, vad_level=vad_level):
                if include_silent:
                    output_lines.append(f"[SKIP silence {st:.2f}-{ed:.2f}]")
                continue
            # クリップshapeチェック・値確認（リサンプリングは行わずpad/trimmingのみ）
            if not isinstance(clip, np.ndarray) or clip.ndim != 1 or clip.size == 0:
                logger.debug(f"[SKIP] invalid clip shape: {clip.shape}")
                continue
            # 強制デュアル: 常に JA/RU 両方文字起こしし、_detect_lang_probs の確率を基準に主表示を選択
            detected_lang, ja_prob, ru_prob = _detect_lang_probs(
                model, clip, ja_weight, ru_weight
            )
            ja_res = _transcribe_clip(model, clip, 'ja')
            ru_res = _transcribe_clip(model, clip, 'ru')
            ja_text_raw = ja_res.get('text', '').strip()
            ru_text_raw = ru_res.get('text', '').strip()
            ja_text = clean_hallucination(ja_text_raw)
            ru_text = clean_hallucination(ru_text_raw)
            # 確率の高い方を main text とする
            if ja_prob >= ru_prob:
                seg_text = ja_text
                chosen_lang = 'ja'
            else:
                seg_text = ru_text
                chosen_lang = 'ru'
            if not (ja_text or ru_text):
                continue
            def fmt_ts(t: float) -> str:
                td = timedelta(seconds=t)
                total_seconds = td.total_seconds()
                h = int(total_seconds // 3600)
                m = int((total_seconds % 3600) // 60)
                s = total_seconds % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}"
            ts_start, ts_end = fmt_ts(st), fmt_ts(ed)
            line = f"[{ts_start} -> {ts_end}] [JA:{ja_prob:05.2f}%] [RU:{ru_prob:05.2f}%] JA={ja_text} | RU={ru_text}".strip()
            output_lines.append(line)
            gui_segments.append({
                'start': st,
                'end': ed,
                'text': seg_text,
                'text_ja': ja_text,
                'text_ru': ru_text,
                'chosen_language': chosen_lang,
                'id': idx,
                'ja_prob': ja_prob,
                'ru_prob': ru_prob
            })
            # SRT/JSONL用
            srt_entries.append((idx+1, st, ed, seg_text))
            amb = abs(ja_prob - ru_prob) < ambiguous_threshold
            jsonl_entries.append({
                'index': idx+1,
                'start': st,
                'end': ed,
                'text': seg_text,
                'ja_prob': ja_prob,
                'ru_prob': ru_prob,
                'ambiguous': amb,
                'language': detected_lang
            })
            idx += 1
        if status_callback:
            status_callback('出力整形中...')
        full_text = '\n'.join(output_lines)
        # SRT/JSONL出力対応
        if cancel_flag and cancel_flag():
            if status_callback:
                status_callback('キャンセル完了 (部分結果)')
        if output_format == 'srt':
            def to_srt_timestamp(sec: float) -> str:
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                ms = int((sec - int(sec)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            blocks: list[str] = []
            for idx0, st, ed, txt in srt_entries:
                if len(txt) > srt_max_line:
                    lines = [txt[i:i + srt_max_line] for i in range(0, len(txt), srt_max_line)]
                    txt_fmt = '\n'.join(lines)
                else:
                    txt_fmt = txt
                blocks.append(f"{idx0}\n{to_srt_timestamp(st)} --> {to_srt_timestamp(ed)}\n{txt_fmt}\n")
            return {
                'text': '\n'.join(blocks),
                'segments': gui_segments,
                'language': 'mixed'
            }
        if output_format == 'jsonl':
            import json
            return {
                'text': '\n'.join(json.dumps(e, ensure_ascii=False) for e in jsonl_entries),
                'segments': gui_segments,
                'language': 'mixed'
            }
        return {'text': full_text, 'segments': gui_segments, 'language': 'mixed'}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ====================================================================================================

class TranscriptionThread(QThread):
    """GUI から起動されるバックグラウンド文字起こしスレッド。"""

    progress = Signal(int)
    status = Signal(str)
    finished_transcription = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: str, options: dict):  # options は GUI で構築
        super().__init__()
        self.video_path = video_path
        self.options = options
        self._cancel_requested = False

    def request_cancel(self):
        """GUI からキャンセル要求。"""
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:  # QThread 既定 run override
        try:
            self.status.emit("モデルを読み込み中...")
            self.progress.emit(10)
            result = advanced_process_video(
                self.video_path,
                model_size=self.options.get('model', 'large-v3'),
                segmentation_model_size=self.options.get('segmentation_model_size', 'turbo'),
                seg_mode=self.options.get('seg_mode', 'hybrid'),
                device=self.options.get('device'),
                ja_weight=self.options.get('ja_weight', 0.80),
                ru_weight=self.options.get('ru_weight', 1.25),
                min_seg_dur=self.options.get('min_seg_dur', 0.60),
                ambiguous_threshold=self.options.get('ambiguous_threshold', 10.0),
                vad_level=self.options.get('vad_level', 2),
                gap_threshold=self.options.get('gap_threshold', 0.5),
                output_format=self.options.get('output_format', 'txt'),
                srt_max_line=self.options.get('srt_max_line', 50),
                include_silent=self.options.get('include_silent', False),
                debug=self.options.get('debug_segments', False),
                progress_callback=lambda p: self.progress.emit(p),
                status_callback=lambda m: self.status.emit(m),
                cancel_flag=self.is_cancelled,
            )
            self.progress.emit(100)
            if self.is_cancelled():
                self.status.emit("キャンセルされました")
            else:
                self.status.emit("文字起こし完了")
            self.finished_transcription.emit(result)
        except Exception as e:  # 例外はログ + シグナル
            logger.exception("[ERROR] Transcription thread exception:")
            self.error.emit(str(e))
            self.status.emit("エラーが発生しました")


class RangeTranscriptionThread(QThread):
    """部分選択再文字起こし専用スレッド。"""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: str, start_sec: float, end_sec: float, options: dict):
        super().__init__()
        self.video_path = video_path
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.options = options

    def run(self):
        try:
            res = transcribe_range(
                self.video_path,
                self.start_sec,
                self.end_sec,
                model_size=self.options.get('model', 'large-v3'),
                device=self.options.get('device'),
                ja_weight=self.options.get('ja_weight', 1.0),
                ru_weight=self.options.get('ru_weight', 1.0),
                ambiguous_threshold=self.options.get('ambiguous_threshold', 10.0),
                progress_callback=lambda p: self.progress.emit(p),
                status_callback=lambda m: self.status.emit(m),
            )
            self.progress.emit(100)
            self.status.emit('部分再文字起こし完了')
            self.finished.emit(res)
        except Exception as e:
            logger.exception('[ERROR] RangeTranscriptionThread exception:')
            self.error.emit(str(e))
            self.status.emit('部分再文字起こし失敗')


# =============================================================================================
# 部分区間 再文字起こしユーティリティ
# =============================================================================================

_MODEL_CACHE: dict[tuple[str,str], any] = {}

def _load_cached_model(model_size: str, device: str):
    key = (model_size, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = whisper.load_model(model_size, device=device)
    _MODEL_CACHE[key] = m
    return m

def transcribe_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    model_size: str = 'large-v3',
    device: str | None = None,
    ja_weight: float = 1.0,
    ru_weight: float = 1.0,
    ambiguous_threshold: float = 10.0,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """動画ファイル中の指定秒区間 [start_sec, end_sec] を抽出し JA/RU 両言語で再文字起こし。

    戻り値: {
        'start': float, 'end': float,
        'text': str, 'text_ja': str, 'text_ru': str,
        'ja_prob': float, 'ru_prob': float,
        'chosen_language': 'ja' | 'ru'
    }
    例外は呼び出し側で処理。
    """
    if end_sec <= start_sec:
        raise ValueError('end_sec must be greater than start_sec')
    dur = end_sec - start_sec
    if dur > 30.0:
        raise ValueError('選択区間が30秒を超えています')
    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback('部分再文字起こし: モデル取得中...')
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_cached_model(model_size, dev)
    # 一時 wav へ ffmpeg で切り出し (-accurate_seek を期待)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_cut:
        cut_path = tmp_cut.name
    try:
        if status_callback:
            status_callback('部分再文字起こし: 区間抽出中...')
        cmd = [
            'ffmpeg', '-ss', f'{start_sec:.3f}', '-to', f'{end_sec:.3f}', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', cut_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if progress_callback:
            progress_callback(25)
        audio_arr = whisper.load_audio(cut_path)
        if audio_arr.size == 0:
            raise RuntimeError('音声抽出に失敗しました')
        if status_callback:
            status_callback('部分再文字起こし: 言語判定中...')
        detected_lang, ja_prob, ru_prob = _detect_lang_probs(model, audio_arr, ja_weight, ru_weight)
        if progress_callback:
            progress_callback(45)
        if status_callback:
            status_callback('部分再文字起こし: JA 推論中...')
        ja_res = _transcribe_clip(model, audio_arr, 'ja')
        if progress_callback:
            progress_callback(65)
        if status_callback:
            status_callback('部分再文字起こし: RU 推論中...')
        ru_res = _transcribe_clip(model, audio_arr, 'ru')
        if progress_callback:
            progress_callback(85)
        ja_text = clean_hallucination(ja_res.get('text','').strip())
        ru_text = clean_hallucination(ru_res.get('text','').strip())
        if ja_prob >= ru_prob:
            main_text = ja_text
            chosen = 'ja'
        else:
            main_text = ru_text
            chosen = 'ru'
        if status_callback:
            status_callback('部分再文字起こし: 整形中...')
        if progress_callback:
            progress_callback(95)
        return {
            'start': start_sec,
            'end': end_sec,
            'text': main_text,
            'text_ja': ja_text,
            'text_ru': ru_text,
            'ja_prob': ja_prob,
            'ru_prob': ru_prob,
            'chosen_language': chosen,
        }
    finally:
        if os.path.exists(cut_path):
            try:
                os.remove(cut_path)
            except Exception:
                pass
