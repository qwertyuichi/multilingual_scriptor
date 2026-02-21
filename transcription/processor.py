"""文字起こし処理コア。

提供関数:
 - `advanced_process_video(...)` : 動画全体をハイブリッドセグメント分割→JA/RU 二重推論で処理
 - `transcribe_range(...)` : 動画の指定秒区間のみを再文字起こし

どちらも GUI 互換の `{'text': str, 'segments': list[dict], 'language': str}` 形式の辞書を返す。
"""
from __future__ import annotations

import os
import tempfile
import subprocess
from typing import Callable, Optional

import numpy as np
import torch
import whisper

from core.constants import (
    DEFAULT_SILENCE_RMS_THRESHOLD,
    DEFAULT_MIN_VOICE_RATIO,
    DEFAULT_MAX_SILENCE_REPEAT,
    DUP_MERGE_MAX_SEG_DUR,
    DUP_MERGE_MAX_GAP,
    MAX_RANGE_SEC,
)
from transcription.audio import (
    clean_hallucination,
    _extract_audio,
    _build_hybrid_segments,
    _has_voice,
    _detect_lang_probs,
    _transcribe_clip,
)
from transcription.model_cache import _load_cached_model
from utils.timefmt import format_ms, to_srt_timestamp
from core.logging_config import get_logger

logger = get_logger(__name__)


def _fmt_ts(t: float) -> str:
    """秒を HH:MM:SS.mmm 形式に変換 (ログ出力用)。"""
    return format_ms(int(t * 1000))


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
    duplicate_merge: bool = True,
    duplicate_debug: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    segment_callback: Optional[Callable[[dict], None]] = None,
    silence_rms_threshold: float | None = None,
    min_voice_ratio: float | None = None,
    max_silence_repeat: int | None = None,
) -> dict:
    """動画を高度ルールで処理し GUI 互換の結果 dict を返す。

    戻り値: {'text': str, 'segments': list[dict], 'language': 'mixed'}
    各 segment dict: {'start', 'end', 'text', 'text_ja', 'text_ru',
                      'chosen_language', 'id', 'ja_prob', 'ru_prob'}
    """
    if debug:
        # debug=True 時はルートロガーを DEBUG に下げて全体反映
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    silence_rms_threshold = silence_rms_threshold if silence_rms_threshold is not None else DEFAULT_SILENCE_RMS_THRESHOLD
    min_voice_ratio       = min_voice_ratio       if min_voice_ratio       is not None else DEFAULT_MIN_VOICE_RATIO
    max_silence_repeat    = max_silence_repeat    if max_silence_repeat    is not None else DEFAULT_MAX_SILENCE_REPEAT

    # デバイス決定
    selected_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if selected_device == 'cuda' and not torch.cuda.is_available():
        logger.warning('[DEVICE] cuda 選択されたが利用不可のため cpu へフォールバックします')
        selected_device = 'cpu'
    logger.info(f'[DEVICE] using device={selected_device}')

    # モデルロード
    model = _load_cached_model(model_size, selected_device)
    if segmentation_model_size:
        if segmentation_model_size == model_size:
            seg_model = model  # 同一サイズなら共有
        else:
            # セグメンテーション専用は都度ロード (VRAM 節約)
            seg_model = whisper.load_model(segmentation_model_size, device=selected_device)
    else:
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
        if seg_mode == 'hybrid':
            # JA/RU 二重走査で境界統合
            segs = _build_hybrid_segments(seg_model, audio_path, min_seg_dur=min_seg_dur)
            initial_segments = [{'start': s['start'], 'end': s['end']} for s in segs]
        else:
            # Whisper デフォルト分割
            res = seg_model.transcribe(
                audio_path, language=None, verbose=False,
                word_timestamps=False, condition_on_previous_text=False,
                task='transcribe'
            )
            initial_segments = [
                {'start': s['start'], 'end': s['end']} for s in res.get('segments', [])
            ]

        output_lines: list[str] = []
        gui_segments: list[dict] = []
        srt_entries: list[tuple] = []
        idx = 0
        total_segments = len(initial_segments)

        # 低エネルギー重複テキスト抑止用バッファ
        recent_low_energy_texts: list[str] = []
        LOW_TXT_HISTORY = 8

        for seg in initial_segments:
            if cancel_flag and cancel_flag():
                if status_callback:
                    status_callback('キャンセル中...')
                break

            # 進捗を 10%→90% で線形更新
            if progress_callback is not None and initial_segments:
                prog = 10 + int(80 * (idx + 1) / len(initial_segments))
                progress_callback(prog)

            if status_callback and total_segments:
                step = max(1, total_segments // 50)
                if (idx % step) == 0:
                    status_callback(f'文字起こし中... ({idx + 1}/{total_segments})')

            st = seg['start']
            ed = seg['end']
            start_sample = max(0, int(st * 16000))
            end_sample = min(len(full_audio), int(ed * 16000))
            if end_sample <= start_sample:
                continue

            clip = full_audio[start_sample:end_sample].astype(np.float32)
            if len(clip) < 16000 * 0.1:  # 100ms 未満はスキップ
                if include_silent:
                    output_lines.append(f"[SKIP short {st:.2f}-{ed:.2f}]")
                continue

            # ---------- 無音/低エネルギーフィルタ ----------
            rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
            has_voice_basic, voiced_ratio = _has_voice(clip, vad_level=vad_level, return_analysis=True)
            low_energy = (rms < silence_rms_threshold) or (voiced_ratio < min_voice_ratio)
            if low_energy and not has_voice_basic:
                if include_silent:
                    output_lines.append(
                        f"[SKIP silence_rms {st:.2f}-{ed:.2f} rms={rms:.5f} vr={voiced_ratio:.3f}]"
                    )
                continue

            if not isinstance(clip, np.ndarray) or clip.ndim != 1 or clip.size == 0:
                logger.debug(f"[SKIP] invalid clip shape: {clip.shape}")
                continue

            # ---------- 言語判定 + 両言語推論 ----------
            detected_lang, ja_prob, ru_prob = _detect_lang_probs(model, clip, ja_weight, ru_weight)
            ja_res = _transcribe_clip(model, clip, 'ja')
            ru_res = _transcribe_clip(model, clip, 'ru')
            ja_text = clean_hallucination(ja_res.get('text', '').strip())
            ru_text = clean_hallucination(ru_res.get('text', '').strip())

            # 確率の高い言語をメインテキストとする
            if ja_prob >= ru_prob:
                seg_text = ja_text
                chosen_lang = 'ja'
            else:
                seg_text = ru_text
                chosen_lang = 'ru'

            if not (ja_text or ru_text):
                continue

            # ---------- 低エネルギー下の定型テキスト抑止 ----------
            if low_energy:
                key_pair = f"{ja_text}|{ru_text}"
                if recent_low_energy_texts.count(key_pair) >= max_silence_repeat:
                    if include_silent:
                        output_lines.append(
                            f"[SUPPRESS patterned {st:.2f}-{ed:.2f} '{ja_text[:20]}' "
                            f"rms={rms:.5f} vr={voiced_ratio:.3f}]"
                        )
                    continue
                recent_low_energy_texts.append(key_pair)
                if len(recent_low_energy_texts) > LOW_TXT_HISTORY:
                    recent_low_energy_texts.pop(0)

            ts_start = _fmt_ts(st)
            ts_end   = _fmt_ts(ed)
            line = (
                f"[{ts_start} -> {ts_end}] [JA:{ja_prob:05.2f}%] [RU:{ru_prob:05.2f}%] "
                f"JA={ja_text} | RU={ru_text}"
            ).strip()

            # ---------- 重複マージ ----------
            merged = False
            prev_st = st  # merged ブランチで参照するため初期化
            if duplicate_merge and gui_segments:
                prev = gui_segments[-1]
                prev_st = float(prev.get('start', st))
                prev_ed = float(prev.get('end', prev_st))
                gap = st - prev_ed
                seg_len = ed - st
                if (
                    abs(gap) <= DUP_MERGE_MAX_GAP and
                    seg_len <= DUP_MERGE_MAX_SEG_DUR and
                    ja_text == prev.get('text_ja', '') and
                    ru_text == prev.get('text_ru', '')
                ):
                    # 直前セグメントを時間方向に延長
                    prev['end'] = ed
                    prev['ja_prob'] = (prev.get('ja_prob', ja_prob) + ja_prob) / 2.0
                    prev['ru_prob'] = (prev.get('ru_prob', ru_prob) + ru_prob) / 2.0
                    merged = True
                    if duplicate_debug:
                        logger.debug(
                            f"[DUP_MERGE] merged: ({prev_st:.2f}-{prev_ed:.2f}) -> "
                            f"({prev_st:.2f}-{ed:.2f}) text='{seg_text[:40]}'"
                        )

            if merged:
                # ログ行を延長後の時間で差し替え
                if output_lines:
                    g = gui_segments[-1]
                    output_lines[-1] = (
                        f"[{_fmt_ts(prev_st)} -> {_fmt_ts(ed)}] "
                        f"[JA:{g['ja_prob']:05.2f}%] [RU:{g['ru_prob']:05.2f}%] "
                        f"JA={ja_text} | RU={ru_text}"
                    ).strip()
                # SRT 最終エントリの終端も更新
                if srt_entries:
                    num, st0, _, txt0 = srt_entries[-1]
                    srt_entries[-1] = (num, st0, ed, txt0)
            else:
                output_lines.append(line)
                seg_dict = {
                    'start': st,
                    'end': ed,
                    'text': seg_text,
                    'text_ja': ja_text,
                    'text_ru': ru_text,
                    'chosen_language': chosen_lang,
                    'id': idx,
                    'ja_prob': ja_prob,
                    'ru_prob': ru_prob,
                }
                gui_segments.append(seg_dict)
                srt_entries.append((idx, st, ed, seg_text))
                if segment_callback:
                    try:
                        segment_callback(seg_dict)
                    except Exception:
                        pass
                idx += 1

        if status_callback:
            status_callback('出力整形中...')

        if cancel_flag and cancel_flag():
            if status_callback:
                status_callback('キャンセル完了 (部分結果)')

        full_text = '\n'.join(output_lines)

        # SRT 出力
        if output_format == 'srt':
            blocks: list[str] = []
            for idx0, st, ed, txt in srt_entries:
                if len(txt) > srt_max_line:
                    txt_fmt = '\n'.join(
                        txt[i:i + srt_max_line] for i in range(0, len(txt), srt_max_line)
                    )
                else:
                    txt_fmt = txt
                ts_s = to_srt_timestamp(int(st * 1000))
                ts_e = to_srt_timestamp(int(ed * 1000))
                blocks.append(f"{idx0}\n{ts_s} --> {ts_e}\n{txt_fmt}\n")
            return {
                'text': '\n'.join(blocks),
                'segments': gui_segments,
                'language': 'mixed',
            }

        return {'text': full_text, 'segments': gui_segments, 'language': 'mixed'}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


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
    silence_rms_threshold: float | None = None,
    min_voice_ratio: float | None = None,
) -> dict:
    """動画ファイル中の指定秒区間 [start_sec, end_sec] を抽出し JA/RU 両言語で再文字起こし。

    戻り値:
        {
            'start': float, 'end': float,
            'text': str, 'text_ja': str, 'text_ru': str,
            'ja_prob': float, 'ru_prob': float,
            'chosen_language': 'ja' | 'ru'
        }

    例外は呼び出し側で処理する。
    """
    if end_sec <= start_sec:
        raise ValueError('end_sec must be greater than start_sec')
    dur = end_sec - start_sec
    if dur > MAX_RANGE_SEC:
        raise ValueError(f'選択区間が{MAX_RANGE_SEC:.0f}秒を超えています')

    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback('部分再文字起こし: モデル取得中...')

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_cached_model(model_size, dev)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_cut:
        cut_path = tmp_cut.name

    try:
        if status_callback:
            status_callback('部分再文字起こし: 区間抽出中...')
        cmd = [
            'ffmpeg',
            '-ss', f'{start_sec:.3f}', '-to', f'{end_sec:.3f}',
            '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', cut_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if progress_callback:
            progress_callback(25)

        audio_arr = whisper.load_audio(cut_path)
        if audio_arr.size == 0:
            raise RuntimeError('音声抽出に失敗しました')

        silence_rms_threshold = silence_rms_threshold if silence_rms_threshold is not None else DEFAULT_SILENCE_RMS_THRESHOLD
        min_voice_ratio       = min_voice_ratio       if min_voice_ratio       is not None else DEFAULT_MIN_VOICE_RATIO

        rms = float(np.sqrt(np.mean(np.square(audio_arr)))) if audio_arr.size else 0.0
        has_voice_basic, voiced_ratio = _has_voice(audio_arr, vad_level=2, return_analysis=True)

        if (rms < silence_rms_threshold or voiced_ratio < min_voice_ratio) and not has_voice_basic:
            if status_callback:
                status_callback('部分再文字起こし: 無音区間 (スキップ)')
            return {
                'start': start_sec, 'end': end_sec,
                'text': '', 'text_ja': '', 'text_ru': '',
                'ja_prob': 0.0, 'ru_prob': 0.0,
                'chosen_language': 'ja',
            }

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

        ja_text = clean_hallucination(ja_res.get('text', '').strip())
        ru_text = clean_hallucination(ru_res.get('text', '').strip())

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

        logger.debug(
            f"[RANGE] start={start_sec:.3f} end={end_sec:.3f} chosen={chosen} "
            f"ja_prob={ja_prob:.3f} ru_prob={ru_prob:.3f} "
            f"ja='{ja_text[:60]}' ru='{ru_text[:60]}'"
        )

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
