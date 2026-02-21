"""文字起こし処理コア (faster-whisper 版)。

提供関数:
 - `advanced_process_video(...)` : 動画全体を faster-whisper + Silero VAD で処理
 - `transcribe_range(...)` : 動画の指定秒区間のみを再文字起こし

どちらも GUI 互換の `{'text': str, 'segments': list[dict], 'language': str}` 形式の辞書を返す。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable, Optional

import torch

from faster_whisper.audio import decode_audio as fw_decode_audio

from core.constants import DUP_MERGE_MAX_SEG_DUR, DUP_MERGE_MAX_GAP, MAX_RANGE_SEC
from transcription.audio import (
    clean_hallucination,
    _extract_audio,
    _weight_lang_probs,
)
from transcription.model_cache import _load_cached_model
from utils.timefmt import format_ms
from core.logging_config import get_logger

logger = get_logger(__name__)


def _fmt_ts(t: float) -> str:
    """秒を HH:MM:SS.mmm 形式に変換 (ログ出力用)。"""
    return format_ms(int(t * 1000))


def advanced_process_video(
    video_path: str,
    model_size: str = 'large-v3',
    device: str | None = None,
    languages: list[str] | None = None,
    ja_weight: float = 1.0,
    ru_weight: float = 1.0,
    beam_size: int = 5,
    no_speech_threshold: float = 0.6,
    initial_prompt: str | None = None,
    vad_filter: bool = True,
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 250,
    vad_min_silence_ms: int = 2000,
    ambiguous_threshold: float = 30.0,
    condition_on_previous_text: bool = True,
    compression_ratio_threshold: float = 2.4,
    log_prob_threshold: float = -1.0,
    repetition_penalty: float = 1.0,
    speech_pad_ms: int = 400,
    duplicate_merge: bool = True,
    phase1_beam_size: int | None = None,
    phase2_detect_beam_size: int | None = None,
    phase2_retranscribe_beam_size: int | None = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    segment_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """動画を faster-whisper + Silero VAD で処理し GUI 互換の結果 dict を返す。

    戻り値: {'text': str, 'segments': list[dict], 'language': 'mixed'}
    各 segment dict: {'start', 'end', 'text', 'text_ja', 'text_ru',
                      'chosen_language', 'id', 'ja_prob', 'ru_prob'}
    """
    # Phase別のbeam_sizeのデフォルトフォールバック
    _phase1_beam = phase1_beam_size if phase1_beam_size is not None else beam_size
    _phase2_detect_beam = phase2_detect_beam_size if phase2_detect_beam_size is not None else 1
    _phase2_retrans_beam = phase2_retranscribe_beam_size if phase2_retranscribe_beam_size is not None else beam_size
    
    if languages is None:
        languages = ['ja']
    # Phase 1 は常に自動検出モード (JA/RU 混在に対応)
    lang_arg = None
    multilingual = True

    # デバイス決定
    selected_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_fallback = False
    if selected_device == 'cuda' and not torch.cuda.is_available():
        logger.warning('[DEVICE] cuda 選択されたが利用不可のため cpu へフォールバックします')
        selected_device = 'cpu'
        device_fallback = True
    logger.info(f'[DEVICE] using device={selected_device}')

    model = _load_cached_model(model_size, selected_device)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
        audio_path = tmp_aud.name

    try:
        if status_callback:
            status_callback('[1/3] 音声抽出中...')
        if progress_callback:
            progress_callback(5)
        _extract_audio(video_path, audio_path)

        # ---- Phase 1: 全体一括転写 (テキスト・タイムスタンプ取得) ----
        if status_callback:
            status_callback('[2/3] 全体転写中...')
        if progress_callback:
            progress_callback(10)

        vad_params = {
            'threshold': vad_threshold,
            'min_speech_duration_ms': vad_min_speech_ms,
            'min_silence_duration_ms': vad_min_silence_ms,
            'speech_pad_ms': speech_pad_ms,
        }

        segments_gen, info = model.transcribe(
            audio_path,
            language=lang_arg,
            multilingual=multilingual,
            beam_size=_phase1_beam,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt or None,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            repetition_penalty=repetition_penalty,
            temperature=0.0,
            word_timestamps=False,
        )

        # グローバル言語確率 (フォールバック用)
        ja_global, ru_global = _weight_lang_probs(info.all_language_probs, ja_weight, ru_weight)
        logger.info(
            f'[LANG] all_language_probs size={len(info.all_language_probs or [])} '
            f'ja_global={ja_global:.1f}% ru_global={ru_global:.1f}%'
        )

        # ジェネレータを実体化 (ここで実際の Whisper 推論が走る)
        total_dur = max(info.duration or 1.0, 1.0)
        raw_segs: list[tuple[float, float, str]] = []
        for seg in segments_gen:
            if cancel_flag and cancel_flag():
                if status_callback:
                    status_callback('キャンセル中...')
                break
            text = clean_hallucination(seg.text.strip())
            if not text:
                continue
            raw_segs.append((seg.start, seg.end, text))
            pct1 = min(99, int(seg.end / total_dur * 100))
            if progress_callback:
                progress_callback(10 + int(40 * pct1 / 100))

        # ---- Phase 2: 各クリップ個別言語検出 + 非 JA クリップの再転写 ----
        if status_callback:
            status_callback('[3/3] 言語判定・再転写中...')
        if progress_callback:
            progress_callback(55)

        # WAV を numpy として読み込む (16kHz mono float32)
        SR = 16000
        audio_arr = fw_decode_audio(audio_path, sampling_rate=SR)

        gui_segments: list[dict] = []
        output_lines: list[str] = []
        idx = 0
        total_segs = len(raw_segs)
        
        # 無音区間の最小長さ（秒）
        SILENCE_MIN_GAP = 0.5
        
        # 前のセグメント終了時刻を記録（無音区間検出用）
        prev_end_time = 0.0

        for i, (st, ed, text) in enumerate(raw_segs):
            if cancel_flag and cancel_flag():
                if status_callback:
                    status_callback('キャンセル中...')
                break
            
            # 無音区間の挿入（前のセグメント終了から現在のセグメント開始までのギャップ）
            gap = st - prev_end_time
            if gap >= SILENCE_MIN_GAP:
                silence_seg = {
                    'start': prev_end_time,
                    'end': st,
                    'text': '',
                    'text_ja': '',
                    'text_ru': '',
                    'chosen_language': 'silence',
                    'id': idx,
                    'ja_prob': 0.0,
                    'ru_prob': 0.0,
                }
                gui_segments.append(silence_seg)
                output_lines.append(
                    f'[{_fmt_ts(prev_end_time)} -> {_fmt_ts(st)}] [無音]'
                )
                if segment_callback:
                    try:
                        segment_callback(silence_seg)
                    except Exception:
                        pass
                idx += 1

            # クリップ切り出し
            clip_s = max(0, int(st * SR))
            clip_e = min(len(audio_arr), int(ed * SR))
            clip = audio_arr[clip_s:clip_e]

            # クリップ個別言語検出 (beam_size=_phase2_detect_beam で高速化)
            if len(clip) >= SR // 5:  # 200ms 以上のクリップのみ再検出
                try:
                    _, clip_info = model.transcribe(
                        clip,
                        language=None,
                        beam_size=_phase2_detect_beam,
                        temperature=0.0,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.95,
                    )
                    ja_prob, ru_prob = _weight_lang_probs(
                        clip_info.all_language_probs, ja_weight, ru_weight
                    )
                    logger.debug(
                        f'[CLIP_LANG] #{idx} t={st:.2f}-{ed:.2f} '
                        f'ja={ja_prob:.1f}% ru={ru_prob:.1f}% '
                        f'(clip: {clip_info.language} p={clip_info.language_probability:.2f})'
                    )
                except Exception as e:
                    logger.warning(f'[CLIP_LANG] #{idx} t={st:.2f}-{ed:.2f} 失敗、グローバルで代替: {e}')
                    ja_prob, ru_prob = ja_global, ru_global
            else:
                ja_prob, ru_prob = ja_global, ru_global

            # キリル文字があれば RU に強制 (カタカナ音訳にも保険)
            ru_chars = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
            if ru_chars > 0:
                ru_prob = max(ru_prob, 92.0)
                ja_prob = 100.0 - ru_prob

            chosen_lang = 'ja' if ja_prob >= ru_prob else 'ru'
            # 確率差が ambiguous_threshold 未満なら「あいまい」判定 → 両言語を転写
            is_ambiguous = (
                abs(ja_prob - ru_prob) < ambiguous_threshold
                and len(clip) >= SR // 5
            )

            if chosen_lang == 'ja':
                # Phase 1 テキストはすでに JA 転写なのでそのまま使用
                text_ja = text
                # あいまいなら RU も転写しておく
                if is_ambiguous:
                    try:
                        amb_ru_gen, _ = model.transcribe(
                            clip,
                            language='ru',
                            beam_size=_phase2_retrans_beam,
                            temperature=0.0,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            no_speech_threshold=no_speech_threshold,
                        )
                        amb_ru_texts = [
                            clean_hallucination(s.text.strip())
                            for s in amb_ru_gen
                            if s.text.strip()
                        ]
                        text_ru = ' '.join(amb_ru_texts) if amb_ru_texts else ''
                        logger.debug(
                            f'[AMBIGUOUS_RU] #{idx} t={st:.2f}-{ed:.2f} '
                            f'ja={ja_prob:.1f}% ru={ru_prob:.1f}% '
                            f'text_ru=\"{text_ru[:60]}\"'
                        )
                    except Exception as e:
                        logger.warning(f'[AMBIGUOUS_RU] #{idx} 失敗: {e}')
                        text_ru = ''
                else:
                    text_ru = ''
            else:
                # RU 判定: language='ru' で再転写してキリル文字テキストを取得
                text_ja_orig = text  # Phase 1 の JA テキストを退避
                if len(clip) >= SR // 5:
                    try:
                        ru_gen, _ = model.transcribe(
                            clip,
                            language='ru',
                            beam_size=_phase2_retrans_beam,
                            temperature=0.0,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            no_speech_threshold=no_speech_threshold,
                        )
                        ru_texts = [
                            clean_hallucination(s.text.strip())
                            for s in ru_gen
                            if s.text.strip()
                        ]
                        if ru_texts:
                            text = ' '.join(ru_texts)
                            logger.debug(
                                f'[RU_RETRANSCRIBE] #{idx} t={st:.2f}-{ed:.2f} '
                                f'text=\"{text[:60]}\"'
                            )
                    except Exception as e:
                        logger.warning(f'[RU_RETRANSCRIBE] #{idx} t={st:.2f}-{ed:.2f} 失敗: {e}')
                text_ru = text
                # あいまいなら Phase 1 の JA テキストも保持
                text_ja = text_ja_orig if is_ambiguous else ''

            if progress_callback:
                progress_callback(55 + int(35 * (i + 1) / max(total_segs, 1)))

            # ---------- 重複マージ ----------
            merged = False
            if duplicate_merge and gui_segments:
                prev = gui_segments[-1]
                # 無音セグメントはマージ対象外
                if prev.get('chosen_language') != 'silence':
                    prev_st = float(prev.get('start', st))
                    prev_ed = float(prev.get('end', prev_st))
                    gap = st - prev_ed
                    seg_len = ed - st
                    if (
                        abs(gap) <= DUP_MERGE_MAX_GAP
                        and seg_len <= DUP_MERGE_MAX_SEG_DUR
                        and text == prev.get('text', '')
                    ):
                        prev['end'] = ed
                        prev['ja_prob'] = (prev.get('ja_prob', ja_prob) + ja_prob) / 2.0
                        prev['ru_prob'] = (prev.get('ru_prob', ru_prob) + ru_prob) / 2.0
                        merged = True
                        logger.debug(
                            f'[DUP_MERGE] merged: ({prev_st:.2f}-{prev_ed:.2f}) -> '
                            f'({prev_st:.2f}-{ed:.2f}) text=\'{text[:40]}\''
                        )

            if merged:
                if output_lines:
                    g = gui_segments[-1]
                    output_lines[-1] = (
                        f'[{_fmt_ts(g["start"])} -> {_fmt_ts(ed)}] '
                        f'[JA:{g["ja_prob"]:05.2f}%] [RU:{g["ru_prob"]:05.2f}%] '
                        f'{text}'
                    ).strip()
                # マージした場合も prev_end_time を更新
                prev_end_time = ed
            else:
                line = (
                    f'[{_fmt_ts(st)} -> {_fmt_ts(ed)}] '
                    f'[JA:{ja_prob:05.2f}%] [RU:{ru_prob:05.2f}%] '
                    f'{text}'
                ).strip()
                output_lines.append(line)
                seg_dict = {
                    'start': st,
                    'end': ed,
                    'text': text,
                    'text_ja': text_ja,
                    'text_ru': text_ru,
                    'chosen_language': chosen_lang,
                    'id': idx,
                    'ja_prob': ja_prob,
                    'ru_prob': ru_prob,
                }
                gui_segments.append(seg_dict)
                if segment_callback:
                    try:
                        segment_callback(seg_dict)
                    except Exception:
                        pass
                idx += 1
                prev_end_time = ed
        
        # 最後のセグメントから動画終了までの無音区間を追加
        if raw_segs and total_dur - prev_end_time >= SILENCE_MIN_GAP:
            silence_seg = {
                'start': prev_end_time,
                'end': total_dur,
                'text': '',
                'text_ja': '',
                'text_ru': '',
                'chosen_language': 'silence',
                'id': idx,
                'ja_prob': 0.0,
                'ru_prob': 0.0,
            }
            gui_segments.append(silence_seg)
            output_lines.append(
                f'[{_fmt_ts(prev_end_time)} -> {_fmt_ts(total_dur)}] [無音]'
            )
            if segment_callback:
                try:
                    segment_callback(silence_seg)
                except Exception:
                    pass

        if status_callback:
            status_callback('完了')

        full_text = '\n'.join(output_lines)
        return {
            'text': full_text,
            'segments': gui_segments,
            'language': 'mixed',
            'device_fallback': device_fallback,
        }

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def transcribe_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    model_size: str = 'large-v3',
    device: str | None = None,
    languages: list[str] | None = None,
    ja_weight: float = 1.0,
    ru_weight: float = 1.0,
    beam_size: int = 5,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """動画ファイル中の指定秒区間 [start_sec, end_sec] を抽出し再文字起こし。

    戻り値:
        {
            'start': float, 'end': float,
            'text': str, 'text_ja': str, 'text_ru': str,
            'ja_prob': float, 'ru_prob': float,
            'chosen_language': 'ja' | 'ru'
        }
    """
    if end_sec <= start_sec:
        raise ValueError('end_sec must be greater than start_sec')
    if (end_sec - start_sec) > MAX_RANGE_SEC:
        raise ValueError(f'選択区間が{MAX_RANGE_SEC:.0f}秒を超えています')

    if languages is None:
        languages = ['ja']
    multilingual = len(languages) > 1
    lang_arg = languages[0] if len(languages) == 1 else None

    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback('部分再文字起こし: モデル取得中...')

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_fallback = False
    if dev == 'cuda' and not torch.cuda.is_available():
        logger.warning('[DEVICE] cuda 選択されたが利用不可のため cpu へフォールバックします')
        dev = 'cpu'
        device_fallback = True
    model = _load_cached_model(model_size, dev)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_cut:
        cut_path = tmp_cut.name

    try:
        if status_callback:
            status_callback('部分再文字起こし: 区間抽出中...')
        subprocess.run(
            [
                'ffmpeg',
                '-ss', f'{start_sec:.3f}', '-to', f'{end_sec:.3f}',
                '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', cut_path,
            ],
            check=True,
            capture_output=True,
        )

        SR = 16000
        if progress_callback:
            progress_callback(25)
        if status_callback:
            status_callback('部分再文字起こし: 言語判定中...')

        # Phase A: クリップ単位の言語検出 (advanced_process_video と同方式)
        audio_arr = fw_decode_audio(cut_path, sampling_rate=SR)
        ja_prob: float
        ru_prob: float
        if len(audio_arr) >= SR // 5:
            _det_gen, clip_info = model.transcribe(
                audio_arr,
                language=None,
                multilingual=True,
                beam_size=1,
                vad_filter=False,
                temperature=0.0,
                word_timestamps=False,
            )
            for _ in _det_gen:
                pass
            ja_prob, ru_prob = _weight_lang_probs(clip_info.all_language_probs, ja_weight, ru_weight)
        else:
            ja_prob, ru_prob = 50.0, 50.0
        chosen = 'ja' if ja_prob >= ru_prob else 'ru'

        if progress_callback:
            progress_callback(50)
        if status_callback:
            status_callback('部分再文字起こし: 文字起こし中...')

        # Phase B: 検出言語で本転写
        segments_gen, _ = model.transcribe(
            audio_arr,
            language=chosen,
            multilingual=False,
            beam_size=beam_size,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            temperature=0.0,
            word_timestamps=False,
        )

        texts: list[str] = []
        for seg in segments_gen:
            t = clean_hallucination(seg.text.strip())
            if t:
                texts.append(t)

        if progress_callback:
            progress_callback(80)

        full_text = ' '.join(texts).strip()

        # キリル文字が混入していて JA 判定だった場合は RU で再転写
        ru_chars = sum(1 for c in full_text if '\u0400' <= c <= '\u04ff')
        if ru_chars > 0 and chosen == 'ja':
            ru_gen, _ = model.transcribe(
                audio_arr,
                language='ru',
                multilingual=False,
                beam_size=beam_size,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                temperature=0.0,
                word_timestamps=False,
            )
            ru_texts = [clean_hallucination(s.text.strip()) for s in ru_gen if s.text.strip()]
            if ru_texts:
                full_text = ' '.join(ru_texts).strip()
                chosen = 'ru'
                ru_prob = max(ru_prob, 92.0)
                ja_prob = 100.0 - ru_prob

        text_ja = full_text if chosen == 'ja' else ''
        text_ru = full_text if chosen == 'ru' else ''

        if status_callback:
            status_callback('部分再文字起こし: 完了')
        if progress_callback:
            progress_callback(95)

        logger.debug(
            f'[RANGE] start={start_sec:.3f} end={end_sec:.3f} chosen={chosen} '
            f'ja_prob={ja_prob:.1f} ru_prob={ru_prob:.1f} text=\'{full_text[:60]}\''
        )

        return {
            'start': start_sec,
            'end': end_sec,
            'text': full_text,
            'text_ja': text_ja,
            'text_ru': text_ru,
            'ja_prob': ja_prob,
            'ru_prob': ru_prob,
            'chosen_language': chosen,
            'device_fallback': device_fallback,
        }

    finally:
        if os.path.exists(cut_path):
            try:
                os.remove(cut_path)
            except Exception:
                pass
