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
from typing import Any, Callable, Optional

try:
    import ctranslate2
    _CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0
except Exception:
    _CUDA_AVAILABLE = False

from faster_whisper.audio import decode_audio as fw_decode_audio

from core.constants import DUP_MERGE_MAX_SEG_DUR, DUP_MERGE_MAX_GAP, MAX_RANGE_SEC, SILENCE_MIN_GAP
from transcription.audio import (
    _extract_audio,
    _weight_lang_probs,
)
from transcription.model_cache import _load_cached_model
from utils.timefmt import format_ms
from core.logging_config import get_logger

logger = get_logger(__name__)


def _select_device(requested: str | None) -> tuple[str, bool]:
    """デバイス選択とフォールバック処理を共通化。
    
    Returns:
        (selected_device, fallback_occurred)
    """
    selected = requested or ('cuda' if _CUDA_AVAILABLE else 'cpu')
    fallback = False
    if selected == 'cuda' and not _CUDA_AVAILABLE:
        logger.warning('[DEVICE] cuda 選択されたが利用不可のため cpu へフォールバックします')
        selected = 'cpu'
        fallback = True
    return selected, fallback


def _fmt_ts(t: float) -> str:
    """秒を HH:MM:SS.mmm 形式に変換 (ログ出力用)。"""
    return format_ms(int(t * 1000))


def _detect_clip_language(
    model,
    audio_clip: Any,
    sample_rate: int,
    lang1: str,
    lang2: str | None,
    lang1_weight: float,
    lang2_weight: float,
    beam_size: int = 1,
    fallback_probs: tuple[float, float, bool] | None = None,
) -> tuple[str | None, float, float, bool]:
    """音声クリップの言語を検出し、重み付けスコアを返す。
    
    Args:
        model: faster-whisper モデル
        audio_clip: 音声データ (numpy array)
        sample_rate: サンプリングレート
        lang1, lang2: 主要言語コード
        lang1_weight, lang2_weight: スコア補正係数
        beam_size: 言語検出 beam_size (デフォルト1で高速化)
        fallback_probs: 検出失敗時のフォールバック確率 (lang1_prob, lang2_prob, is_confident)
    
    Returns:
        (detected_lang, lang1_prob, lang2_prob, is_confident)
    """
    min_length = sample_rate // 5  # 200ms
    if len(audio_clip) < min_length:
        if fallback_probs:
            return None, *fallback_probs
        return None, 50.0, 50.0, False
    
    try:
        _, clip_info = model.transcribe(
            audio_clip,
            language=None,
            multilingual=True,
            beam_size=beam_size,
            vad_filter=False,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.95,
            word_timestamps=False,
        )
        detected_lang = clip_info.language
        lang1_prob, lang2_prob, is_confident = _weight_lang_probs(
            clip_info.all_language_probs, lang1, lang2, lang1_weight, lang2_weight
        )
        return detected_lang, lang1_prob, lang2_prob, is_confident
    except Exception as e:
        logger.warning(f'[LANG_DETECT] 失敗: {e}')
        if fallback_probs:
            return None, *fallback_probs
        return None, 50.0, 50.0, False


def advanced_process_video(
    video_path: str,
    model_size: str = 'large-v3',
    device: str | None = None,
    languages: list[str] | None = None,
    lang1_weight: float = 1.0,
    lang2_weight: float = 1.0,
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
    # script-boost & hallucination cleanup removed
    debug_prob_log: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    segment_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """動画を faster-whisper + Silero VAD で処理し GUI 互換の結果 dict を返す。

    戻り値: {'text': str, 'segments': list[dict], 'language': 'mixed'}
    各 segment dict: {'start', 'end', 'text', 'text_lang1', 'text_lang2',
                      'chosen_language', 'id', 'lang1_prob', 'lang2_prob',
                      'lang1_code', 'lang2_code'}
    """
    # Phase別のbeam_sizeのデフォルトフォールバック
    _phase1_beam = phase1_beam_size if phase1_beam_size is not None else beam_size
    _phase2_detect_beam = phase2_detect_beam_size if phase2_detect_beam_size is not None else 1
    _phase2_retrans_beam = phase2_retranscribe_beam_size if phase2_retranscribe_beam_size is not None else beam_size
    
    if languages is None:
        languages = ['ja']
    lang1: str = languages[0]
    lang2: str | None = languages[1] if len(languages) > 1 else None
    # Phase 1 は常に自動検出モード (lang1/lang2 混在に対応)

    # デバイス決定
    selected_device, device_fallback = _select_device(device)
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

        phase1_language = None if lang2 else lang1
        phase1_multilingual = True if lang2 else False
        segments_gen, info = model.transcribe(
            audio_path,
            language=phase1_language,
            multilingual=phase1_multilingual,
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
        lang1_global, lang2_global, _ = _weight_lang_probs(
            info.all_language_probs, lang1, lang2, lang1_weight, lang2_weight
        )
        logger.info(
            f'[LANG] all_language_probs size={len(info.all_language_probs or [])} '
            f'lang1({lang1})_global={lang1_global:.1f}% lang2({lang2 or "-"})_global={lang2_global:.1f}%'
        )

        # ジェネレータを実体化 (ここで実際の Whisper 推論が走る)
        total_dur = max(info.duration or 1.0, 1.0)
        raw_segs: list[tuple[float, float, str]] = []
        for seg in segments_gen:
            if cancel_flag and cancel_flag():
                if status_callback:
                    status_callback('キャンセル中...')
                break
            text = seg.text.strip()
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
        
        # 前のセグメント終了時刻を記録（無音区間検出用）
        prev_end_time = 0.0

        for i, (st, ed, text) in enumerate(raw_segs):
            if cancel_flag and cancel_flag():
                if status_callback:
                    status_callback('キャンセル中...')
                break
            raw_text = text
            boost_applied = False
            
            # 無音区間の挿入（前のセグメント終了から現在のセグメント開始までのギャップ）
            gap = st - prev_end_time
            if gap >= SILENCE_MIN_GAP:
                silence_seg = {
                    'start': prev_end_time,
                    'end': st,
                    'text': '',
                    'text_lang1': '',
                    'text_lang2': '',
                    'chosen_language': 'silence',
                    'id': idx,
                    'lang1_prob': 0.0,
                    'lang2_prob': 0.0,
                    'lang1_code': lang1,
                    'lang2_code': lang2 or '',
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
            clip_too_short = len(clip) < SR // 5  # 200ms 未満

            # クリップ個別言語検出 (beam_size=_phase2_detect_beam で高速化)
            detected_lang, lang1_prob, lang2_prob, is_confident = _detect_clip_language(
                model, clip, SR,
                lang1, lang2, lang1_weight, lang2_weight,
                beam_size=_phase2_detect_beam,
                fallback_probs=(lang1_global, lang2_global, False),
            )
            logger.debug(
                f'[CLIP_LANG] #{idx} t={st:.2f}-{ed:.2f} '
                f'{lang1}={lang1_prob:.1f}% {lang2 or "-"}={lang2_prob:.1f}% '
                f'confident={is_confident} detected={detected_lang}'
            )

            handled_other_lang = False
            if detected_lang and detected_lang not in {lang1, lang2}:
                # 選択外言語が最有力: 可能ならその言語で再転写
                chosen_lang = detected_lang
                lang1_prob, lang2_prob = 0.0, 0.0
                text_lang1 = ''
                text_lang2 = ''
                other_text = ''
                if len(clip) >= SR // 5:  # 200ms 以上
                    try:
                        other_gen, _ = model.transcribe(
                            clip,
                            language=detected_lang,
                            beam_size=_phase2_retrans_beam,
                            temperature=0.0,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            no_speech_threshold=no_speech_threshold,
                        )
                        other_texts = [s.text.strip() for s in other_gen if s.text.strip()]
                        other_text = ' '.join(other_texts).strip()
                    except Exception as e:
                        logger.warning(f'[RETRANSCRIBE_{detected_lang}] #{idx} t={st:.2f}-{ed:.2f} 失敗: {e}')
                text = other_text
                is_ambiguous = False
                handled_other_lang = True
            # 選択外言語の可能性が高い場合: Phase1 テキストを保持し 'other' 扱い (BUG-11修正)
            if not handled_other_lang and not is_confident:
                # 低信頼度クリップは選択外言語扱い
                if lang2 is None:
                    # 単言語モードでは指定言語で文字起こしした結果を採用
                    lang1_prob, lang2_prob = 100.0, 0.0
                    chosen_lang = lang1
                    text_lang1 = text
                    text_lang2 = ''
                else:
                    lang1_prob, lang2_prob = 50.0, 50.0
                    chosen_lang = 'other'
                    text_lang1 = text
                    text_lang2 = ''
                is_ambiguous = False
                logger.debug(f'[OTHER_LANG] #{idx} t={st:.2f}-{ed:.2f} 選択言語に低信頼度')
            elif not handled_other_lang:
                # スクリプト補強は削除済み。通常の確率比較で優勢言語を決定。
                chosen_lang = lang1 if lang1_prob >= lang2_prob else (lang2 or lang1)
                is_ambiguous = (
                    lang2 is not None
                    and abs(lang1_prob - lang2_prob) < ambiguous_threshold
                    and not clip_too_short
                )

                if clip_too_short or lang2 is None:
                    # 短すぎる/単言語モード: Phase1 テキストを lang1 として保持 (BUG-1修正)
                    text_lang1 = text
                    text_lang2 = ''
                elif chosen_lang == lang1:
                    # Phase 1 テキストはすでに lang1 転写なのでそのまま使用
                    text_lang1 = text
                    if is_ambiguous:
                        try:
                            amb_gen, _ = model.transcribe(
                                clip,
                                language=lang2,
                                beam_size=_phase2_retrans_beam,
                                temperature=0.0,
                                vad_filter=False,
                                condition_on_previous_text=False,
                                no_speech_threshold=no_speech_threshold,
                            )
                            amb_texts = [s.text.strip() for s in amb_gen if s.text.strip()]
                            text_lang2 = ' '.join(amb_texts) if amb_texts else ''
                            logger.debug(
                                f'[AMBIGUOUS_{lang2}] #{idx} t={st:.2f}-{ed:.2f} '
                                f'text_lang2="{text_lang2[:60]}"'
                            )
                        except Exception as e:
                            logger.warning(f'[AMBIGUOUS_{lang2}] #{idx} 失敗: {e}')
                            text_lang2 = ''
                    else:
                        text_lang2 = ''
                else:
                    # lang2 判定: language=lang2 で再転写
                    text_lang1_orig = text
                    try:
                        lang2_gen, _ = model.transcribe(
                            clip,
                            language=lang2,
                            beam_size=_phase2_retrans_beam,
                            temperature=0.0,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            no_speech_threshold=no_speech_threshold,
                        )
                        lang2_texts = [s.text.strip() for s in lang2_gen if s.text.strip()]
                        if lang2_texts:
                            text = ' '.join(lang2_texts)
                            logger.debug(
                                f'[RETRANSCRIBE_{lang2}] #{idx} t={st:.2f}-{ed:.2f} '
                                f'text="{text[:60]}"'
                            )
                    except Exception as e:
                        logger.warning(f'[RETRANSCRIBE_{lang2}] #{idx} t={st:.2f}-{ed:.2f} 失敗: {e}')
                    text_lang2 = text
                    # あいまいなら Phase 1 の lang1 テキストも保持
                    text_lang1 = text_lang1_orig if is_ambiguous else ''

                if debug_prob_log:
                    clip_len = ed - st
                    logger.info(
                        f'[PROB_DEBUG] #{idx} t={st:.2f}-{ed:.2f} len={clip_len:.2f}s '
                        f'short={clip_too_short} conf={is_confident} '
                        f'boost={"on" if boost_applied else "off"} '
                        f'p1={lang1_prob:.1f} p2={lang2_prob:.1f} chosen={chosen_lang} '
                        f'text="{raw_text[:40]}"'
                    )

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
                        prev['lang1_prob'] = (prev.get('lang1_prob', lang1_prob) + lang1_prob) / 2.0
                        prev['lang2_prob'] = (prev.get('lang2_prob', lang2_prob) + lang2_prob) / 2.0
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
                        f'[{lang1.upper()}:{g["lang1_prob"]:05.2f}%]'
                        + (f' [{lang2.upper()}:{g["lang2_prob"]:05.2f}%]' if lang2 else '')
                        + f' {text}'
                    ).strip()
                # マージした場合も prev_end_time を更新
                prev_end_time = ed
            else:
                line = (
                    f'[{_fmt_ts(st)} -> {_fmt_ts(ed)}] '
                    f'[{lang1.upper()}:{lang1_prob:05.2f}%]'
                    + (f' [{lang2.upper()}:{lang2_prob:05.2f}%]' if lang2 else '')
                    + f' {text}'
                ).strip()
                output_lines.append(line)
                seg_dict = {
                    'start': st,
                    'end': ed,
                    'text': text,
                    'text_lang1': text_lang1,
                    'text_lang2': text_lang2,
                    'chosen_language': chosen_lang,
                    'id': idx,
                    'lang1_prob': lang1_prob,
                    'lang2_prob': lang2_prob,
                    'lang1_code': lang1,
                    'lang2_code': lang2 or '',
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
                'text_lang1': '',
                'text_lang2': '',
                'chosen_language': 'silence',
                'id': idx,
                'lang1_prob': 0.0,
                'lang2_prob': 0.0,
                'lang1_code': lang1,
                'lang2_code': lang2 or '',
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
    lang1_weight: float = 1.0,
    lang2_weight: float = 1.0,
    beam_size: int = 5,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = False,
    # script-boost & hallucination cleanup removed
    debug_prob_log: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """動画ファイル中の指定秒区間 [start_sec, end_sec] を抽出し再文字起こし。

    戻り値:
        {
            'start': float, 'end': float,
            'text': str, 'text_lang1': str, 'text_lang2': str,
            'lang1_prob': float, 'lang2_prob': float,
            'lang1_code': str, 'lang2_code': str | None,
            'chosen_language': str
        }
    """
    if end_sec <= start_sec:
        raise ValueError('end_sec must be greater than start_sec')
    if (end_sec - start_sec) > MAX_RANGE_SEC:
        raise ValueError(f'選択区間が{MAX_RANGE_SEC:.0f}秒を超えています')

    if languages is None:
        languages = ['ja']
    lang1 = languages[0]
    lang2 = languages[1] if len(languages) > 1 else None

    if progress_callback:
        progress_callback(5)
    if status_callback:
        status_callback('部分再文字起こし: モデル取得中...')

    dev, device_fallback = _select_device(device)
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

        # Phase A: クリップ単位の言語検出
        audio_arr = fw_decode_audio(cut_path, sampling_rate=SR)
        
        detected_lang, lang1_prob, lang2_prob, is_confident = _detect_clip_language(
            model, audio_arr, SR,
            lang1, lang2, lang1_weight, lang2_weight,
            beam_size=1,
            fallback_probs=None,
        )

        if detected_lang and detected_lang not in {lang1, lang2}:
            chosen = detected_lang
            lang1_prob, lang2_prob = 0.0, 0.0
        else:
            chosen = lang1 if lang1_prob >= lang2_prob else (lang2 or lang1)

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
            t = seg.text.strip()
            if t:
                texts.append(t)

        if progress_callback:
            progress_callback(80)

        full_text = ' '.join(texts).strip()

        # スクリプト補強は削除済み。選出された言語で得られた転写結果を使用。

        text_lang1 = full_text if chosen == lang1 else ''
        text_lang2 = full_text if (lang2 and chosen == lang2) else ''

        if status_callback:
            status_callback('部分再文字起こし: 完了')
        if progress_callback:
            progress_callback(95)

        if debug_prob_log:
            logger.info(
                f'[PROB_DEBUG_RANGE] start={start_sec:.3f} end={end_sec:.3f} '
                f'lang1={lang1} lang2={lang2 or "-"} p1={lang1_prob:.1f} p2={lang2_prob:.1f} '
                f'chosen={chosen} text="{full_text[:60]}"'
            )

        logger.debug(
            f'[RANGE] start={start_sec:.3f} end={end_sec:.3f} chosen={chosen} '
            f'{lang1}_prob={lang1_prob:.1f} {lang2 or "-"}_prob={lang2_prob:.1f} '
            f'text=\'{full_text[:60]}\''
        )

        return {
            'start': start_sec,
            'end': end_sec,
            'text': full_text,
            'text_lang1': text_lang1,
            'text_lang2': text_lang2,
            'lang1_prob': lang1_prob,
            'lang2_prob': lang2_prob,
            'lang1_code': lang1,
            'lang2_code': lang2,
            'chosen_language': chosen,
            'device_fallback': device_fallback,
        }

    finally:
        if os.path.exists(cut_path):
            try:
                os.remove(cut_path)
            except Exception:
                pass
