def clean_hallucination(text: str, max_repeat: int = 8) -> str:
    """1文字が極端に繰り返されている場合縮約。連続30文字以上の同種があれば警告タグ。"""
    cleaned = []
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
    # 連続で30文字以上の同種(句読点/同字)があれば警告タグ
    if any(len(block) >= 30 for block in out.split()):
        out = '[HALLUCINATION?] ' + out[:120]
    return out
import whisper
import torch
from PySide6.QtCore import QThread, Signal
import os
import json
import tempfile
import subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import webrtcvad
from datetime import timedelta
import traceback

# =============== 高度処理用ユーティリティ (reference/transcription_for_kapra.py から抽出/簡略化) ===============

def _extract_audio(video_path: str, output_audio_path: str):
    cmd = [
        'ffmpeg','-i', video_path,
        '-vn','-acodec','pcm_s16le','-ar','16000','-ac','1','-y',output_audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def _build_hybrid_segments(model, audio_path: str, min_seg_dur: float = 0.25):
    print("[ADV] Hybrid segmentation ...")
    ja_res = model.transcribe(audio_path, language='ja', verbose=False, condition_on_previous_text=False,
                              word_timestamps=False, task='transcribe')
    ru_res = model.transcribe(audio_path, language='ru', verbose=False, condition_on_previous_text=False,
                              word_timestamps=False, task='transcribe')
    points = set()
    for seg in ja_res.get('segments', []):
        points.add(round(float(seg['start']), 2)); points.add(round(float(seg['end']), 2))
    for seg in ru_res.get('segments', []):
        points.add(round(float(seg['start']), 2)); points.add(round(float(seg['end']), 2))
    pts = sorted(p for p in points if p >= 0)
    merged = []
    for i in range(len(pts)-1):
        st, ed = pts[i], pts[i+1]
        if ed - st >= min_seg_dur:
            merged.append({'start': st, 'end': ed})
    if not merged:
        return [{'start': s['start'], 'end': s['end']} for s in ja_res.get('segments', [])]
    return merged

def _has_voice(segment: np.ndarray, sample_rate: int = 16000, vad_level: int = 2) -> bool:
    if len(segment) < sample_rate * 0.2:
        return True
    vad = webrtcvad.Vad(vad_level)
    frame_dur = 30
    frame_size = int(sample_rate * frame_dur / 1000)
    pcm16 = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    for offset in range(0, len(pcm16), frame_size * 2):
        frame = pcm16[offset: offset + frame_size * 2]
        if len(frame) < frame_size * 2:
            break
        if vad.is_speech(frame, sample_rate):
            return True
    return False

def _detect_lang_probs(model, audio_segment, ja_weight=1.0, ru_weight=1.0, ru_accent_boost: float = 0.0):
    """JA/RU の言語確率のみを取得し、重み補正後に返す。ENは無視。"""
    # reference版と同じpad/trimming・例外時pad再試行方式
    sr = 16000
    min_len = sr * 2
    max_len = sr * 30
    if isinstance(audio_segment, torch.Tensor):
        audio_segment = audio_segment.detach().cpu().numpy()
    audio_segment = np.asarray(audio_segment).flatten().astype(np.float32)
    length = len(audio_segment)
    if length < min_len:
        audio_segment = whisper.pad_or_trim(audio_segment, min_len)
    elif length > max_len:
        audio_segment = whisper.pad_or_trim(audio_segment, max_len)
    # モデルが期待するメルバンド数を動的取得
    try:
        if hasattr(model, 'dims') and hasattr(model.dims, 'n_mels'):
            expected_mels = model.dims.n_mels
        else:
            expected_mels = model.encoder.conv1.weight.shape[1]
        if not isinstance(expected_mels, int):
            expected_mels = 80
    except Exception:
        expected_mels = 80
    # 期待メル数に合わせて計算
    try:
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=expected_mels).to(model.device)
    except TypeError:
        mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    except AssertionError:
        audio_segment = np.asarray(audio_segment).flatten().astype(np.float32)
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=expected_mels).to(model.device)
    if mel.shape[0] not in (expected_mels, 80):
        try:
            mel = whisper.log_mel_spectrogram(audio_segment, n_mels=expected_mels).to(model.device)
        except Exception:
            pass
    # 言語検出
    try:
        with torch.no_grad():
            _, probs = model.detect_language(mel)
        print(f"[LANG_PROB] probs={probs}")
    except Exception as e:
        # フォールバック: 30秒pad + mel再試行
        try:
            padded = whisper.pad_or_trim(audio_segment, sr * 30)
            mel2 = whisper.log_mel_spectrogram(padded, n_mels=expected_mels).to(model.device)
            with torch.no_grad():
                _, probs = model.detect_language(mel2)
        except Exception:
            print(f"[LANG_PROB][EXCEPTION] {e}. fallback probs={{'ja':0.5,'ru':0.5}}")
            probs = {'ja':0.5,'ru':0.5}
    ja_raw = float(probs.get('ja',0.0))
    ru_raw = float(probs.get('ru',0.0))
    # 重み補正 (正規化)
    ja_adj = ja_raw * ja_weight
    ru_adj = ru_raw * ru_weight
    denom = ja_adj + ru_adj
    if denom > 0:
        ja_prob = ja_adj / denom * 100.0
        ru_prob = ru_adj / denom * 100.0
    else:
        ja_prob = ru_prob = 50.0
    if ru_accent_boost > 0 and ru_prob < ja_prob:
        ru_prob = min(100.0, ru_prob + ru_accent_boost)
    detected_lang = 'ja' if ja_prob >= ru_prob else 'ru'
    return detected_lang, ja_prob, ru_prob

def _transcribe_clip(model, audio_segment, language):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        wavfile.write(tmp_path, 16000, (audio_segment*32767).astype(np.int16))
        res = model.transcribe(tmp_path, language=language, temperature=0.2, no_speech_threshold=0.6,
                               logprob_threshold=-1.0, verbose=False, condition_on_previous_text=False,
                               task='transcribe')
        return res
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def advanced_process_video(
        video_path: str,
        model_size: str = 'large-v3',
        segmentation_model_size: str | None = 'turbo',
        seg_mode: str = 'hybrid',
        ja_weight: float = 0.80,
        ru_weight: float = 1.25,
        ru_accent_boost: float = 0.0,
        min_seg_dur: float = 0.60,
        ambiguous_threshold: float = 10.0,
        vad_level: int = 2,
        gap_threshold: float = 0.5,
        output_format: str = 'txt',
        srt_max_line: int = 50,
        include_silent: bool = False,
        debug: bool = False,
        progress_callback=None,
    ) -> dict:
    """動画を高度ルールで処理し GUI 互換の結果 dict を返す。
    戻り値: {'text': str, 'segments': [{'start','end','text','id'}], 'language': 'mixed'}
    """
    print('[ADV] loading models ...')
    if segmentation_model_size:
        seg_model = whisper.load_model(segmentation_model_size)
        model = whisper.load_model(model_size)
    else:
        model = whisper.load_model(model_size)
        seg_model = model
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_aud:
        audio_path = tmp_aud.name
    try:
        _extract_audio(video_path, audio_path)
        full_audio = whisper.load_audio(audio_path)
        if seg_mode == 'hybrid':
            segs = _build_hybrid_segments(seg_model, audio_path, min_seg_dur=min_seg_dur)
            initial_segments = [{'start': s['start'], 'end': s['end']} for s in segs]
        else:
            res = seg_model.transcribe(audio_path, language=None, verbose=False, word_timestamps=False,
                                       condition_on_previous_text=False, task='transcribe')
            initial_segments = [{'start': s['start'], 'end': s['end']} for s in res.get('segments', [])]
        output_lines = []
        gui_segments = []
        srt_entries = []
        jsonl_entries = []
        prev_end = 0.0
        idx = 0
        for seg in initial_segments:
            # 進捗を10%→90%で細かくemit
            if progress_callback is not None:
                prog = 10 + int(80 * (idx + 1) / len(initial_segments)) if len(initial_segments) > 0 else 10
                progress_callback(prog)
            st = seg['start']; ed = seg['end']
            gap = st - prev_end
            if include_silent and gap >= gap_threshold and prev_end>0:
                output_lines.append(f"[GAP {gap:.2f}s]")
            prev_end = ed
            start_sample = int(st*16000); end_sample = int(ed*16000)
            start_sample = max(0,start_sample); end_sample = min(len(full_audio), end_sample)
            if end_sample <= start_sample: continue
            clip = full_audio[start_sample:end_sample].astype(np.float32)
            if len(clip) < 16000*0.1:
                if include_silent: output_lines.append(f"[SKIP short {st:.2f}-{ed:.2f}]")
                continue
            if not _has_voice(clip, vad_level=vad_level):
                if include_silent: output_lines.append(f"[SKIP silence {st:.2f}-{ed:.2f}]")
                continue
            # クリップshapeチェック・値確認（リサンプリングは行わずpad/trimmingのみ）
            if not isinstance(clip, np.ndarray) or clip.ndim != 1 or clip.size == 0:
                print(f"[SKIP] invalid clip shape: {clip.shape}")
                continue
            detected_lang, ja_prob, ru_prob = _detect_lang_probs(
                model, clip, ja_weight, ru_weight, ru_accent_boost=ru_accent_boost
            )
            amb = abs(ja_prob - ru_prob) < ambiguous_threshold
            if amb:
                ja_res = _transcribe_clip(model, clip, 'ja')
                ru_res = _transcribe_clip(model, clip, 'ru')
                def safe_avg_logprob(res):
                    segs = res.get('segments') or []
                    if len(segs) == 0:
                        return -9999.0
                    return segs[0].get('avg_logprob', -9999.0)
                ja_score = safe_avg_logprob(ja_res)
                ru_score = safe_avg_logprob(ru_res)
                ja_text = ja_res.get('text','').strip(); ru_text = ru_res.get('text','').strip()
                if ja_score == ru_score:
                    if len(ja_text) >= len(ru_text):
                        detected_lang, seg_text = 'ja', ja_text
                    else:
                        detected_lang, seg_text = 'ru', ru_text
                elif ja_score > ru_score:
                    detected_lang, seg_text = 'ja', ja_text
                else:
                    detected_lang, seg_text = 'ru', ru_text
            else:
                seg_res = _transcribe_clip(model, clip, detected_lang)
                seg_text = seg_res.get('text','').strip()
            if not seg_text:
                continue
            seg_text = clean_hallucination(seg_text)
            # [MIX]機能削除
            def fmt_ts(t: float):
                td = timedelta(seconds=t)
                total_seconds = td.total_seconds()
                h = int(total_seconds // 3600)
                m = int((total_seconds % 3600) // 60)
                s = total_seconds % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}"
            ts_start = fmt_ts(st)
            ts_end = fmt_ts(ed)
            line = f"[{ts_start} -> {ts_end}] [JA:{ja_prob:05.2f}%] [RU:{ru_prob:05.2f}%] {seg_text}".strip()
            output_lines.append(line)
            gui_segments.append({'start': st, 'end': ed, 'text': seg_text, 'id': idx, 'ja_prob': ja_prob, 'ru_prob': ru_prob})
            # SRT/JSONL用
            srt_entries.append((idx+1, st, ed, seg_text))
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
        full_text = '\n'.join(output_lines)
        # SRT/JSONL出力対応
        if output_format == 'srt':
            def to_srt_timestamp(sec: float):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                ms = int((sec - int(sec)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            blocks = []
            for idx, st, ed, txt in srt_entries:
                # 長過ぎる行を適宜改行
                if len(txt) > srt_max_line:
                    lines = [txt[i:i+srt_max_line] for i in range(0, len(txt), srt_max_line)]
                    txt_fmt = '\n'.join(lines)
                else:
                    txt_fmt = txt
                blocks.append(f"{idx}\n{to_srt_timestamp(st)} --> {to_srt_timestamp(ed)}\n{txt_fmt}\n")
            srt_text = '\n'.join(blocks)
            return {'text': srt_text, 'segments': gui_segments, 'language': 'mixed'}
        elif output_format == 'jsonl':
            import json
            jsonl_text = '\n'.join(json.dumps(e, ensure_ascii=False) for e in jsonl_entries)
            return {'text': jsonl_text, 'segments': gui_segments, 'language': 'mixed'}
        else:
            return {'text': full_text, 'segments': gui_segments, 'language': 'mixed'}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ====================================================================================================

class TranscriptionThread(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_transcription = Signal(dict)
    error = Signal(str)
    
    def __init__(self, video_path, options):
        super().__init__()
        self.video_path = video_path
        self.options = options
        
    def run(self):
        try:
            self.status.emit("モデルを読み込み中...")
            self.progress.emit(10)
            # デバイスの設定
            # 高度処理: advanced_process_video のみ利用
            adv_result = advanced_process_video(
                self.video_path,
                model_size=self.options.get('model', 'large-v3'),
                segmentation_model_size=self.options.get('segmentation_model_size', 'turbo'),
                seg_mode=self.options.get('seg_mode', 'hybrid'),
                ja_weight=self.options.get('ja_weight', 0.80),
                ru_weight=self.options.get('ru_weight', 1.25),
                ru_accent_boost=self.options.get('ru_accent_boost', 0.0),
                min_seg_dur=self.options.get('min_seg_dur', 0.60),
                ambiguous_threshold=self.options.get('ambiguous_threshold', 10.0),
                vad_level=self.options.get('vad_level', 2),
                gap_threshold=self.options.get('gap_threshold', 0.5),
                output_format=self.options.get('output_format', 'txt'),
                srt_max_line=self.options.get('srt_max_line', 50),
                include_silent=self.options.get('include_silent', False),
                debug=self.options.get('debug_segments', False),
                progress_callback=lambda p: self.progress.emit(p)
            )
            transcription_result = adv_result
            self.progress.emit(100)
            self.status.emit("文字起こし完了")
            self.finished_transcription.emit(transcription_result)
            
        except Exception as e:
            # 標準出力へスタックトレース付きでエラー表示
            print("[ERROR] Transcription thread exception:", e)
            traceback.print_exc()
            self.error.emit(str(e))
            self.status.emit("エラーが発生しました")
