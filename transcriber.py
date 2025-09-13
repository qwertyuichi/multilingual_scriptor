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

def _detect_lang_probs(model, audio_segment, ja_weight=1.0, ru_weight=1.0, en_weight=1.0):
    if isinstance(audio_segment, torch.Tensor):
        audio_segment = audio_segment.detach().cpu().numpy()
    audio_segment = np.asarray(audio_segment).flatten().astype(np.float32)
    sr = 16000
    if len(audio_segment) < sr*2:
        audio_segment = whisper.pad_or_trim(audio_segment, sr*2)
    elif len(audio_segment) > sr*30:
        audio_segment = whisper.pad_or_trim(audio_segment, sr*30)
    mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    try:
        with torch.no_grad():
            _, probs = model.detect_language(mel)
    except Exception:
        probs = {'ja':0.5,'ru':0.5,'en':0.5}
    ja_prob = probs.get('ja',0.0)*100; ru_prob = probs.get('ru',0.0)*100; en_prob = probs.get('en',0.0)*100
    if ja_weight!=1.0 or ru_weight!=1.0 or en_weight!=1.0:
        ja_adj=(ja_prob/100)*ja_weight; ru_adj=(ru_prob/100)*ru_weight; en_adj=(en_prob/100)*en_weight; denom=ja_adj+ru_adj+en_adj
        if denom>0:
            ja_prob=ja_adj/denom*100; ru_prob=ru_adj/denom*100; en_prob=en_adj/denom*100
    detected_lang = 'ja' if ja_prob >= ru_prob and ja_prob >= en_prob else 'ru' if ru_prob >= en_prob else 'en'
    return detected_lang, ja_prob, ru_prob, en_prob

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
        en_weight: float = 1.00,
        min_seg_dur: float = 0.60,
        ambiguous_threshold: float = 10.0,
        mix_threshold: float = 6.0,
        vad_level: int = 2,
        gap_threshold: float = 0.5,
        output_format: str = 'txt',
        srt_max_line: int = 50,
        include_silent: bool = False,
        debug: bool = False,
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
        prev_end = 0.0
        idx = 0
        for seg in initial_segments:
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
            detected_lang, ja_prob, ru_prob, en_prob = _detect_lang_probs(model, clip, ja_weight, ru_weight, en_weight)
            amb = abs(ja_prob - ru_prob) < ambiguous_threshold
            if amb:
                ja_res = _transcribe_clip(model, clip, 'ja')
                ru_res = _transcribe_clip(model, clip, 'ru')
                # choose longer text if tie
                ja_text = ja_res.get('text','').strip(); ru_text = ru_res.get('text','').strip()
                if len(ja_text) >= len(ru_text):
                    chosen = ('ja', ja_text)
                else:
                    chosen = ('ru', ru_text)
                detected_lang = chosen[0]; seg_text = chosen[1]
            else:
                seg_res = _transcribe_clip(model, clip, detected_lang)
                seg_text = seg_res.get('text','').strip()
            if not seg_text:
                continue
            mix_tag = '[MIX]' if abs(ja_prob - ru_prob) < mix_threshold else ''
            line = f"[{st:07.2f}->{ed:07.2f}] [JA:{ja_prob:05.2f}% RU:{ru_prob:05.2f}%]{' [AMB]' if amb else ''} {mix_tag} {seg_text}".strip()
            output_lines.append(line)
            gui_segments.append({'start': st, 'end': ed, 'text': seg_text, 'id': idx})
            idx += 1
        full_text = '\n'.join(output_lines)
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
            advanced = self.options.get('advanced', False)
            if advanced:
                self.status.emit("高度モード: モデル読み込み中...")
            else:
                self.status.emit("モデルを読み込み中...")
            self.progress.emit(10)
            
            # デバイスの設定
            device = self.options.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            if advanced:
                # 高度処理: advanced_process_video を利用
                adv_result = advanced_process_video(
                    self.video_path,
                    model_size=self.options.get('model', 'large-v3'),
                    segmentation_model_size=self.options.get('segmentation_model_size', 'turbo'),
                    seg_mode=self.options.get('seg_mode', 'hybrid'),
                    ja_weight=self.options.get('ja_weight', 0.80),
                    ru_weight=self.options.get('ru_weight', 1.25),
                    en_weight=self.options.get('en_weight', 1.00),
                    min_seg_dur=self.options.get('min_seg_dur', 0.60),
                    ambiguous_threshold=self.options.get('ambiguous_threshold', 10.0),
                    mix_threshold=self.options.get('mix_threshold', 6.0),
                    vad_level=self.options.get('vad_level', 2),
                    gap_threshold=self.options.get('gap_threshold', 0.5),
                    output_format=self.options.get('output_format', 'txt'),
                    srt_max_line=self.options.get('srt_max_line', 50),
                    include_silent=self.options.get('include_silent', False),
                    debug=self.options.get('debug_segments', False)
                )
                transcription_result = adv_result
                self.progress.emit(100)
            else:
                model = whisper.load_model(
                    self.options.get('model', 'base'),
                    device=device,
                    download_root=self.options.get('model_dir', None)
                )
                self.progress.emit(30)
                self.status.emit("音声を抽出中...")
                transcribe_options = {
                    'language': self.options.get('language', None),
                    'task': self.options.get('task', 'transcribe'),
                    'temperature': self.options.get('temperature', 0),
                    'best_of': self.options.get('best_of', 5),
                    'beam_size': self.options.get('beam_size', 5),
                    'patience': self.options.get('patience', None),
                    'length_penalty': self.options.get('length_penalty', None),
                    'suppress_tokens': self.options.get('suppress_tokens', '-1'),
                    'initial_prompt': self.options.get('initial_prompt', None),
                    'condition_on_previous_text': self.options.get('condition_on_previous_text', True),
                    'fp16': self.options.get('fp16', True),
                    'compression_ratio_threshold': self.options.get('compression_ratio_threshold', 2.4),
                    'logprob_threshold': self.options.get('logprob_threshold', -1.0),
                    'no_speech_threshold': self.options.get('no_speech_threshold', 0.6),
                    'word_timestamps': self.options.get('word_timestamps', False),
                    'prepend_punctuations': self.options.get('prepend_punctuations', r"\"'¿([{-"),
                    'append_punctuations': self.options.get('append_punctuations', r"\"'.。,，!！?？:：”)]}、"),
                }
                transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
                if device == 'cpu' and transcribe_options.get('fp16'):  # CPU では fp16 無効
                    transcribe_options['fp16'] = False
                self.progress.emit(50)
                self.status.emit("文字起こしを実行中...")
                result = model.transcribe(self.video_path, **transcribe_options)
                self.progress.emit(90)
                self.status.emit("結果を処理中...")
                segments = [{
                    'start': s['start'], 'end': s['end'], 'text': s['text'].strip(), 'id': s['id']
                } for s in result['segments']]
                transcription_result = {
                    'text': result['text'],
                    'segments': segments,
                    'language': result.get('language', 'unknown')
                }
            
            self.progress.emit(100)
            self.status.emit("文字起こし完了")
            self.finished_transcription.emit(transcription_result)
            
        except Exception as e:
            self.error.emit(str(e))
            self.status.emit("エラーが発生しました")
