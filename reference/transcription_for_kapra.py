import whisper
import argparse
import os
import tempfile
import subprocess
import json
import numpy as np
from pathlib import Path
import torch
import warnings
from scipy.io import wavfile
import webrtcvad
warnings.filterwarnings("ignore")

def build_hybrid_segments(model, audio_path: str, min_seg_dur: float = 0.25):
    """ja / ru の2回 transcription を行い、両者の start/end 境界を統合して細分化区間を返す。
    return: [{'start': float, 'end': float}] 最低長 min_seg_dur でフィルタ。
    Whisper の内部 token 化境界の union を利用し、片言や混在発話の取りこぼし減少を狙う。
    """
    print("[HYBRID] ja固定・ru固定の2パスで境界抽出中...")
    ja_res = model.transcribe(
        audio_path,
        language='ja',
        verbose=False,
        condition_on_previous_text=False,
        word_timestamps=False,
        task='transcribe'
    )
    ru_res = model.transcribe(
        audio_path,
        language='ru',
        verbose=False,
        condition_on_previous_text=False,
        word_timestamps=False,
        task='transcribe'
    )
    points = set()
    for seg in ja_res.get('segments', []):
        points.add(round(float(seg['start']), 2))
        points.add(round(float(seg['end']), 2))
    for seg in ru_res.get('segments', []):
        points.add(round(float(seg['start']), 2))
        points.add(round(float(seg['end']), 2))
    pts = sorted(p for p in points if p >= 0)
    merged = []
    for i in range(len(pts) - 1):
        st = pts[i]; ed = pts[i+1]
        if ed - st >= min_seg_dur:
            merged.append({'start': st, 'end': ed})
    if not merged:  # フォールバック: ja_res をそのまま使う
        return [{'start': s['start'], 'end': s['end']} for s in ja_res.get('segments', [])]
    print(f"[HYBRID] 統合境界数: {len(merged)}")
    return merged

def refine_low_conf_segments(model, audio, segs, lowconf_logprob=-1.05, sample_rate=16000, debug=False):
    """平均 logprob が低い / 長尺でやや低い セグメントを word_timestamps=True で細分化。
    segs: [{'start','end','text','avg_logprob'?}] のリスト
    戻り値: 置換後セグメントリスト（start/end/text を保持）
    """
    refined = []
    for seg in segs:
        st = seg['start']; ed = seg['end']
        dur = ed - st
        avg_lp = seg.get('avg_logprob', 0.0)
        need_refine = (avg_lp < lowconf_logprob and dur > 0.8) or (dur > 4.0 and avg_lp < -0.6)
        if not need_refine:
            refined.append(seg)
            continue
        if debug:
            print(f"[REFINE] {st:.2f}-{ed:.2f}s avg_logprob={avg_lp:.2f} dur={dur:.2f} -> word再分割")
        start_smp = int(st * sample_rate); end_smp = int(ed * sample_rate)
        clip = audio[start_smp:end_smp]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
            tmp_path = tmpf.name
        try:
            wavfile.write(tmp_path, sample_rate, (clip * 32767).astype(np.int16))
            sub_res = model.transcribe(
                tmp_path,
                language=None,
                verbose=False,
                word_timestamps=True,
                condition_on_previous_text=False,
                task='transcribe'
            )
            words = []
            for s2 in sub_res.get('segments', []):
                for w in s2.get('words', []) or []:
                    # word dict: start, end, word
                    if 'start' in w and 'end' in w:
                        words.append(w)
            if len(words) < 2:
                refined.append(seg)
            else:
                # gap >0.6s でチャンク区切り
                cur = [words[0]]
                for w in words[1:]:
                    if w['start'] - cur[-1]['end'] > 0.6:
                        txt = ''.join(x['word'] for x in cur).strip()
                        if txt:
                            refined.append({'start': cur[0]['start'], 'end': cur[-1]['end'], 'text': txt})
                        cur = [w]
                    else:
                        cur.append(w)
                if cur:
                    txt = ''.join(x['word'] for x in cur).strip()
                    if txt:
                        refined.append({'start': cur[0]['start'], 'end': cur[-1]['end'], 'text': txt})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    refined.sort(key=lambda x: x['start'])
    return refined

def extract_audio_from_video(video_path, output_audio_path):
    """動画から音声を抽出"""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # ビデオなし
        '-acodec', 'pcm_s16le',  # 16bit PCM
        '-ar', '16000',  # サンプリングレート16kHz
        '-ac', '1',  # モノラル
        '-y',  # 上書き
        output_audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def detect_language_probabilities(model, audio_segment, ja_weight: float = 1.0, ru_weight: float = 1.0):
    """音声セグメントから日本語とロシア語の確率を計算"""
    # --- 可変長パディング方針 ---
    # 1) 2秒未満の極端に短いクリップは 2 秒にパディング（言語検出が安定）
    # 2) 2秒以上は原音長を尊重し過剰パディングしない
    # 3) 30秒を超える長さは Whisper の内部前提を超えないよう trim
    sr = 16000
    min_len = sr * 2        # 2秒
    max_len = sr * 30       # 30秒（安全上限）
    # 正規化: numpy 1D float32 に揃える
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
        expected_mels = None
        if hasattr(model, 'dims') and hasattr(model.dims, 'n_mels'):
            expected_mels = model.dims.n_mels
        else:
            # encoder.conv1 の入力チャネル数 (=期待メル数)
            expected_mels = model.encoder.conv1.weight.shape[1]
        if not isinstance(expected_mels, int):
            expected_mels = 80
    except Exception:
        expected_mels = 80

    # 期待メル数に合わせて計算（以前 n_mels=128 で学習された派生モデルにも対応）
    try:
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=expected_mels).to(model.device)
    except TypeError:
        mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    except AssertionError as e:
        # 形状関連の例外フォールバック（再フラット化して再試行）
        audio_segment = np.asarray(audio_segment).flatten().astype(np.float32)
        mel = whisper.log_mel_spectrogram(audio_segment, n_mels=expected_mels).to(model.device)

    # 追加フォールバック: 予期しない形状エラー全般
    if mel.shape[0] not in (expected_mels, 80):
        try:
            mel = whisper.log_mel_spectrogram(audio_segment, n_mels=expected_mels).to(model.device)
        except Exception:
            pass

    # 初回のみデバッグ表示（関数属性でフラグ管理）
    if not hasattr(detect_language_probabilities, '_debug_shown'):
        print(f"[DEBUG] language detection uses n_mels={mel.shape[0]} (model expects {expected_mels})")
        detect_language_probabilities._debug_shown = True

    # Whisper の言語検出は基本 30秒分想定だが、短い場合はそのままでも概ね動作するため過剰 pad はしない
    
    # 言語検出
    try:
        with torch.no_grad():
            _, probs = model.detect_language(mel)
    except Exception as e:
        # フォールバック: 30秒 pad + mel 最低長調整再試行
        try:
            sr = 16000
            padded = whisper.pad_or_trim(audio_segment, sr * 30)
            mel2 = whisper.log_mel_spectrogram(padded, n_mels=expected_mels).to(model.device)
            with torch.no_grad():
                _, probs = model.detect_language(mel2)
        except Exception:
            print(f"[WARN] detect_language フォールバック失敗: {e}")
            probs = {'ja': 0.5, 'ru': 0.5}
    
    # 日本語とロシア語の確率を取得 (0-100)
    ja_prob = probs.get('ja', 0.0) * 100
    ru_prob = probs.get('ru', 0.0) * 100

    # 重み付け（尤度に対する簡易補正）
    if ja_weight != 1.0 or ru_weight != 1.0:
        # 0-100 を 0-1 に戻し重み乗算後再正規化
        ja_adj = (ja_prob / 100.0) * ja_weight
        ru_adj = (ru_prob / 100.0) * ru_weight
        denom = ja_adj + ru_adj
        if denom > 0:
            ja_prob = ja_adj / denom * 100.0
            ru_prob = ru_adj / denom * 100.0
        # denom==0 は両方0のまま
    
    # 最も確率の高い言語を選択
    if ja_prob > ru_prob:
        detected_lang = 'ja'
    else:
        detected_lang = 'ru'
    
    return detected_lang, ja_prob, ru_prob

def has_voice(audio_segment: np.ndarray, sample_rate: int = 16000, vad_level: int = 2) -> bool:
    """短いフレームに分割し webrtcvad で音声区間が存在するか簡易判定。
    vad_level: 0(寛容) - 3(厳格)
    戻り値 True: 何らかの音声フレームあり / False: ほぼ無音
    """
    if len(audio_segment) < sample_rate * 0.2:  # 0.2秒未満はそのまま扱う
        return True
    vad = webrtcvad.Vad(vad_level)
    # 30ms フレーム (VAD は 10/20/30ms サポート)
    frame_dur = 30  # ms
    frame_size = int(sample_rate * frame_dur / 1000)
    # 16bit PCM へ変換
    pcm16 = (np.clip(audio_segment, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    voiced = 0
    total = 0
    for offset in range(0, len(pcm16), frame_size * 2):  # 2 bytes per sample
        frame = pcm16[offset: offset + frame_size * 2]
        if len(frame) < frame_size * 2:
            break
        if vad.is_speech(frame, sample_rate):
            voiced += 1
        total += 1
    if total == 0:
        return False
    # 1フレームでも音声判定あれば True（過検出を避けたければ閾値調整可能）
    return voiced > 0

def transcribe_segment_with_audio(model, audio_segment, language):
    """音声セグメントを文字起こし（音声データから直接）"""
    # 音声セグメントを一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # scipy.io.wavfileを使用してWAVファイルを書き込み
        # 音声データを16ビット整数に変換
        audio_int16 = (audio_segment * 32767).astype(np.int16)
        wavfile.write(temp_path, 16000, audio_int16)
        
        # 日本語なまりのロシア語対応のため、temperatureを調整
        result = model.transcribe(
            temp_path,
            language=language,
            temperature=0.2,  # より確実な認識のため低めに設定
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            verbose=False
        )
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def format_time(seconds):
    """秒数を MM:SS.mmm 形式に変換"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"

def process_video(video_path, model_size='large', ja_weight=1.0, ru_weight=1.0, ambiguous_diff_threshold=10.0,
                  vad_level: int = 2, include_silent: bool = False, debug_segments: bool = False,
                  gap_threshold: float = 0.35, seg_mode: str = 'auto', min_seg_dur: float = 0.25,
                  lowconf_logprob: float | None = None, mix_threshold: float = 5.0,
                  output_format: str = 'txt', segmentation_model_size: str | None = None,
                  srt_max_line: int = 42):
    """動画を処理してテキストを生成"""
    print(f"動画を処理中: {video_path}")
    
    # Whisperモデルをロード
    # 二段モデル: segmentation_model_size を指定すればそのモデルで境界、最終は model_size
    if segmentation_model_size:
        print(f"[LOAD] segmentation model ({segmentation_model_size}) をロード中...")
        seg_model = whisper.load_model(segmentation_model_size)
        print(f"[LOAD] final model ({model_size}) をロード中...")
        model = whisper.load_model(model_size)
    else:
        print(f"Whisperモデル ({model_size}) をロード中...")
        model = whisper.load_model(model_size)
        seg_model = model  # 同一
    
    # 一時ファイルで音声を抽出
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    try:
        # 動画から音声抽出
        print("音声を抽出中...")
        extract_audio_from_video(video_path, audio_path)
        
        # 音声全体を読み込み
        audio = whisper.load_audio(audio_path)
        
        # セグメント生成
        print("音声をセグメントに分割中...")
        if seg_mode == 'hybrid':
            hybrid_segments = build_hybrid_segments(seg_model, audio_path, min_seg_dur=min_seg_dur)
            # 既存処理互換のため initial_result 形式へラップ
            initial_result = {'segments': []}
            for seg in hybrid_segments:
                initial_result['segments'].append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': '',
                    'avg_logprob': 0.0
                })
        else:
            initial_result = seg_model.transcribe(
                audio_path,
                language=None,  # 自動検出
                verbose=False,
                word_timestamps=False,
                condition_on_previous_text=False,
                task='transcribe'
            )
        # 低信頼再分割
        if lowconf_logprob is not None and seg_mode != 'hybrid':
            initial_result['segments'] = refine_low_conf_segments(
                seg_model, audio, initial_result['segments'], lowconf_logprob=lowconf_logprob, debug=debug_segments
            )
        
        # 各セグメントを個別に処理
        output_lines = []
        print("セグメントごとに言語検出と文字起こしを実行中...")

        if debug_segments:
            print("[DEBUG] 初期セグメント一覧 (start -> end / len_sec / text)")
            for s in initial_result['segments']:
                st = s['start']; ed = s['end']; txt = s['text'].strip().replace('\n', ' ')
                print(f"  - {format_time(st)} -> {format_time(ed)} / {(ed-st):.3f}s / '{txt}'")

        weighted_mode = (ja_weight != 1.0 or ru_weight != 1.0)

        def safe_avg_logprob(res):
            try:
                segs = res.get('segments') or []
                if len(segs) == 0:
                    return -9999.0
                return segs[0].get('avg_logprob', -9999.0)
            except Exception:
                return -9999.0

        def clean_hallucination(text: str, max_repeat: int = 8) -> str:
            # 1文字が極端に繰り返されている場合縮約（例: プ ルルル...）
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
            # 連続で 30 文字以上の同種 (句読点/同字) があれば警告タグ
            if any(len(block) >= 30 for block in out.split()):
                out = '[HALLUCINATION?] ' + out[:120]
            return out

        prev_end = 0.0
        srt_entries = []  # (index,start,end,text)
        jsonl_entries = []
        seg_index = 1
        for i, segment in enumerate(initial_result['segments']):
            start_time = segment['start']
            end_time = segment['end']

            # 前のセグメントとのギャップを検出し、しきい値を超えた場合プレースホルダ出力
            gap_dur = start_time - prev_end
            if include_silent and gap_dur >= gap_threshold and prev_end > 0:
                gap_line = f"[{format_time(prev_end)} -> {format_time(start_time)}] [GAP:{gap_dur:.2f}s]"
                output_lines.append(gap_line)
                if debug_segments:
                    print(f"  [DEBUG] ギャップ挿入: {gap_line}")
            prev_end = end_time
            
            # セグメントの音声データを取得
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            
            # 範囲チェック
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample <= start_sample:
                continue
                
            audio_segment = audio[start_sample:end_sample].astype(np.float32)
            
            # セグメントが短すぎる場合はスキップ (必要ならプレースホルダ)
            if len(audio_segment) < 16000 * 0.1:  # 0.1秒未満
                if include_silent:
                    placeholder = f"[{format_time(start_time)} -> {format_time(end_time)}] [SKIP:too_short]"
                    output_lines.append(placeholder)
                    if debug_segments:
                        print(f"  [DEBUG] too_short: {placeholder}")
                continue
            
            try:
                # 無音判定 (VAD) - 無音ならスキップ（初回結果のテキストは出さない）
                if not has_voice(audio_segment):
                    if debug_segments:
                        print(f"  [DEBUG] セグメント {i+1}: 無音と判定 (VAD level={vad_level})")
                    if include_silent:
                        placeholder = f"[{format_time(start_time)} -> {format_time(end_time)}] [SKIP:silence vad={vad_level}]"
                        output_lines.append(placeholder)
                    continue

                # 言語検出
                detected_lang, ja_prob, ru_prob = detect_language_probabilities(model, audio_segment, ja_weight, ru_weight)
                if debug_segments:
                    print(f"  [DEBUG] lang probs after weight: JA={ja_prob:.2f}% RU={ru_prob:.2f}% -> {detected_lang}")

                # あいまい判定: 差が閾値未満なら両言語で再トライし平均 logprob 比較
                amb_flag = False
                if abs(ja_prob - ru_prob) < ambiguous_diff_threshold:
                    amb_flag = True
                    ja_result = transcribe_segment_with_audio(model, audio_segment, 'ja')
                    ru_result = transcribe_segment_with_audio(model, audio_segment, 'ru')
                    if debug_segments:
                        print("  [DEBUG] ambiguous -> 再トライ ja/ru 完了")
                    # avg_logprob が高い方を採用 (空配列安全化)
                    ja_score = safe_avg_logprob(ja_result)
                    ru_score = safe_avg_logprob(ru_result)
                    # テキスト抽出
                    ja_text = ja_result.get('text', '').strip()
                    ru_text = ru_result.get('text', '').strip()
                    if ja_score == ru_score:
                        # 同点なら長い方（情報量多い方）
                        chosen = ('ja', ja_result) if len(ja_text) >= len(ru_text) else ('ru', ru_result)
                    else:
                        chosen = ('ja', ja_result) if ja_score > ru_score else ('ru', ru_result)
                    detected_lang = chosen[0]
                    segment_result = chosen[1]
                    # 最新確率を表示用に再計算しなくても良いが、元の確率は保持
                else:
                    # 通常処理
                    segment_result = transcribe_segment_with_audio(model, audio_segment, detected_lang)

                seg_text = segment_result.get('text', '').strip()
                if seg_text:
                    text = clean_hallucination(seg_text)
                    ja_percent = int(round(ja_prob))
                    ru_percent = int(round(ru_prob))
                    amb_tag = ' AMB' if amb_flag else ''
                    weight_tag = ' [W]' if weighted_mode else ''
                    mix_tag = ''
                    if abs(ja_prob - ru_prob) < mix_threshold:
                        mix_tag = ' [MIX]'
                    line = f"[{format_time(start_time)} -> {format_time(end_time)}]{amb_tag}{weight_tag}{mix_tag} [JA:{ja_percent:03d}%] [RU:{ru_percent:03d}%] {text}"
                    output_lines.append(line)
                    # SRT/JSONL 用に保存
                    srt_entries.append((seg_index, start_time, end_time, text))
                    jsonl_entries.append({
                        'index': seg_index,
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'ja_prob': ja_prob,
                        'ru_prob': ru_prob,
                        'ambiguous': amb_flag,
                        'mix': bool(mix_tag),
                        'language': 'ja' if ja_prob >= ru_prob else 'ru'
                    })
                    seg_index += 1
                    if debug_segments:
                        print(f"  [DEBUG] 出力: {line}")
                else:
                    if include_silent:
                        placeholder = f"[{format_time(start_time)} -> {format_time(end_time)}] [EMPTY]"
                        output_lines.append(placeholder)
                        if debug_segments:
                            print(f"  [DEBUG] empty_text: {placeholder}")
            except Exception as e:
                print(f"  セグメント {i+1} の処理中にエラー: {e}")
                # エラーが発生した場合は元のテキストを使用
                if segment['text'].strip():
                    text = segment['text'].strip()
                    line = f"[{format_time(start_time)} -> {format_time(end_time)}] [JA:---%] [RU:---%] {text}"
                    output_lines.append(line)
        
        # フォーマット出力構築
        if output_format == 'txt':
            return '\n'.join(output_lines)
        elif output_format == 'srt':
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
                    # 単純分割（日本語はスペース無いこと多いので幅で切る）
                    lines = [txt[i:i+srt_max_line] for i in range(0, len(txt), srt_max_line)]
                    txt_fmt = '\n'.join(lines)
                else:
                    txt_fmt = txt
                blocks.append(f"{idx}\n{to_srt_timestamp(st)} --> {to_srt_timestamp(ed)}\n{txt_fmt}\n")
            return '\n'.join(blocks)
        elif output_format == 'jsonl':
            return '\n'.join(json.dumps(e, ensure_ascii=False) for e in jsonl_entries)
        else:
            print(f"[WARN] 未知の output_format '{output_format}' -> txt で出力")
            return '\n'.join(output_lines)
    
    finally:
        # 一時ファイルを削除
        if os.path.exists(audio_path):
            os.remove(audio_path)

def main():
    parser = argparse.ArgumentParser(description='日本語・ロシア語動画文字起こしプログラム')
    parser.add_argument('video_path', help='処理する動画ファイル（mp4）のパス')
    parser.add_argument('--model', default='large', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='使用するWhisperモデルのサイズ（デフォルト: large）')
    parser.add_argument('--output', '-o', help='出力ファイルのパス。未指定なら output/ 元動画ファイル名 + 拡張子(出力形式) で自動保存')
    parser.add_argument('--ja-weight', type=float, default=1.0, help='日本語確率へ乗算する重み (デフォルト1.0)')
    parser.add_argument('--ru-weight', type=float, default=1.0, help='ロシア語確率へ乗算する重み (デフォルト1.0)')
    parser.add_argument('--ambiguous-threshold', type=float, default=10.0, help='あいまい再トライを行う確率差しきい値(パーセント)')
    parser.add_argument('--vad-level', type=int, default=2, choices=[0,1,2,3], help='VAD 厳しさ (0=寛容,3=厳格)')
    parser.add_argument('--include-silent', action='store_true', help='スキップされた無音/短すぎ/空テキストもプレースホルダ表示')
    parser.add_argument('--debug-segments', action='store_true', help='デバッグ: 各セグメントの判定理由を表示')
    parser.add_argument('--gap-threshold', type=float, default=0.35, help='ギャップ挿入の下限秒数 (include-silent時)')
    parser.add_argument('--seg-mode', choices=['auto','hybrid'], default='auto', help='セグメント生成方式: auto=従来1パス, hybrid=ja/ru二重パス境界統合')
    parser.add_argument('--min-seg-dur', type=float, default=0.25, help='hybrid 方式で残す最短区間秒数')
    parser.add_argument('--lowconf-logprob', type=float, default=None, help='avg_logprob がこの値未満のセグメントを word 再分割 (auto モードのみ)')
    parser.add_argument('--mix-threshold', type=float, default=5.0, help='JA/RU 確率差がこの値未満なら [MIX] タグ付与')
    parser.add_argument('--output-format', choices=['txt','srt','jsonl'], default='txt', help='出力フォーマット')
    parser.add_argument('--seg-model', dest='segmentation_model_size', help='二段モデル: セグメント生成専用モデル (例: small)')
    parser.add_argument('--srt-max-line', type=int, default=42, help='SRT 出力時の1行最大文字数（超過で任意位置改行）')
    
    args = parser.parse_args()
    
    # 動画ファイルの存在確認
    if not os.path.exists(args.video_path):
        print(f"エラー: 動画ファイル '{args.video_path}' が見つかりません")
        return
    
    # 処理実行
    result = process_video(
        args.video_path,
        args.model,
        ja_weight=args.ja_weight,
        ru_weight=args.ru_weight,
        ambiguous_diff_threshold=args.ambiguous_threshold,
        vad_level=args.vad_level,
        include_silent=args.include_silent,
        debug_segments=args.debug_segments,
        gap_threshold=args.gap_threshold,
        seg_mode=args.seg_mode,
        min_seg_dur=args.min_seg_dur,
        lowconf_logprob=args.lowconf_logprob,
        mix_threshold=args.mix_threshold,
        output_format=args.output_format,
        segmentation_model_size=args.segmentation_model_size,
        srt_max_line=args.srt_max_line
    )
    
    # 出力パス決定
    if args.output:
        out_path = Path(args.output)
    else:
        # 自動決定: output/<video_stem>.<ext>
        video_stem = Path(args.video_path).stem
        ext = args.output_format if args.output_format != 'txt' else 'txt'
        out_dir = Path('output')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_stem}.{ext}"

    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding='utf-8')
    print(f"\n保存先: {out_path}")
    if not args.output:
        print("( --output 未指定のため自動保存 )")

if __name__ == "__main__":
    main()
