import whisper
import argparse
import os
import tempfile
import subprocess
import numpy as np
from pathlib import Path
import torch
import warnings
from scipy.io import wavfile
import webrtcvad
warnings.filterwarnings("ignore")

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
                  gap_threshold: float = 0.35):
    """動画を処理してテキストを生成"""
    print(f"動画を処理中: {video_path}")
    
    # Whisperモデルをロード
    print(f"Whisperモデル ({model_size}) をロード中...")
    model = whisper.load_model(model_size)
    
    # 一時ファイルで音声を抽出
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    try:
        # 動画から音声抽出
        print("音声を抽出中...")
        extract_audio_from_video(video_path, audio_path)
        
        # 音声全体を読み込み
        audio = whisper.load_audio(audio_path)
        
        # 最初に全体をセグメントに分割（言語検出なし）
        print("音声をセグメントに分割中...")
        initial_result = model.transcribe(
            audio_path,
            language=None,  # 自動検出
            verbose=False,
            word_timestamps=False,
            condition_on_previous_text=False,
            task='transcribe'
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
                    line = f"[{format_time(start_time)} -> {format_time(end_time)}]{amb_tag}{weight_tag} [JA:{ja_percent:03d}%] [RU:{ru_percent:03d}%] {text}"
                    output_lines.append(line)
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
    parser.add_argument('--output', '-o', help='出力テキストファイルのパス（指定しない場合は標準出力）')
    parser.add_argument('--ja-weight', type=float, default=1.0, help='日本語確率へ乗算する重み (デフォルト1.0)')
    parser.add_argument('--ru-weight', type=float, default=1.0, help='ロシア語確率へ乗算する重み (デフォルト1.0)')
    parser.add_argument('--ambiguous-threshold', type=float, default=10.0, help='あいまい再トライを行う確率差しきい値(%)')
    parser.add_argument('--vad-level', type=int, default=2, choices=[0,1,2,3], help='VAD 厳しさ (0=寛容,3=厳格)')
    parser.add_argument('--include-silent', action='store_true', help='スキップされた無音/短すぎ/空テキストもプレースホルダ表示')
    parser.add_argument('--debug-segments', action='store_true', help='デバッグ: 各セグメントの判定理由を表示')
    parser.add_argument('--gap-threshold', type=float, default=0.35, help='ギャップ挿入の下限秒数 (include-silent時)')
    
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
        gap_threshold=args.gap_threshold
    )
    
    # 結果を出力
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result, encoding='utf-8')
        print(f"\n結果を {args.output} に保存しました")
    else:
        print("\n=== 文字起こし結果 ===")
        print(result)

if __name__ == "__main__":
    main()
