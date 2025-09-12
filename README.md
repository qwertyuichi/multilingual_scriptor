# 音声認識（日本語 / ロシア語 セグメント再トランスクリプト）

`transcription_for_kapra.py` は動画 (mp4 等) から音声を抽出し、OpenAI Whisper ローカルモデルを用いて
1回目に全体を粗く文字起こし → 各セグメントを個別に再言語判定・再トランスクリプトする 2段階方式で、
日本語とロシア語を主対象に確率ラベル付きの結果を生成します。

---
## 特徴
- FFmpeg で動画から 16kHz モノラル PCM 抽出
- 初回 `model.transcribe` により粗いセグメント列を取得
- 各セグメントを再度 切り出し & 言語(ja/ru)確率推定 (Whisper の `detect_language` を軽量ラップ)
- 確率差が小さい(あいまい)場合、日/露 両言語で再トランスクリプト → `avg_logprob` 比較で良い方採用 (AMB 表示)
- 日本語 / ロシア語確率に任意の重みを乗算しバイアス調整 (`--ja-weight`, `--ru-weight`) → 加重時は `[W]` タグ
- VAD (webrtcvad) により無音/ノイズ区間をスキップ (閾値調整 `--vad-level`)
- スキップや空結果・ギャップを可視化するプレースホルダ出力 `--include-silent`
- デバッグ用 詳細ログ表示 (`--debug-segments`) で分割・言語判定・再トライ過程を追跡
- ハルシネーション疑いの長い反復文字列に `[HALLUCINATION?]` タグ付与
- 出力形式: `[開始 -> 終了][オプションタグ] [JA:xxx%] [RU:yyy%] テキスト`
- 失敗時は初期セグメントの粗テキストへフォールバック

---
## 依存関係
`requirements.txt` (主要):
```
openai-whisper
torch
numpy
scipy
ffmpeg-python
webrtcvad
```
追加システム要件:  
- FFmpeg 実行バイナリ (パスが通っていること)  
- GPU (任意) CUDA があれば高速化 / CPU のみでも可

PowerShell (Windows) での FFmpeg 確認例:
```pwsh
ffmpeg -version
```

FFmpeg 未導入なら (例: winget):
```pwsh
winget install --id=Gyan.FFmpeg  -e --source winget
```

Python パッケージのインストール:
```pwsh
pip install -r requirements.txt
```

---
## 使い方
基本:
```pwsh
python transcription_for_kapra.py "video\サンプル動画.mp4" --model large-v3
```

出力をファイル保存:
```pwsh
python transcription_for_kapra.py "video\サンプル動画.mp4" --model large-v3 -o result.txt
```

### 引数一覧
| 引数 | 必須 | 説明 | 例 |
|------|------|------|----|
| `video_path` | はい | 入力動画ファイルパス | `video\example.mp4` |
| `--model` | いいえ | Whisper モデルサイズ (`tiny`…`large-v3`) | `--model small` |
| `--output`, `-o` | いいえ | 出力テキストファイル | `-o out.txt` |
| `--ja-weight` | いいえ | 日本語確率へ乗算する重み (初期 1.0) | `--ja-weight 1.2` |
| `--ru-weight` | いいえ | ロシア語確率へ乗算する重み (初期 1.0) | `--ru-weight 1.8` |
| `--ambiguous-threshold` | いいえ | JA/RU 確率差 (%) がこの値未満なら両言語で再トライ | `--ambiguous-threshold 12` |
| `--vad-level` | いいえ | VAD 厳しさ 0(寛容)〜3(厳格) | `--vad-level 1` |
| `--include-silent` | いいえ | スキップ/ギャップ/空結果プレースホルダを出力 | `--include-silent` |
| `--debug-segments` | いいえ | 初期分割と判定過程の詳細ログ | `--debug-segments` |
| `--gap-threshold` | いいえ | ギャップ表示する最小秒数 (`--include-silent` 有効時) | `--gap-threshold 0.25` |



---
## ディレクトリ構成（例）
```
音声認識/
  transcription_for_kapra.py
  requirements.txt
  README.md (本ファイル)
  video/
    sample.mp4
  temp_segments/        # 任意: 中間検証用 (現スクリプトは直接ここを使っていません)
  temp_langprob/        # 任意: 言語判定用に保存する場合の例 (現スクリプトは直接ここを使っていません)
  old/
    whisper_test.py
```

---
## 処理フロー (Mermaid 概要)
```mermaid
flowchart TD
  A[開始] --> B[引数解析]
  B --> C{動画ファイル存在?}
  C -- いいえ --> E[エラーメッセージ表示 終了]
  C -- はい --> F[Whisperモデル読み込み]
  F --> G[FFmpegで音声抽出(16kHz mono)]
  G --> H[初回 transcribe (自動言語)]
  H --> I[segments ループ]
  I --> J{次 segment?}
  J -- なし --> O[出力結合]
  J -- あり --> K[開始/終了→サンプル計算]
  K --> L[長さ/範囲チェック]
  L -- 短い --> I
  L -- OK --> M[言語判定 (ja/ru 確率)]
  M --> M2{確率差 < 閾値?}
  M2 -- はい --> M3[ja/ru 両言語再トライ + avg_logprob比較]
  M2 -- いいえ --> N[選択言語で再 transcribe]
  M3 --> P{結果空?}
  N --> P{結果空?}
  P -- はい --> I
  P -- いいえ --> Q[確率/時刻フォーマット行追加]
  Q --> I
  O --> R{--output 指定?}
  R -- はい --> S[ファイル保存]
  R -- いいえ --> T[標準出力]
  S --> U[終了]
  T --> U[終了]
```

---
## 出力タグ一覧
| タグ | 意味 |
|------|------|
| `[W]` | 言語確率に重み付け (ja/ru weight ≠ 1.0) 適用済み |
| `AMB` | JA/RU 確率差が閾値未満 → 両言語再トライし良い方採用 |
| `[SKIP:silence vad=X]` | VAD で無音判定され処理スキップ (X=VAD level) |
| `[SKIP:too_short]` | 0.1 秒未満でスキップ |
| `[EMPTY]` | 再トランスクリプト結果が空 |
| `[GAP:x.xx s]` | 前後セグメント間に x.xx 秒以上の空白区間 |
| `[HALLUCINATION?]` | 長い反復文字列を検出し疑わしいと判断 |

---
## デバッグ・診断手順例
1. 欠落したと感じる時間帯がある → `--include-silent --debug-segments` を付け再実行
2. `[DEBUG] 初期セグメント一覧` を確認し該当時間帯の行が存在するか判定
3. 存在しない: Whisper 初期分割で生成されていない → モデル/temperature/VAD 無効化検証
4. 存在するが `[SKIP:*]` 表示: スキップ理由 (silence / too_short) を調整 (`--vad-level` 変更)
5. `[EMPTY]` のみ: 音量極小や語彙外 → 同区間重ね録り / モデルサイズ変更
6. 改善後差異を比較するには出力テキスト差分を取る (PowerShell: `Compare-Object` 等)

### 例: ロシア語判定を強めつつデバッグ
```pwsh
python transcription_for_kapra.py "video\sample.mp4" --model large-v3 \
  --ru-weight 1.8 --ambiguous-threshold 15 --vad-level 1 \
  --include-silent --debug-segments --gap-threshold 0.25
```

---
## 言語確率の重み付けロジック
- Whisper の生確率 (ja_prob, ru_prob) を 0-1 に戻しそれぞれ指定重みを乗算 → 正規化
- 例: JA=0.6, RU=0.4, `--ru-weight 1.5` → JA=0.6, RU=0.4*1.5=0.6 → 正規化後 JA=50%, RU=50%
- 重みは「最終判定前のバイアス」なので AMB 判定にも効く (差が縮まりやすく再トライ誘発する場合あり)
- 過度に重みを上げると他方が常に 0% 近くになり AMB が起きにくくなるため注意

---
## コード改善提案（任意）
| 項目 | 現状 | 改善案 | 期待効果 |
|------|------|--------|----------|
| `n_mels` 動的化 | 実装済 | 80/128 自動検出 | モデル差異吸収 |
| 可変長パディング | 実装済 | 2〜30秒のみ調整 | 過剰 pad 回避 |
| VAD スキップ | 実装済 | 閾値調整/無効化対応 | ノイズ削減 |
| AMB 再トライ | 実装済 | avg_logprob 比較 | あいまい改善 |
| 言語重み付け | 実装済 | ja/ru weight | バイアス制御 |
| ハルシネ検出 | 実装済 | 反復文字縮約 | ノイズ抑止 |
| プレースホルダ | 実装済 | GAP / SKIP / EMPTY | 欠落原因可視化 |
| 並列化 | 未 | マルチプロセス | 速度向上 |
| faster-whisper | 未 | CTranslate2 | 高速化 |
| 多言語汎化 | 部分 | 他言語閾値導入 | 誤検出低減 |
| JSON/SRT 出力 | 未 | フォーマット追加 | 連携容易 |


### `n_mels` を標準へ戻す例
`detect_language_probabilities` 内:
```python
# 変更前
mel = whisper.log_mel_spectrogram(audio_segment, n_mels=128).to(model.device)
# 変更後
mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
```

---
## よくあるエラーと対処
| 症状 | 原因候補 | 対処 |
|------|----------|------|
| `FileNotFoundError: ffmpeg` | FFmpeg 未導入 or PATH 未設定 | FFmpeg インストールし PATH 追加 |
| CUDA 関連警告 | GPU 未検出 / ドライバ不整合 | CPU で続行可、必要ならドライバ更新 |
| メモリ不足 | large 系モデル使用 | smaller モデル or GPU / 量子化導入 |
| 言語が逆判定 | 短区間 / ノイズ | セグメント長調整 / 閾値導入 |

---
## パフォーマンスチューニング簡易メモ
- モデルサイズ縮小: `small` / `medium` で速度向上
- `temperature` を低く維持 → 安定性優先 (既に 0.2)
- faster-whisper (CTranslate2) へ移行し `compute_type="int8_float16"` などを検討
- セグメント再トランスクリプトを条件付き（確率差が小さい時のみ など）にする

---
## ライセンス / 出典
- OpenAI Whisper: MIT License (https://github.com/openai/whisper)
- 本スクリプト: （未設定。必要に応じて MIT などを追記してください）

---
## チェックリスト（運用）
- [ ] FFmpeg が動作する
- [ ] Python 依存関係インストール済
- [ ] GPU (任意) が利用可能か `torch.cuda.is_available()` で確認
- [ ] `--model` を適切に選定
- [ ] 出力文字コード UTF-8 (外部連携時に確認)

---
## 変更履歴 (例)
| 日付 | 内容 |
|------|------|
| 2025-09-12 | 初回 README 作成 |
| 2025-09-12 | VAD / AMB / 重み付け / デバッグオプション / タグ説明追記 |

---
## 今後の発展アイデア
- Web UI (Gradio / Streamlit) 化
- 多言語拡張 (英語/韓国語等) と自動翻訳パイプライン
- 信頼度指標 (平均 logprob) 付与
- JSON / SRT / VTT など字幕フォーマット出力
- バッチ処理ディレクトリ対応

---
ご不明点や追加したい項目があればお知らせください。
