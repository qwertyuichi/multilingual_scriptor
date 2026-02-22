<br>

# Multilingual Scriptor

**多言語対応動画文字起こし & セグメント編集 GUI**

Whisper ベースで動画を文字起こしし、GUI 上で「再生」「シーク」「セグメント編集 (分割/結合/境界調整)」「部分再トランスクリプト」「書き出し (TXT / SRT / JSON)」を行うデスクトップツールです。複数言語の確率を保持しながら編集でき、日本語・ロシア語をはじめとした多言語コンテンツに対応します。

---
## 1. 主な特徴
| 分類 | 機能概要 |
|------|----------|
| 初回文字起こし | Whisper 指定モデルで全編を一括処理 (日露混在可) |
| 動画再生連動 | 再生位置に合わせてテーブル選択が自動追従 (逆シークも可) |
| セグメント編集 | テキスト編集 / カーソル位置で 2 分割 / 動的分割 / 2 行境界のドラッグ的調整 (時間位置で再計算) |
| 逐次反映 | セグメント確定ごとにテーブルへ即時追加 (進捗件数表示) |
| モデル最適化 | 初回全文開始時に部分再文字起こし用モデルを事前ロード (ウォームアップ) |
| 再トランスクリプト | 分割直後 / 境界調整直後 / 指定範囲(行)などで部分再実行 (バックグラウンド Thread) |
| 言語確率 | JA / RU 推定確率を保持し優勢言語表示 / 手動で表示言語選択も可能 |
| 書き出し | TXT / SRT / JSON |
| コンフィグ | `config.toml` でモデル/デバイス/閾値等プリセット切替 |
| UI 操作性 | ダブルクリックで該当開始秒へ再生 / 選択行が 1 行と 2 行で挙動切替 |

---
## 2. 画面概要
左: 動画 + 再生コントロール / 右: セグメントテーブル & 編集操作パネル / 下部: ステータス & 進捗バー。テーブル列は概ね `START | END | LANG | JA_TEXT | RU_TEXT | JA% | RU% | FLAGS` のイメージ (実際の内部キーは `segments` リストに格納)。

---
## 3. インストール
### 3.1 前提
- Python 3.11+ 
- FFmpeg
- **GPU 使用時（推奨）:** CUDA または ROCm が必要です。詳細は [CTranslate2 GitHub](https://github.com/OpenNMT/CTranslate2) を参照してください。

> **注意:** CPU のみでも動作しますが、文字起こし速度が大幅に低下します。GPU環境を強く推奨します。

### 3.2 依存パッケージ
`requirements.txt`:
```
faster-whisper
PySide6
```

**注**: `faster-whisper` は内部で CTranslate2、Silero VAD、およびその他の依存パッケージ（av, onnxruntime, tokenizers, tqdm など）を使用しています。これらは `faster-whisper` のインストール時に自動的にインストールされます。

インストール例 (Windows PowerShell):
```pwsh
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### FFmpeg のインストール

**Windows (winget 使用):**
```pwsh
winget install ffmpeg
```

インストール後、新しいターミナルを開いて確認:
```pwsh
ffmpeg -version
```

> **注意**: winget でインストールした場合、自動的に PATH が設定されます。PATH が通らない場合は、一度ターミナルを再起動してください。

**その他の方法:**
- [FFmpeg 公式サイト](https://ffmpeg.org/download.html) からダウンロードして手動でインストール
- Chocolatey: `choco install ffmpeg`
- Scoop: `scoop install ffmpeg`

### 3.3 GPU環境のセットアップ

GPU を使用する場合、CUDA または ROCm が必要です。詳細なセットアップ手順については、[CTranslate2 GitHub](https://github.com/OpenNMT/CTranslate2) を参照してください。

---
## 4. 起動
```pwsh
python main.py
```
初回起動時に `config.toml` の `[default]` セクション値を GUI へ反映します。別プリセット (例: `[kapra]`) へ切替する UI がある場合はそちらから再読込できます。

---
## 5. 基本操作フロー
1. 右上/上部の「動画を開く」ボタンで MP4 等を選択
2. 「文字起こし開始」を押下 → モデル読み込み & 全体セグメント生成
3. 再生しながら内容確認（テーブル行クリックでシーク）
4. 必要に応じて編集: 
   - 1 行選択 + ボタン: 現在再生位置で動的分割 → 分割後 2 区間を自動で再トランスクリプト
   - 2 行連続選択 + ボタン: 両行境界を現在再生位置へ移動 → 双方を再トランスクリプト
   - 行ダブルクリック: 詳細編集ダイアログ (JA/RU テキスト編集 / カーソル分割)
  - 不要行選択 → 削除: セグメントを物理削除
5. エクスポート: 書き出し形式 (TXT / SRT / JSON) を選んで保存

---
## 6. セグメント編集詳細
| 操作 | 条件 | 動作 |
|------|------|------|
| 行ダブルクリック | 任意行 | 編集ダイアログ (JA/RU 切替, テキスト修正, カーソル分割) |
| 1 行選択 + 分割ボタン | 任意行 | 現在再生位置で 2 分割 (閾値未満の極小断片は拒否) |
| 2 行連続選択 + 境界調整 | 任意行 | 境界を現在再生位置へ移動 (最小長確保) |
| 複数行選択 + 削除 | 任意行 | セグメント削除 (番号再採番) |
| カーソル分割 (編集ダイアログ) | テキスト内カーソル位置 | 言語別テキスト長比で開始/終了時間を線形割当 |

再トランスクリプトは時間枠確定後に順次スレッドで実行され、完了ごとにテーブル更新。実行中は関連ボタンが無効化され進捗バーが進む。

---
## 7. 書き出し形式
| 形式 | 拡張子 | 内容 |
|------|--------|------|
| TXT | `.txt` | `[HH:MM:SS.mmm -> HH:MM:SS.mmm] [JA:xx.xx%] [RU:yy.yy%] JA=... | RU=...` 行列 |
| SRT | `.srt` | 標準 SRT。優勢 (確率が高い) 言語テキストを採用。連番 + 時刻範囲 + 本文 |
| JSON | `.json` | 全セグメントを 1 つの JSON オブジェクトに集約。`{"segments": [...], "metadata": {...}}` のような形で、各セグメント要素は `start,end,text,text_lang1,text_lang2,lang1_prob,lang2_prob,lang1_code,lang2_code,chosen_language` 等を保持 |

---
## 8. 設定 (`config.toml`)
`[default]` 例:
```toml
[default]
device = "cuda"                    # cpu / cuda
model = "large-v3"                 # large-v3 / distil-large-v3
default_languages = ["ja", "ru"]

lang1_weight = 0.50
lang2_weight = 0.50

no_speech_threshold = 0.6          # 無音スキップ感度
initial_prompt = ""                # 認識ヒント文

# VAD (音声区間の自動検出) - Silero VAD を使用
vad_filter = true
vad_threshold = 0.5
vad_min_speech_ms = 250
vad_min_silence_ms = 2000
```
別プリセット `[kapra]` などを追加して GUI で切替可。

---
## 9. ⚠️ 言語組み合わせの注意事項

言語選択は99言語に対応していますが、**組み合わせによって精度が大幅に下がる**ケースがあります。

### 9.1 スクリプト共有ペア

本ツールはスクリプト文字そのものによる自動確率補正を行いません。選択した両言語が**同じスクリプト**を使う場合は、文字情報による区別ができないため、音響特徴や語彙に依存した判定になります。

| 言語組み合わせ | 共有スクリプト | 備考 |
|---|---|---|
| JA + ZH (日中) | CJK文字（漢字）| ひらがな・カタカナが出現すればJA確定。漢字のみのテキストは区別困難 |
| RU + UK / BG / SR / MK / BE (露+キリル圏) | キリル文字 | 全言語がキリル文字使用。文字情報での区別は困難 |
| AR + FA / UR / PS / SD (アラビア語+アラビア文字圏) | アラビア文字 | 類似文字体系で文字情報による区別は困難 |
| HE + YI (ヘブライ語+イディッシュ) | ヘブライ文字 | 文字情報での区別は困難 |
| ZH + YUE (中国語+広東語) | CJK文字 | ほぼ同一スクリプト。音響特徴のみで判定 |

**対策:** 上記の組み合わせでは `lang1_weight`/`lang2_weight` を手動調整するか、優勢言語を単独指定してください。

### 9.2 ラテン文字ペア (スクリプト検出不可)

EN, FR, DE, ES, PT, IT, NL, PL, CS, RO, など多くのヨーロッパ言語はラテン文字を使用します。これらを組み合わせると**スクリプト検出が一切働かず**、確率推定のみで判定します。

| 例 | 備考 |
|---|---|
| EN + FR | どちらもラテン文字。語彙・音響特徴のみで判定 |
| DE + NL | 非常に近縁。精度低下の可能性大 |
| ES + PT | 類似語彙多数。短文では混乱しやすい |

**対策:** 近縁言語の組み合わせは避け、単独指定を推奨します。どちらかを `なし` に設定してください。

### 9.3 推奨される組み合わせ

スクリプトが明確に異なる言語ペアは高精度で動作します:

| ペア | 理由 |
|---|---|
| JA + RU (デフォルト) | ひらがな/カタカナ vs キリル文字で明確に区別 |
| JA + EN | CJK/かな vs ラテン文字 |
| ZH + EN | CJK vs ラテン文字 |
| AR + EN | アラビア文字 vs ラテン文字 |
| KO + EN | ハングル vs ラテン文字 |
| TH + EN | タイ文字 vs ラテン文字 |

---
## 10. トラブルシュート
| 症状 | 主原因候補 | 対処 |
|------|------------|------|
| `ctranslate2.dll` が読み込めない | ROCm/CUDA ランタイム未インストール | 下記「11.1 CTranslate2 DLL依存関係エラー」参照 |
| ffmpeg が見つからない | PATH 未設定 | FFmpeg インストール & パス確認 (`ffmpeg -version`) |
| GPU が使われない | CUDA/ROCm ドライバ不足 | ドライバ更新、GPU対応確認 (`nvidia-smi` / `rocm-smi`) |
| 途中でフリーズ感 | 長時間モデル推論 | 進捗バー運用 / 小さいモデルへ変更 |
| 再分割が無効 | 再生位置が端 / 最小長未満 | 再生位置を中央付近に調整 |
| SRT 文字化け | エディタの文字コード | UTF-8 (BOM 無) で開く |
| JSON が空 (segments が 0) | セグメントが GAP のみ / 全て除外 | GAP 変換し過ぎていないか確認 |

---
### 11.1 CTranslate2 DLL依存関係エラー

**エラー例:**
```
FileNotFoundError: Could not find module 'ctranslate2.dll' (or one of its dependencies)
```

このエラーは `ctranslate2.dll` 自体は存在するが、**その依存DLLが見つからない**場合に発生します。

#### 原因と対処法

**AMD ROCm GPU を使用している場合（最も一般的）**

ROCm版のCTranslate2は、AMD ROCmランタイムライブラリに依存します：
- `amdhip64.dll`
- `rocblas.dll`
- その他のROCmコンポーネント

**対処法:**
1. [AMD ROCm SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) をインストール
2. システムを**再起動**（重要）
3. 環境変数 `PATH` に ROCm の bin ディレクトリが追加されていることを確認

**NVIDIA CUDA GPU を使用している場合**

CUDA版のCTranslate2は、CUDAランタイムライブラリに依存します。

**対処法:**
1. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) をインストール
2. システムを再起動

**その他の依存関係**

Visual C++ Redistributables が不足している場合もあります。

**対処法:**
```pwsh
winget install Microsoft.VCRedist.2015+.x64
```

#### 診断方法

不足している具体的なDLLを確認するには：

```python
import ctypes
import os

dll_path = r"D:\multilingual_scriptor\venv\Lib\site-packages\ctranslate2\ctranslate2.dll"
os.add_dll_directory(os.path.dirname(dll_path))

try:
    ctypes.CDLL(dll_path)
    print("✓ DLL読み込み成功")
except OSError as e:
    print(f"✗ エラー: {e}")
```

---
### 11.2 詳細診断ガイド (ログ/再トランスクリプト監視)

恒久修正フェーズで導入されたロギング & デバッグ支援機能の使い方です。

#### (A) ロギング有効化
`config.toml` に任意で `[logging]` セクションを追加:
```toml
[logging]
level = "DEBUG"          # INFO 以上を推奨。差分調査時のみ DEBUG
file_enabled = true       # ファイル出力を有効化
file_path = "app.log"     # 出力ファイルパス
max_bytes = 1048576       # ローテーション閾値
backup_count = 3          # 世代数
```

#### (B) セグメント再構築差分 (rebuild diff) 追跡
`config.toml` に `[debug]` セクションを追加し `rebuild_diff = true` を設定すると、
テーブル再描画時のセグメント配列差分が DEBUG ログに出力されます。
```toml
[debug]
rebuild_diff = true
```
出力例:
```
12:34:56 [DEBUG] main: [DIFF][rebuild] +0 -0 modified=1
12:34:56 [DEBUG] main: [DIFF][detail] row 12: end:14.32->14.56, text_lang1:旧->新
```
意味:
- `+N` 追加行数 / `-N` 削除行数
- `modified=M` 変更行 (最大 10 行まで詳細)
- `row i:` start / end / text_lang1 / text_lang2 / lang の差分一覧

#### (C) 分割/結合再文字起こし監視 (ウォッチドッグ)
`[再解析中]` プレースホルダが既定 (15 秒) 超えて残る場合は内部ウォッチドッグがチェックします。

#### (D) 運用戻し (ログ抑制)
1. `[logging].level = "INFO"`
2. `[debug]` セクション削除または `rebuild_diff = false`
3. 既存ログはローテーション設定で整理

---
## 11. パフォーマンス
小規模モデル選択または GPU 利用で処理速度が大幅に向上します。

---
## 12. ライセンス
- faster-whisper (MIT) https://github.com/SYSTRAN/faster-whisper
- 本ツール: ライセンス未記載（必要なら `LICENSE` 追加推奨: MIT など）

---
## 13. 開発について

このプロジェクトの内部アーキテクチャ、セグメントデータ構造、モジュール詳細については [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

---

ご要望・改善案があれば Issue / コメントなどでお知らせください。
