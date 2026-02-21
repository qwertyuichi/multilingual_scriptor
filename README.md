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
- Python 3.11+ 推奨 (標準 `tomllib` 利用)
- FFmpeg (PATH 通し済み)
- (任意) CUDA 対応 GPU または AMD ROCm 対応 GPU

### 3.2 依存パッケージ
`requirements.txt`:
```
openai-whisper
torch
numpy
scipy
webrtcvad
PySide6
```

インストール例 (Windows PowerShell):
```pwsh
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

FFmpeg 確認:
```pwsh
ffmpeg -version
```

### 3.3 AMD ROCm GPU を使用する場合

`faster-whisper` の推論エンジンである **CTranslate2** は、AMD ROCm GPU 向けの公式 PyPI パッケージを提供していません。`pip install ctranslate2` でインストールされる標準ホイールは ROCm に対応していないため、**ROCm 専用ホイール**を別途入手してインストールする必要があります。

> **CTranslate2 公式より:**
> *"If you have an AMD ROCm GPU, we provide specific Python wheels on the [releases page](https://github.com/OpenNMT/CTranslate2/releases/)."*

**インストール手順:**

1. [CTranslate2 Releases](https://github.com/OpenNMT/CTranslate2/releases/) から自身の環境に合ったホイール (Python バージョン・OS・ROCm バージョン) をダウンロードする。
2. 通常の `pip install -r requirements.txt` でその他の依存関係を先にインストールする。
3. `ctranslate2` だけ手動でホイールを指定してインストールする:

```pwsh
pip install ctranslate2-X.Y.Z-cpXXX-cpXXX-*.whl --force-reinstall
```

> **注意:** ROCm 対応ホイールを `--force-reinstall` で上書きした後、`pip install faster-whisper` が `ctranslate2` を再度上書きしないよう注意してください。必要に応じて `faster-whisper` インストール後にホイールを再適用してください。

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
| JSON | `.json` | 全セグメントを 1 つの JSON オブジェクトに集約。`{"segments": [...], "metadata": {...}}` のような形で、各セグメント要素は `start,end,text,text_ja,text_ru,ja_prob,ru_prob,chosen_language` 等を保持 |

`exporter.py` が共通ロジック。GUI は内部結果 (`transcription_result['segments']`) をそのまま利用し JSON ではまとめて 1 ファイルに格納します。

---
## 8. セグメントデータ構造 (概念)
```jsonc
{
  "start": 12.345,      // 秒
  "end": 15.678,
  "text": "表示用(選択言語)",
  "text_ja": "日本語候補",
  "text_ru": "ロシア語候補",
  "ja_prob": 72.13,     // %
  "ru_prob": 24.91,
  "chosen_language": "ja" | "ru" | null,
  // gap フラグは廃止済み
}
```

---
## 9. 設定 (`config.toml`)
`[default]` 例:
```toml
[default]
device = "cuda"            # cpu / cuda
transcription_model = "large-v3"
segmentation_model = "turbo"
default_languages = ["ja", "ru"]
ja_weight = 1.0
ru_weight = 1.0
min_seg_dur = 0.60
vad_level = 2
# gap_threshold = 0.5 (旧: 無音→GAP 生成用。現在は内部タイミング調整用途のみ/または未使用)
ambiguous_threshold = 30.0
include_silent = true
output_format = "json"     # txt / srt / json
initial_prompt = ""
```
別プリセット `[kapra]` などを追加して GUI で切替可 (実装状況に依存)。

---
## 10. アーキテクチャ概要
| モジュール | 役割 |
|------------|------|
| `main.py` | GUI 本体 (再生/編集/スレッド管理) |
| `transcriber.py` | Whisper + VAD + 部分再文字起こしスレッド (`TranscriptionThread` / `RangeTranscriptionThread`) |
| `models/segment.py` | `Segment` データクラス & リスト操作補助 |
| `services/segment_ops.py` | 分割・境界調整など純粋操作 |
| `services/retranscribe_ops.py` | 動的時間分割・連続結合等の高レベル再処理 |
| `ui/table_presenter.py` | テーブル構築・集約テキスト再構築 |
| `exporter.py` | TXT / SRT 文字列生成 & JSON ペイロード |
| `utils/timefmt.py` | 時刻フォーマットユーティリティ |
| `utils/segment_utils.py` | 表示テキスト整形 / ID 正規化 |

スレッド完了シグナル → GUI スロットで `segments` 更新 → `table_presenter` 経由で再描画、の流れ。

---
## 11. トラブルシュート
| 症状 | 主原因候補 | 対処 |
|------|------------|------|
| `No module named 'pkg_resources'` | venv 内の `webrtcvad.py` が古い | [venv/Lib/site-packages/webrtcvad.py](venv/Lib/site-packages/webrtcvad.py) の `import pkg_resources` を `import importlib.metadata` に、`pkg_resources.get_distribution(...).version` を `importlib.metadata.version(...)` に書き換える |
| ffmpeg が見つからない | PATH 未設定 | FFmpeg インストール & パス確認 (`ffmpeg -version`) |
| GPU が使われない | CUDA ドライバ不足 / CPU ビルド | `torch.cuda.is_available()` 確認 / ドライバ更新 |
| 途中でフリーズ感 | 長時間モデル推論 | 進捗バー運用 / 小さいモデルへ変更 |
| 再分割が無効 | 再生位置が端 / 最小長未満 | 再生位置を中央付近に調整 |
| SRT 文字化け | エディタの文字コード | UTF-8 (BOM 無) で開く |
| JSON が空 (segments が 0) | セグメントが GAP のみ / 全て除外 | GAP 変換し過ぎていないか確認 |

---
### 11.1 詳細診断ガイド (ログ/再トランスクリプト監視)

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
12:34:56 [DEBUG] main: [DIFF][detail] row 12: end:14.32->14.56, text_ja:旧->新
```
意味:
- `+N` 追加行数 / `-N` 削除行数
- `modified=M` 変更行 (最大 10 行まで詳細)
- `row i:` start / end / text_ja / text_ru / lang の差分一覧

#### (C) 分割/結合再文字起こし監視 (ウォッチドッグ)
`[再解析中]` プレースホルダが既定 (15 秒) 超えて残る場合は内部ウォッチドッグがチェックします。
DEBUG ログで以下を追跡:
1. プレースホルダ化された行数
2. 対応 RangeTranscriptionThread の開始/完了 (今後詳細ログ追加予定)
3. 完了後 `[DIFF][rebuild]` が出力され差分が反映されるか

#### (D) 代表的な調査シナリオ
| 症状 | チェック | 次アクション |
|------|----------|--------------|
| プレースホルダ残留 | `[DIFF][rebuild]` 無 | スレッド未完了/例外。例外ログ確認 |
| 行数異常 | `+N -M` 値 | 分割/結合ロジック再確認 |
| 時刻ズレ | detail 内 start/end 変化 | 丸め/境界調整計算確認 |
| 言語ラベル不更新 | lang 差分欠落 | 言語確率再計算未実施 (transcriber範囲) |

#### (E) 運用戻し (ログ抑制)
1. `[logging].level = "INFO"`
2. `[debug]` セクション削除または `rebuild_diff = false`
3. 既存ログはローテーション設定で整理

#### (F) 既知制限
- 差分詳細は最大 10 行。大量一括更新はサマリのみ。
- 行番号ベース比較のため再並び替えが起きた場合、差分解釈が困難になる可能性 (将来 ID 比較検討)。

---
## 12. パフォーマンス & 改善余地
- smaller モデル選択 / GPU 利用
- faster-whisper (CTranslate2) 置換検討
- 連続複数セグメントの一括再トランスクリプト最適化
- 事前 VAD マスク適用で無音区間スキップ向上

---
## 13. ライセンス / 出典
- OpenAI Whisper (MIT) https://github.com/openai/whisper
- 本ツール: ライセンス未記載（必要なら `LICENSE` 追加推奨: MIT など）

---
## 14. 開発メモ / 今後のアイデア
- SRT 行長自動折返し最適化
- Waveform 可視化オーバーレイ
- 英語追加 / 多言語 UI
- モデルキャッシュ管理 UI
- バッチ一括処理 (フォルダドロップ)

---
## 15. 簡易チェックリスト
- [ ] FFmpeg 動作確認
- [ ] `pip install -r requirements.txt`
- [ ] `python main.py` 起動
- [ ] 文字起こし完了後 テーブル表示
- [ ] 逐次追加（セグメントが徐々に増えていく）
- [ ] 分割 / 境界調整 / 削除 が機能
- [ ] 書き出し成功 (TXT / SRT / JSON)

---
## 16. 逐次更新 / モデル事前ロード / キャンセル挙動

### 16.1 逐次更新
従来は全文完了後にまとめてテーブルへ表示していましたが、現在は Whisper 推論で 1 セグメント確定するたびに即座にテーブル最下行へ追加し、ステータスバーに件数 `(N)` を表示します。これにより長時間動画でも途中経過を確認しながら編集方針を検討できます。

### 16.2 モデル事前ロード (ウォームアップ)
全文文字起こし開始直後に以下をキャッシュへロードします。
1. セグメンテーション用モデル (`segmentation_model_size`)
2. メイン文字起こしモデル (`model`)

部分再文字起こし (範囲選択 → 再文字起こし) は同じモデルインスタンスを再利用するため、初回のロード遅延がほぼ解消されます。

### 16.3 キャンセル挙動
キャンセル要求後は:
- 進行中スレッドへキャンセルフラグ送信
- 既に表示済みの途中結果は即座にクリア (テーブル行数 0 / 内部 `segments` 初期化)
- 以降遅延して届く逐次セグメントシグナルは無視されます

これにより「一度キャンセルしたのに途中結果が残っていた」混乱を防ぎます。部分結果を保持したい場合はキャンセルではなく一時停止や再生操作で確認してください。

### 16.4 既知の注意
- 大量行(数千) の逐次追加でスクロールが頻繁に動く場合、将来的にバッチ描画最適化を追加予定です。
- キャンセル直後に別動画を開いた際、まれに旧スレッドから遅延シグナルが届いても無視されるようガードしています。

---

---
ご要望・改善案があれば Issue / コメントなどでお知らせください。
