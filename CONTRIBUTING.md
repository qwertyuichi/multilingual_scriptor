# 開発ガイド (CONTRIBUTING)

本ドキュメントは、Multilingual Scriptor の内部アーキテクチャ、セグメントデータ構造、モジュール構成、および開発時の参考情報を提供します。

---

## 1. セグメントデータ構造

### 1.1 概念
各セグメント（文字起こしの最小単位）は以下データセットをもちます：

```jsonc
{
  "start": 12.345,              // 秒単位の開始時刻
  "end": 15.678,                // 秒単位の終了時刻
  "text": "表示用(選択言語)",     // GUI / 書き出し時の表示テキスト
  "text_lang1": "言語1候補",     // 言語1テキスト（Whisper認識結果）
  "text_lang2": "言語2候補",     // 言語2テキスト（Whisper認識結果）
  "lang1_prob": 72.13,          // 言語1確率 (%)
  "lang2_prob": 24.91,          // 言語2確率 (%)
  "lang1_code": "ja",           // 言語1のISOコード
  "lang2_code": "ru",           // 言語2のISOコード (null = 1言語モード)
  "chosen_language": "ja"       // 優勢言語 ("ja" | "ru" | "other" | null)
}
```

### 1.2 内部表現
Python 内部では `models/segment.py` の `Segment` クラスで統一：

```python
@dataclass
class Segment:
    start: float
    end: float
    text: str
    text_lang1: str
    text_lang2: str
    lang1_prob: float
    lang2_prob: float
    lang1_code: str = "ja"
    lang2_code: str = "ru"
    chosen_language: Optional[str] = None
    
    def to_dict(self) -> dict:
        """辞書形式へ変換"""
        ...
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Segment':
        """辞書からインスタンス化"""
        ...
```

### 1.3 表示テキスト決定ロジック
`text` フィールドは `utils/segment_utils.py` の `display_text()` で決定：

- `chosen_language` が lang1_code と一致 → `text_lang1` を採用
- `chosen_language` が lang2_code と一致 → `text_lang2` を採用
- `chosen_language` が "other" / 不明 → `text` (気険変換) を利用
- 両方が空 → `"[ギャップ]"`

このロジックにより、セグメント編集時に選択言語に応じた表示が自動更新されます。

### 1.4 言語確率の計算
`transcription/processor.py` で Whisper の多言語確率から算出：

```python
lang1_prob = logits[lang1] / (logits[lang1] + logits.get(lang2, 0)) * 100
lang2_prob = logits.get(lang2, 0) / (logits[lang1] + logits.get(lang2, 0)) * 100
```

---

## 2. アーキテクチャ概要

### 2.1 モジュール構成

| モジュール | ファイル | 役割 |
|------------|---------|------|
| **UI (メイン)** | `ui/app.py` | GUI 本体。再生・編集・スレッド管理 |
| **テーブル表示** | `ui/table_presenter.py` | テーブル更新・集約テキスト再構築 |
| **編集ダイアログ** | `ui/edit_dialog.py` | セグメントのテキスト編集・分割ダイアログ |
| **高度な設定ダイアログ** | `ui/hidden_params_dialog.py` | 詳細パラメータの編集 |
| **文字起こし (メイン)** | `transcription/processor.py` | Whisper + VAD + 言語判定 |
| **マルチスレッド** | `transcription/threads.py` | `TranscriptionThread` / `RangeTranscriptionThread` |
| **モデルキャッシュ** | `transcription/model_cache.py` | モデルの事前ロード・保持 |
| **音声処理** | `transcription/audio.py` | FFmpeg・WAV 凖備 |
| **セグメント操作** | `services/segment_ops.py` | 分割・結合・削除などの純粋操作 |
| **再文字起こし操作** | `services/retranscribe_ops.py` | 動的分割・連結など高レベル処理 |
| **セグメント型** | `models/segment.py` | `Segment` データクラス |
| **時刻フォーマット** | `utils/timefmt.py` | `HH:MM:SS.mmm` ↔ 秒変換 |
| **セグメント表示** | `utils/segment_utils.py` | 表示テキスト・言語ラベル整形 |
| **書き出し** | `core/exporter.py` | TXT / SRT / JSON 生成 |
| **設定管理** | `core/constants.py` | 定数・WHISPER_MODELS 等 |
| **ロギング** | `core/logging_config.py` | ロギング初期化 |

### 2.2 処理フロー

#### (A) 初回文字起こし
```
ユーザー操作
  ↓
app.py: on_transcribe_button_clicked()
  ↓
TranscriptionThread スタート
  ├─ processor.process(): Whisper 全文処理
  ├─ セグメント逐次生成
  └─ signals.segment_added.emit(segment) 
      ↓
  app.py: on_segment_added_signal()
    ├─ テーブルへ追加
    └─ table_presenter.populate_table()
```

#### (B) 部分再文字起こし
```
ユーザー操作（分割 / 境界調整）
  ↓
segment_ops.py: split_segment_at_position()
  または segment_ops.py: adjust_boundary()
  ↓
RangeTranscriptionThread スタート
  ├─ processor.process_range(): 指定時間枠のみ処理
  └─ signals.range_completed.emit(segments)
      ↓
  app.py: on_range_retranscription_completed()
    ├─ 対象セグメントを置き換え
    └─ table_presenter.populate_table() で再描画
```

### 2.3 スレッド管理

#### TranscriptionThread (全文処理)
- **開始**: 「文字起こし開始」ボタン
- **処理**: `processor.process()` → Whisper で逐次セグメント生成
- **信号**: `segment_added` → テーブル逐次追加
- **キャンセル**: `cancel()` メソッドで中断フラグ設定

#### RangeTranscriptionThread (範囲処理)
- **開始**: セグメント分割・境界調整後
- **処理**: `processor.process_range(start, end)` → 指定枠内のみ
- **信号**: `range_completed` → テーブル置き換え
- **キューイング**: 複数要求が重なった場合は順次実行

---

## 3. 主要な操作処理

### 3.1 セグメント分割

**ファイル**: `services/segment_ops.py`

```python
def split_segment_at_position(
    segment: Segment,
    position: float,  # 秒
    text_lang1: str = None,
    text_lang2: str = None
) -> tuple[Segment, Segment]:
    """
    1 つのセグメントをカーソル位置で 2 分割。
    
    lang1 / lang2 テキスト長の比率でタイミングを按分。
    """
    ...
```

直後に `RangeTranscriptionThread` で両セグメントを再文字起こし。

### 3.2 セグメント結合

**ファイル**: `services/retranscribe_ops.py`

```python
def merge_contiguous_segments(
    segments: list[Segment],
    indices: list[int]  # 結合対象行の indices
) -> Segment:
    """複数セグメントを 1 つに結合し、時間枠を統合。"""
    ...
```

### 3.3 動的時間分割

**ファイル**: `services/retranscribe_ops.py`

```python
def dynamic_time_split(
    segments: list[Segment],
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 15.0
) -> list[Segment]:
    """
    Whisper の最適セグメント長に自動調整。
    実装詳細は要確認。
    """
    ...
```

---

## 4. モデル & キャッシュ

### 4.1 モデル初期化

**ファイル**: `transcription/model_cache.py`

```python
class ModelCache:
    def __init__(self, device: str = "cuda", model_name: str = "large-v3"):
        """
        モデルを遅延ロードで初期化。
        実際のロードは .get_xxx() 呼び出し時。
        """
        ...
    
    def get_whisper_model(self):
        """Whisper モデルを取得（未ロードなら初回ロード）"""
        ...
    
    def get_vad_model(self):
        """Silero VAD モデルを取得"""
        ...
```

### 4.2 ウォームアップ

**ファイル**: `ui/app.py: on_transcribe_button_clicked()`

全文文字起こし開始直後に：
1. `ModelCache.get_whisper_model()` 呼び出し
2. `ModelCache.get_vad_model()` 呼び出し

これにより初回セグメント生成時の遅延が最小化されます。

---

## 5. 設定・定数

### 5.1 config.toml

**プリセット例**:
```toml
[default]
device = "cuda"
model = "large-v3"
default_languages = ["ja", "ru"]

lang1_weight = 0.50
lang2_weight = 0.50

no_speech_threshold = 0.6
initial_prompt = ""

vad_filter = true
vad_threshold = 0.5
vad_min_speech_ms = 250
vad_min_silence_ms = 2000
```

**別プリセット例** （GUI で切り替え可能）:
```toml
[kapra]
device = "rocm"
model = "distil-large-v3"

[debug_mode]
device = "cpu"
model = "tiny"
```

### 5.2 core/constants.py

```python
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
DEFAULT_WATCHDOG_TIMEOUT_MS = 15000
MAX_RANGE_SEC = 3600  # 最大 1 時間単位の再文字起こし
```

---

## 6. ロギング & デバッグ

### 6.1 ロギング設定

`config.toml`:
```toml
[logging]
level = "DEBUG"
file_enabled = true
file_path = "app.log"
max_bytes = 1048576
backup_count = 3
```

### 6.2 デバッグ機能

セグメント再構築差分追跡:
```toml
[debug]
rebuild_diff = true
```

→ テーブル更新時に差分ログが出力される。

---

## 7. 開発時の参考実装

### 7.1 新規診断デバッグ機能の追加

1. `core/logging_config.py` で新しいロガーを定義
2. 必要な箇所で `logger.debug("...")` を挿入
3. `config.toml` に新しい `[debug]` キーを追加
4. `ui/app.py` で設定値を読み込んで機能選択

### 7.2 新規セグメント操作の追加

1. `services/segment_ops.py` または `services/retranscribe_ops.py` に関数を追加
2. `ui/app.py` の適切なボタンクリック / メニュー処理で呼び出し
3. 必要に応じて `RangeTranscriptionThread` をキュー追加

### 7.3 新規書き出し形式の追加

1. `core/exporter.py` に新しい `build_xxx_text(segments)` 関数を追加
2. `ui/app.py` のエクスポートダイアログで選択肢追加

---

## 8. 既知の制限 & 今後の改善

### 8.1 既知の制限
- セグメント数が数千を超える場合、UI スクロール性能の低下の可能性
- キャンセル直後の遅延シグナル処理は GUI 側で guard

### 8.2 検討中の改善
- SRT 行長自動折返し最適化
- Waveform 可視化オーバーレイ
- バッチ一括処理 (フォルダドロップ)
- 多言語 UI の拡張

---

## 9. コーディング規約

### 9.1 命名規則
- **クラス**: PascalCase (`VideoTranscriptionApp`)
- **関数/メソッド**: snake_case (`split_segment_at_position`)
- **定数**: UPPER_SNAKE_CASE (`DEFAULT_WATCHDOG_TIMEOUT_MS`)
- **プライベートメソッド**: `_` プレフィックス (`_toggle_log_panel`)

### 9.2 型アノテーション
可能な限り全関数に型アノテーションを指定。

```python
def split_segment(segment: Segment, pos: float) -> tuple[Segment, Segment]:
    ...
```

### 9.3 ドキュメント文字列
モジュール・クラス・公開関数には docstring を記述：

```python
def split_segment_at_position(segment: Segment, position: float) -> tuple[Segment, Segment]:
    """
    1 つのセグメントを指定位置で 2 分割。
    
    Args:
        segment: 分割対象セグメント
        position: カーソル位置（秒）
    
    Returns:
        (前半セグメント, 後半セグメント)
    """
    ...
```

---

## 10. テスト & 検証

### 10.1 基本動作チェック
```pwsh
python main.py
- [ ] 動画読み込み可能
- [ ] 文字起こし開始から完了
- [ ] テーブル逐次追加
- [ ] 分割 / 結合 / 削除機能
- [ ] 書き出し (TXT / SRT / JSON)
```

### 10.2 エッジケース
- 超短い動画 (< 1 秒)
- 無音区間のみ
- 極端に長い動画 (> 10 時間)
- 複数言語混在

---

ご質問・改善提案は Issue でお知らせください。
