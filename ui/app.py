"""アプリケーション本体 (VideoTranscriptionApp)。

main.py からクラスを分離し、以下を整理:
  - 重複インポート / 重複メソッドの除去
  - 欠落していた _check_split_watchdog の実装
  - EditDialog を ui.edit_dialog から利用
  - 逐次セグメント表示の format_ms 統一
  - 再文字起こし完了後の rebuild_aggregate_text 利用
  - デッドコード / 互換シムの削除
"""
from __future__ import annotations

import json
import re
import logging
import os
import subprocess
import sys
import time
import tomllib as _toml

try:
    import ctranslate2
    _CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0
except Exception:
    _CUDA_AVAILABLE = False

from PySide6.QtCore import Qt, QTimer, QUrl, Slot, QLoggingCategory
from PySide6.QtGui import QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.constants import (
    WHISPER_MODELS,
    WHISPER_LANGUAGES,
    WHISPER_LANGUAGES_JA,
    PLACEHOLDER_PENDING,
    DEFAULT_WATCHDOG_TIMEOUT_MS,
    MAX_RANGE_SEC,
)
from core.exporter import build_export_text, build_json_payload
from core.logging_config import setup_logging, get_logger
from models.segment import Segment, as_segment_list
from services.retranscribe_ops import dynamic_time_split, merge_contiguous_segments
from services.segment_ops import split_segment_at_position
from transcription.threads import TranscriptionThread, RangeTranscriptionThread
from ui.edit_dialog import EditDialog
from ui.hidden_params_dialog import HiddenParamsDialog
from ui.table_presenter import apply_prob_colors, populate_table, rebuild_aggregate_text
from utils.segment_utils import display_text
from utils.timefmt import format_ms, parse_to_ms

# Qt Multimedia FFmpeg ログを非表示
QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg=false")

logger = get_logger(__name__)


class VideoTranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("動画文字起こしエディタ")

        # 基本フィールド初期化
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.transcription_thread = None
        self.current_video_path = None
        self.transcription_result = None
        # 部分(結合)再文字起こし進行中フラグ
        self.range_retranscribing = False
        # 編集ダイアログのデフォルトサイズ
        self.split_dialog_size = (640, 200)
        # UI初期化中フラグ
        self._initializing = True

        # 設定ロード
        try:
            self.config = self._load_config()
        except Exception as e:
            logger.error(f"設定ロード失敗: {e}")
            self.config = {"default": {}}

        # UI 初期化とシグナル接続
        self._init_ui()
        self._connect_signals()

        # Application is ready
        self._initializing = False

        # テーブル操作シグナル
        self.transcription_table.cellClicked.connect(self._seek_to_table_row)
        self.transcription_table.cellDoubleClicked.connect(self._play_row_on_doubleclick)
        self.transcription_table.itemSelectionChanged.connect(self._update_split_button_state)

        # 初期ウィンドウサイズ
        self.setMinimumSize(800, 600)
        self.resize(1280, 800)
        self.move(0, 0)

    # ================================================================
    # 設定ロード
    # ================================================================

    def _load_config(self) -> dict:
        """config.toml を読み込み logging を初期化して返す。"""
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
        cfg_path = os.path.normpath(cfg_path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.toml not found: {cfg_path}")
        with open(cfg_path, "rb") as f:
            data = _toml.load(f)
        try:
            # Priority for logging config:
            # 1) data['advanced']['logging'] (new desired place)
            # 2) data['logging'] (legacy top-level)
            # 3) flattened logging keys inside data['advanced'] or data['hidden'] (back-compat)
            log_conf = None
            if isinstance(data, dict):
                # prefer advanced section
                adv = data.get('advanced') or data.get('hidden')
                if isinstance(adv, dict) and isinstance(adv.get('logging'), dict):
                    log_conf = dict(adv.get('logging'))
                elif isinstance(data.get('logging'), dict):
                    log_conf = dict(data.get('logging'))
                elif isinstance(adv, dict):
                    # look for flattened logging keys inside advanced/hidden
                    log_conf = {}
                    if 'log_file_enabled' in adv:
                        log_conf['file_enabled'] = bool(adv.get('log_file_enabled'))
                    if 'log_level' in adv:
                        log_conf['level'] = str(adv.get('log_level'))
                    if 'log_file_path' in adv:
                        log_conf['file_path'] = str(adv.get('log_file_path'))
                    if 'log_max_bytes' in adv:
                        log_conf['max_bytes'] = int(adv.get('log_max_bytes'))
                    if 'log_backup_count' in adv:
                        log_conf['backup_count'] = int(adv.get('log_backup_count'))
                    if not log_conf:
                        log_conf = None
            setup_logging(log_conf)
            logger.info("Logging initialized level=%s", (log_conf or {}).get('level', 'INFO'))
        except Exception:
            pass
        return data

    # ================================================================
    # UI 構築
    # ================================================================

    def _init_ui(self) -> None:
        """メイン UI を構築する。"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ---- 表示名/ヘルプテキスト (ツールチップ用) ----
        DISPLAY_LABELS = {
            "default_languages": "初期選択言語",
            "lang1_weight": "言語1の重み",
            "lang2_weight": "言語2の重み",
            "no_speech_threshold": "無音判定しきい値",
            "initial_prompt": "認識ヒント",
            "vad_filter": "VADで無音除外",
            "vad_threshold": "VADしきい値",
            "vad_min_speech_ms": "最短発話長 (ms)",
            "vad_min_silence_ms": "無音区切り (ms)",
        }

        HELP_TEXTS = {
            "プロファイル": (
                "プロファイル (= 設定プリセット) を切り替えます。\n"
                "例: 'kapra' は配信向けチューニング、'default' は汎用。\n"
                "切替時に: デバイス / モデル / 言語初期選択 / ウェイト / しきい値類 が再読込されます。\n"
                "config.toml に新しいセクションを追加して独自プリセットを作成できます。"
            ),
            "デバイス": (
                "Whisper を実行する計算デバイス。\n"
                "cuda: GPU 使用 (推奨 / 速度向上)\ncpu: CPU フォールバック (低速)\n"
                "・CUDA が利用不可の場合は自動的に cpu のみ選択可"
            ),
            "トランスクリプションモデル": (
                "音声→文字変換に使用する Whisper モデルサイズ。"
            ),
            "セグメンテーションモデル": (
                "音声区間の切り出し(セグメント化)に用いるモデル。"
            ),
            "lang1_weight": (
                "言語1の言語判定スコアへの補正係数。"
            ),
            "lang2_weight": (
                "言語2の言語判定スコアへの補正係数。"
            ),
            "default_languages": (
                "起動時にチェックされる言語。複数選択可。"
            ),
            "no_speech_threshold": (
                "無音とみなす確率しきい値。高いほど無音判定が厳しくなる。\n"
                "範囲: 0.0〜1.0"
            ),
            "vad_filter": (
                "音声区間検出 (VAD) で無音を除外する。"
            ),
            "vad_threshold": (
                "VAD の感度。高いほど音声検出が厳しくなる。\n"
                "範囲: 0.0〜1.0"
            ),
            "vad_min_speech_ms": (
                "発話として認める最小長さ (ms)。短すぎる発話は無視。\n"
                "範囲: 50〜2000"
            ),
            "vad_min_silence_ms": (
                "発話と発話を区切る無音の最小長さ (ms)。\n"
                "範囲: 100〜5000"
            ),
            "min_seg_dur": (
                "1 つのセグメントがこれ未満秒なら分割候補から除外 (過分割防止)。\n"
                "推奨: 0.4～0.8 秒。"
            ),
            "ambiguous_threshold": (
                "主要言語間の確信度差がこの値未満で 'あいまい' とみなし、\n"
                "両(複数)言語再トライ比較を行う境界。"
            ),
            "vad_level": (
                "VAD (Voice Activity Detection) 感度。\n"
                "0: もっとも寛容  3: 最も厳格"
            ),
            "gap_threshold": (
                "前セグメント終端と次セグメント開始の間隔がこの秒数以上なら '無音ギャップ' として扱う。"
            ),
            "include_silent": (
                "True で無音/スキップ理由を出力ログに残す。デバッグ用途向け。"
            ),
            "srt_max_line": (
                "SRT 生成時、1 セグメントを強制的に改行分割する際の文字数上限目安。"
            ),
            "initial_prompt": (
                "最初の推論呼び出しに与えるコンテキスト文字列。\n"
                "固有名詞/話題/口調を誘導したい場合に使用。空文字なら無効。"
            ),
            "output_format": (
                "保存出力フォーマット。\n"
                "txt: 単純テキスト  srt: 字幕フォーマット  json: 構造化データ"
            ),
            "silence_rms_threshold": (
                "これ未満の RMS (平均振幅) なら無音としてスキップ。"
            ),
            "min_voice_ratio": (
                "VAD で音声と判定されたフレーム比率の下限 (0～1)。下回れば無音扱い。"
            ),
            "max_silence_repeat": (
                "低エネルギー条件下で同一テキストが連続した場合に許容する回数。"
            ),
        }

        def display_label(key: str) -> str:
            return DISPLAY_LABELS.get(key, key)

        def help_label(key: str) -> QLabel:
            """ツールチップ付き '?' ラベルを生成。"""
            lbl = QLabel("?")
            lbl.setFixedWidth(16)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "QLabel { color: #007acc; font-weight: bold; "
                "border:1px solid #007acc; border-radius:8px; }"
            )
            lbl.setToolTip(HELP_TEXTS.get(key, key))
            return lbl

        # ================================================================
        # 左側: 動画プレイヤー + テーブル (垂直 Splitter)
        # ================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        vsplitter = QSplitter(Qt.Vertical)

        # ---- 動画コンテナ ----
        video_container = QWidget()
        video_vlayout = QVBoxLayout(video_container)
        video_vlayout.setContentsMargins(0, 0, 0, 0)
        video_vlayout.setSpacing(4)

        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.media_player.setVideoOutput(self.video_widget)
        video_vlayout.addWidget(self.video_widget, 1)

        # コントロール
        control_layout = QVBoxLayout()

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        control_layout.addWidget(self.position_slider)

        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00:00.000")
        self.total_time_label   = QLabel("00:00:00.000")
        self.auto_sync_check = QCheckBox("自動同期")
        self.auto_sync_check.setChecked(True)
        self.auto_sync_check.setToolTip(
            "再生中は現在位置に応じて対応行を自動選択します。停止/一時停止中は固定。"
        )
        time_layout.addWidget(self.current_time_label)
        time_layout.addWidget(self.auto_sync_check)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        control_layout.addLayout(time_layout)

        # Unified toolbar: open / play / stop / seek buttons / export buttons
        TOOL_BTN_HEIGHT = 32
        SEEK_BTN_WIDTH = 52

        def make_tool_btn(label: str | None, icon: QStyle.StandardPixmap | None, tooltip: str, enabled: bool = True) -> QPushButton:
            btn = QPushButton(label if label else "")
            if icon is not None:
                btn.setIcon(self.style().standardIcon(icon))
            btn.setToolTip(tooltip)
            btn.setEnabled(enabled)
            btn.setFixedHeight(TOOL_BTN_HEIGHT)
            return btn

        def make_seek_btn(label: str, delta_ms: int, tooltip: str) -> QPushButton:
            btn = QPushButton(label)
            btn.setEnabled(False)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _=None, d=delta_ms: self.seek_relative(d))
            btn.setFixedWidth(SEEK_BTN_WIDTH)
            btn.setFixedHeight(TOOL_BTN_HEIGHT)
            return btn

        toolbar = QHBoxLayout()
        # Open button
        self.open_button = make_tool_btn("動画を開く", None, "ファイルを開きます")
        self.open_button.clicked.connect(self.open_file)
        toolbar.addWidget(self.open_button)

        # Play / Stop
        self.play_button = make_tool_btn(None, QStyle.SP_MediaPlay, "再生 / 一時停止", False)
        self.play_button.clicked.connect(self.play_pause)
        toolbar.addWidget(self.play_button)

        self.stop_button = make_tool_btn(None, QStyle.SP_MediaStop, "停止", False)
        self.stop_button.clicked.connect(self.stop_video)
        toolbar.addWidget(self.stop_button)

        # Seek buttons
        self.seek_back_10_btn = make_seek_btn("◀10s", -10000, "10秒戻る")
        self.seek_back_1_btn  = make_seek_btn("◀1s",  -1000,  "1秒戻る")
        self.seek_fwd_1_btn   = make_seek_btn("1s▶",   1000,  "1秒進む")
        self.seek_fwd_10_btn  = make_seek_btn("10s▶", 10000,  "10秒進む")
        for btn in (self.seek_back_10_btn, self.seek_back_1_btn, self.seek_fwd_1_btn, self.seek_fwd_10_btn):
            toolbar.addWidget(btn)

        toolbar.addStretch()

        # Partial / Full export buttons (moved into toolbar so all target buttons align)
        self.partial_export_button = QPushButton("部分書き出し")
        self.partial_export_button.setEnabled(False)
        self.partial_export_button.setToolTip(
            "選択された行範囲の音声(WAV)とテキスト(TXT)を ./output に書き出し"
        )
        self.partial_export_button.clicked.connect(self.partial_export_selected)
        self.partial_export_button.setFixedHeight(TOOL_BTN_HEIGHT)
        toolbar.addWidget(self.partial_export_button)

        self.export_button = QPushButton("全文書き出し...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_transcription)
        self.export_button.setFixedHeight(TOOL_BTN_HEIGHT)
        toolbar.addWidget(self.export_button)

        control_layout.addLayout(toolbar)
        video_vlayout.addLayout(control_layout)

        # ---- テーブルコンテナ ----
        table_container = QWidget()
        table_vlayout = QVBoxLayout(table_container)
        table_vlayout.setContentsMargins(0, 0, 0, 0)
        table_vlayout.setSpacing(4)

        # Keep an empty spacer area where export buttons used to be (buttons moved to toolbar)
        top_export_bar = QHBoxLayout()
        top_export_bar.setContentsMargins(0, 0, 0, 0)
        top_export_bar.addStretch()
        table_vlayout.addLayout(top_export_bar)

        self.transcription_table = QTableWidget()
        self.transcription_table.setStyleSheet("""
            QTableWidget::item:selected { background: #1976d2; color: #ffffff; }
            QTableWidget::item:focus    { outline: none; }
            QTableWidget::item          { padding-top: 1px; padding-bottom: 1px; }
        """)
        self.transcription_table.setColumnCount(5)
        self.transcription_table.setHorizontalHeaderLabels(
            ["START", "END", "LANG1%", "LANG2%", "TEXT"]
        )
        # ヘッダーのコード表示は言語設定後に _update_table_headers() で更新
        self.transcription_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transcription_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transcription_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.transcription_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.transcription_table.customContextMenuRequested.connect(
            self._show_table_context_menu
        )
        self.transcription_table.setColumnWidth(0, 80)
        self.transcription_table.setColumnWidth(1, 80)
        self.transcription_table.setColumnWidth(2, 60)
        self.transcription_table.setColumnWidth(3, 60)
        self.transcription_table.setColumnWidth(4, 400)
        header = self.transcription_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        for i in range(5):
            item = self.transcription_table.horizontalHeaderItem(i)
            if item:
                item.setTextAlignment(Qt.AlignCenter)
        # 行高をコンパクト化
        vh = self.transcription_table.verticalHeader()
        compact_height = max(18, self.transcription_table.fontMetrics().height() + 4)
        vh.setDefaultSectionSize(compact_height)
        vh.setMinimumSectionSize(compact_height)
        table_vlayout.addWidget(self.transcription_table, 1)

        video_container.setMinimumHeight(180)

        vsplitter.addWidget(video_container)
        vsplitter.addWidget(table_container)
        vsplitter.setStretchFactor(0, 1)
        vsplitter.setStretchFactor(1, 4)
        vsplitter.setHandleWidth(6)
        vsplitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                              stop:0 #555, stop:0.5 #666, stop:1 #555);
                border-left: 1px solid #404040; border-right: 1px solid #404040;
            }
            QSplitter::handle:hover   { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                          stop:0 #666, stop:0.5 #777, stop:1 #666); }
            QSplitter::handle:pressed { background: #2d89ef; }
        """)
        try:
            vsplitter.setSizes([2000, 1200])
        except Exception:
            pass
        left_layout.addWidget(vsplitter, 1)

        # ================================================================
        # 右側: 設定パネル
        # ================================================================
        right_widget = QWidget()
        fixed_settings_width = 320
        right_widget.setMinimumWidth(fixed_settings_width)
        right_widget.setMaximumWidth(fixed_settings_width)
        right_layout = QVBoxLayout(right_widget)

        scroll_area = QScrollArea()
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # ---- 基本設定 ----
        model_group = QGroupBox("基本設定")
        model_layout = QGridLayout()

        model_layout.addWidget(QLabel("プロファイル:"), 0, 0)
        self.profile_combo = QComboBox()
        self.profiles = [k for k, v in self.config.items() if isinstance(v, dict) and k != 'hidden']
        # 'kapra' 優先、次に 'default'、残りはアルファベット順
        ordered_profiles = (
            [p for p in self.profiles if p == 'kapra'] +
            [p for p in self.profiles if p == 'default'] +
            sorted(p for p in self.profiles if p not in ('kapra', 'default'))
        )
        self.profiles = ordered_profiles
        self.profile_combo.addItems(self.profiles)
        self.current_profile_name = 'kapra' if 'kapra' in self.profiles else 'default'
        self.profile_combo.setCurrentText(self.current_profile_name)
        model_layout.addWidget(self.profile_combo, 0, 1)
        model_layout.addWidget(help_label("プロファイル"), 0, 2)

        model_layout.addWidget(QLabel("デバイス:"), 1, 0)
        self.device_combo = QComboBox()
        device_items = [("CPU", "cpu")]
        if _CUDA_AVAILABLE:
            device_items.append(("GPU", "cuda"))
        for label, value in device_items:
            self.device_combo.addItem(label, value)
        base_prof = self.config.get(self.current_profile_name, self.config.get('default', {}))
        default_device = base_prof.get("device", "cuda")
        if default_device not in [v for _, v in device_items]:
            label = "GPU" if default_device.lower() == "cuda" else default_device.upper()
            self.device_combo.addItem(label, default_device)
        idx = self.device_combo.findData(default_device)
        if idx < 0:
            idx = self.device_combo.findText(default_device)
        self.device_combo.setCurrentIndex(max(0, idx))
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        model_layout.addWidget(self.device_combo, 1, 1)
        model_layout.addWidget(help_label("デバイス"), 1, 2)

        # CPUモード警告ラベル
        self.cpu_warning_label = QLabel("⚠ CPUモードは処理が遅くなります")
        self.cpu_warning_label.setStyleSheet("""
            QLabel {
                color: #ff6b00;
                font-weight: bold;
                font-size: 10px;
                background-color: #fff3cd;
                padding: 4px 8px;
                border: 1px solid #ffc107;
                border-radius: 3px;
            }
        """)
        self.cpu_warning_label.setVisible(default_device.lower() == "cpu")
        model_layout.addWidget(self.cpu_warning_label, 3, 0, 1, 3)

        model_layout.addWidget(QLabel("トランスクリプションモデル:"), 4, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        default_model = base_prof.get("model", "large-v3")
        if default_model not in WHISPER_MODELS:
            self.model_combo.addItem(default_model)
        self.model_combo.setCurrentText(default_model)
        model_layout.addWidget(self.model_combo, 4, 1)
        model_layout.addWidget(help_label("トランスクリプションモデル"), 4, 2)

        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)

        # ---- 言語設定 ----
        lang_group  = QGroupBox("言語設定")
        lang_layout = QVBoxLayout()

        # プルダウン選択肢 ("code - 日本語名" 形式)
        # advanced section (preferred) - fall back to legacy 'hidden' for compat
        advanced_cfg = self.config.get("advanced", self.config.get("hidden", {})) if isinstance(self.config, dict) else {}
        allowed_langs = advanced_cfg.get("available_languages")
        if not isinstance(allowed_langs, list) or not allowed_langs:
            allowed_langs = [
                "en", "es", "it", "ja", "de", "zh",
                "ru", "ko", "pt", "fr", "pl", "nl",
            ]
        lang_items = []
        for code in allowed_langs:
            if code in WHISPER_LANGUAGES:
                name = WHISPER_LANGUAGES_JA.get(code, WHISPER_LANGUAGES[code])
                lang_items.append((code, f"{code} - {name}"))

        def make_lang_combo_row(default_code: str, weight_key: str, prof: dict,
                                include_none: bool = False):
            """言語コンボ + 重みスライダーの行を生成 (2 行レイアウト)。"""
            row_w = QWidget()
            vl = QVBoxLayout(row_w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(4)
            combo = QComboBox()
            if include_none:
                combo.addItem("なし", None)
            for code, label in lang_items:
                combo.addItem(label, code)
            # デフォルト選択
            idx = combo.findData(default_code)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif not include_none:
                combo.setCurrentIndex(0)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0); slider.setMaximum(10); slider.setSingleStep(1)
            slider.setFixedWidth(140)
            val = prof.get(weight_key, 0.5)
            if not isinstance(val, (int, float)):
                val = 0.5
            slider.setValue(int(round(val * 10)))
            value_label = QLabel(f"{val:.1f}")
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.addWidget(combo)
            top.addStretch()
            bottom = QHBoxLayout()
            bottom.setContentsMargins(0, 0, 0, 0)
            bottom.addWidget(QLabel("重み"))
            bottom.addWidget(slider)
            bottom.addWidget(value_label)
            bottom.addWidget(help_label(weight_key))
            vl.addLayout(top)
            vl.addLayout(bottom)
            return combo, slider, value_label, row_w

        dlangs = base_prof.get("default_languages", ["ja", "ru"])
        default_lang1 = dlangs[0] if dlangs else "ja"
        default_lang2 = dlangs[1] if len(dlangs) > 1 else "ru"

        self.lang1_combo, self.lang1_weight_slider, self.lang1_weight_value_label, lang1_row = \
            make_lang_combo_row(default_lang1, "lang1_weight", base_prof, include_none=False)
        self.lang2_combo, self.lang2_weight_slider, self.lang2_weight_value_label, lang2_row = \
            make_lang_combo_row(default_lang2, "lang2_weight", base_prof, include_none=True)

        # BUG-6 修正: weight 連動スライダーは両方 blockSignals してから値を設定
        def on_lang1_weight_changed(v):
            self.lang1_weight_value_label.setText(f"{v/10:.1f}")
            self.lang2_weight_slider.blockSignals(True)
            self.lang1_weight_slider.blockSignals(True)
            self.lang2_weight_slider.setValue(10 - v)
            self.lang2_weight_value_label.setText(f"{(10-v)/10:.1f}")
            self.lang2_weight_slider.blockSignals(False)
            self.lang1_weight_slider.blockSignals(False)

        def on_lang2_weight_changed(v):
            self.lang2_weight_value_label.setText(f"{v/10:.1f}")
            self.lang1_weight_slider.blockSignals(True)
            self.lang2_weight_slider.blockSignals(True)
            self.lang1_weight_slider.setValue(10 - v)
            self.lang1_weight_value_label.setText(f"{(10-v)/10:.1f}")
            self.lang1_weight_slider.blockSignals(False)
            self.lang2_weight_slider.blockSignals(False)

        self.lang1_weight_slider.valueChanged.connect(on_lang1_weight_changed)
        self.lang2_weight_slider.valueChanged.connect(on_lang2_weight_changed)

        # 初期値を正規化 (合計1.0)
        w1 = base_prof.get("lang1_weight", 0.5)
        w2 = base_prof.get("lang2_weight", 0.5)
        total = (w1 or 0) + (w2 or 0)
        if total > 0:
            w1, w2 = w1 / total, w2 / total
        self.lang1_weight_slider.setValue(int(round(w1 * 10)))
        self.lang2_weight_slider.setValue(int(round(w2 * 10)))

        def _sync_weights_for_none() -> None:
            # lang2 が None のときは lang1=1.0, lang2=0.0 に固定
            if self.lang2_combo.currentData() is None:
                self.lang1_weight_slider.blockSignals(True)
                self.lang2_weight_slider.blockSignals(True)
                self.lang1_weight_slider.setValue(10)
                self.lang2_weight_slider.setValue(0)
                self.lang1_weight_value_label.setText("1.0")
                self.lang2_weight_value_label.setText("0.0")
                self.lang1_weight_slider.blockSignals(False)
                self.lang2_weight_slider.blockSignals(False)
                self.lang1_weight_slider.setEnabled(False)
                self.lang2_weight_slider.setEnabled(False)
            else:
                self.lang1_weight_slider.setEnabled(True)
                self.lang2_weight_slider.setEnabled(True)

        # 言語変更でテーブルヘッダを更新
        self.lang1_combo.currentIndexChanged.connect(self._update_table_headers)
        self.lang2_combo.currentIndexChanged.connect(self._update_table_headers)
        self.lang2_combo.currentIndexChanged.connect(_sync_weights_for_none)
        _sync_weights_for_none()

        lang_layout.addWidget(lang1_row)
        lang_layout.addWidget(lang2_row)
        lang_group.setLayout(lang_layout)
        scroll_layout.addWidget(lang_group)

        # ---- プロファイル適用ロジック ----
        def apply_profile(name: str) -> None:
            if name not in self.config:
                return
            prof = self.config[name]
            self.current_profile_name = name

            dev = prof.get("device")
            if dev:
                idx = self.device_combo.findData(dev)
                if idx < 0:
                    idx = self.device_combo.findText(dev)
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)

            for combo, key in (
                (self.model_combo, "model"),
            ):
                val = prof.get(key)
                if val and val in [combo.itemText(i) for i in range(combo.count())]:
                    combo.setCurrentText(val)

            dlangs = prof.get("default_languages", [])
            if dlangs:
                idx1 = self.lang1_combo.findData(dlangs[0])
                if idx1 >= 0:
                    self.lang1_combo.setCurrentIndex(idx1)
                lang2_code = dlangs[1] if len(dlangs) > 1 else None
                idx2 = self.lang2_combo.findData(lang2_code)  # None = "なし"
                if idx2 >= 0:
                    self.lang2_combo.setCurrentIndex(idx2)

            # weight の合計が 1.0 になるように正規化
            w1 = prof.get("lang1_weight", 0.5)
            w2 = prof.get("lang2_weight", 0.5)
            if isinstance(w1, (int, float)) and isinstance(w2, (int, float)):
                total = w1 + w2
                if total > 0:
                    w1, w2 = w1 / total, w2 / total
                self.lang1_weight_slider.setValue(int(round(w1 * 10)))
                self.lang2_weight_slider.setValue(int(round(w2 * 10)))

            for key, ctrl in getattr(self, 'detail_controls', {}).items():
                if key not in prof:
                    continue
                val = prof[key]
                if isinstance(ctrl, QCheckBox) and isinstance(val, bool):
                    ctrl.setChecked(val)
                elif isinstance(ctrl, QSpinBox) and isinstance(val, int):
                    ctrl.setValue(val)
                elif isinstance(ctrl, QDoubleSpinBox) and isinstance(val, (int, float)):
                    ctrl.setValue(float(val))
                elif isinstance(ctrl, QLineEdit) and isinstance(val, str):
                    ctrl.setText(val)
                elif isinstance(ctrl, QTextEdit) and isinstance(val, str):
                    ctrl.setPlainText(val)
                elif isinstance(ctrl, QComboBox) and isinstance(val, str):
                    if ctrl.findText(val) < 0:
                        ctrl.addItem(val)
                    ctrl.setCurrentText(val)

        self.profile_combo.currentTextChanged.connect(apply_profile)

        # ---- 詳細設定 ----
        detail_group  = QGroupBox("詳細設定")
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(4)
        self.detail_controls: dict[str, QWidget] = {}

        # 基本設定グループで使用済みのキーは除外
        exclude_keys = {
            "device", "model",
            "default_languages", "lang1_weight", "lang2_weight",
              "vad_filter",
              "beam_size",  # 隠しパラメータダイアログで設定
        }

        # 詳細設定の表示順序を明示的に指定
        detail_keys_order = [
            "no_speech_threshold",  # 無音判定しきい値
            "vad_threshold",         # VAD閾値
            "vad_min_speech_ms",     # 最短発話長
            "vad_min_silence_ms",    # 無音区切り
            "initial_prompt",       # 認識ヒント
        ]

        for key in detail_keys_order:
            if key not in base_prof or key in exclude_keys:
                continue
            value = base_prof[key]

            # initial_prompt は複数行テキストエリア
            if key == "initial_prompt":
                prompt_row = QWidget()
                prompt_row_layout = QHBoxLayout(prompt_row)
                prompt_row_layout.setContentsMargins(0, 0, 0, 0)
                prompt_row_layout.addWidget(QLabel(f"{display_label(key)}:"))
                prompt_row_layout.addStretch()
                prompt_row_layout.addWidget(help_label(key))
                detail_layout.addWidget(prompt_row)
                txt = QTextEdit()
                txt.setMaximumHeight(80)
                txt.setPlainText(value or "")
                detail_layout.addWidget(txt)
                self.detail_controls[key] = txt
                continue

            row_container = QWidget()
            row_h = QHBoxLayout(row_container)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.addWidget(QLabel(f"{display_label(key)}:"))

            ctrl: QWidget | None = None
            if isinstance(value, bool):
                ctrl = QCheckBox()
                if key == "vad_filter":
                    ctrl.setChecked(True)
                    ctrl.setEnabled(False)
                else:
                    ctrl.setChecked(value)
            elif isinstance(value, int):
                if key == "vad_level":
                    ctrl = QComboBox()
                    for i in range(4):
                        ctrl.addItem(str(i), i)
                    idx = ctrl.findText(str(value))
                    ctrl.setCurrentIndex(max(0, idx))
                else:
                    ctrl = QSpinBox()
                    ctrl.setRange(-999999, 999999)
                    if key == "vad_min_speech_ms":
                        ctrl.setRange(50, 2000)
                    if key == "vad_min_silence_ms":
                        ctrl.setRange(100, 5000)
                    if key == "srt_max_line":
                        ctrl.setRange(1, 1000)
                    ctrl.setValue(value)
            elif isinstance(value, float):
                ctrl = QDoubleSpinBox()
                ctrl.setDecimals(4)
                ctrl.setRange(-1e9, 1e9)
                ctrl.setSingleStep(0.05)
                if key == "vad_threshold":
                    ctrl.setRange(0.0, 1.0)
                    ctrl.setDecimals(2)
                    ctrl.setSingleStep(0.05)
                if key == "no_speech_threshold":
                    ctrl.setDecimals(2)
                    ctrl.setSingleStep(0.05)
                if key in {"min_seg_dur", "gap_threshold"}:
                    ctrl.setRange(0.0, 60.0)
                if key in {"mix_threshold", "ambiguous_threshold"}:
                    ctrl.setRange(0.0, 100.0)
                    ctrl.setSingleStep(0.5)
                ctrl.setValue(value)
            elif isinstance(value, str):
                if key == "output_format":
                    ctrl = QComboBox()
                    candidates = ["txt", "srt", "json"]
                    if value not in candidates:
                        candidates.append(value)
                    ctrl.addItems(candidates)
                    ctrl.setCurrentText(value)
                else:
                    ctrl = QLineEdit()
                    ctrl.setText(value)
            else:
                ctrl = QLineEdit()
                ctrl.setText(str(value))

            if ctrl is not None:
                row_h.addWidget(ctrl)
                row_h.addWidget(help_label(key))
                self.detail_controls[key] = ctrl
                detail_layout.addWidget(row_container)

        detail_group.setLayout(detail_layout)
        scroll_layout.addWidget(detail_group)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        right_layout.addWidget(scroll_area, 1)

        # ---- 文字起こし実行エリア ----
        transcribe_layout = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.transcribe_button = QPushButton("文字起こしを開始")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        btn_row.addWidget(self.transcribe_button)
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_transcription)
        btn_row.addWidget(self.cancel_button)
        btn_row.addStretch()
        self.hidden_params_button = QPushButton("上級者向け設定...")
        self.hidden_params_button.clicked.connect(self._open_hidden_params_dialog)
        self.hidden_params_button.setToolTip("Phase別beam_sizeなど上級者向け設定を編集")
        btn_row.addWidget(self.hidden_params_button)
        transcribe_layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        transcribe_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("待機中")
        self.status_label.setMinimumHeight(22)
        transcribe_layout.addWidget(self.status_label)

        # ログパネル（常時表示）
        self.log_panel_container = QWidget()
        log_layout = QVBoxLayout(self.log_panel_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(True)
        self.log_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size:11px; }"
        )
        self.log_panel_container.setMinimumHeight(80)
        self.log_panel_container.setMaximumHeight(120)
        self.log_text.setMinimumHeight(80)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        transcribe_layout.addWidget(self.log_panel_container)
        right_layout.addLayout(transcribe_layout, 0)

        # ---- 左右スプリッター ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.media_player.positionChanged.connect(self._position_changed)
        self.media_player.durationChanged.connect(self._duration_changed)
        self.media_player.playbackStateChanged.connect(self._media_state_changed)
        self.position_slider.sliderMoved.connect(self._set_position)
        self.auto_sync_check.toggled.connect(self._on_auto_sync_toggled)

        # UI初期化完了
        self._initializing = False
        self._update_table_headers()

    # ================================================================
    # ヘルパー
    # ================================================================

    def _get_selected_device(self) -> str:
        data = self.device_combo.currentData()
        if data:
            return str(data)
        return self.device_combo.currentText().strip().lower()

    def _collect_selected_rows(self) -> list[int]:
        rows: set[int] = set()
        for r in self.transcription_table.selectedRanges():
            for i in range(r.topRow(), r.bottomRow() + 1):
                rows.add(i)
        return sorted(rows)

    def _rebuild_text_and_refresh(self) -> None:
        """集約テキスト再構築とテーブル再描画を一括実行。"""
        try:
            rebuild_aggregate_text(self.transcription_result)
            populate_table(self.transcription_table, self.transcription_result)
        except Exception:
            pass

    def _can_retranscribe_selection(self, rows: list[int]) -> bool:
        """選択行リストが再文字起こし可能かチェック。"""
        if not self.transcription_result:
            return False
        segs = self.transcription_result.get('segments', [])
        if not rows:
            return False
        try:
            target = [segs[i] for i in rows]
        except Exception:
            return False
        if len(rows) == 1:
            return True
        # 複数行: 連続性 & 30 秒以内チェック
        for a, b in zip(rows, rows[1:]):
            if b != a + 1:
                return False
        start_sec = float(target[0].get('start', 0.0))
        end_sec   = float(target[-1].get('end', start_sec))
        return (end_sec - start_sec) <= 30.0

    def _update_split_button_state(self) -> None:
        """分割/動的分割/削除ボタンの活性状態を更新。
        各ボタンは旧実装では操作バーにあったが現在は右クリックメニューに統合済み。
        hasattr ガードで将来の再追加に対応する。
        """
        if not hasattr(self, 'split_button'):
            return
        busy   = getattr(self, 'range_retranscribing', False)
        result = getattr(self, 'transcription_result', None)
        if not result:
            for attr in ('split_button', 'dynamic_split_button', 'delete_button'):
                if hasattr(self, attr):
                    getattr(self, attr).setEnabled(False)
            return
        segs = result.get('segments', [])
        rows = self._collect_selected_rows()

        if hasattr(self, 'split_button'):
            self.split_button.setEnabled(
                not busy and len(rows) == 1 and 0 <= rows[0] < len(segs)
            )
        if hasattr(self, 'dynamic_split_button'):
            can_dyn = False
            if not busy and rows:
                if len(rows) == 1 and 0 <= rows[0] < len(segs):
                    can_dyn = True
                elif (len(rows) == 2 and rows[1] == rows[0] + 1
                      and all(0 <= r < len(segs) for r in rows)):
                    can_dyn = True
            self.dynamic_split_button.setEnabled(can_dyn)
        if hasattr(self, 'delete_button'):
            self.delete_button.setEnabled(bool(rows) and not busy and bool(segs))

    def _start_watchdog(self) -> None:
        """ウォッチドッグタイマーを (再)起動する。"""
        try:
            if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                self._split_watchdog_timer.stop()
            self._split_watchdog_start = time.time()
            self._split_watchdog_timer = QTimer(self)
            self._split_watchdog_timer.setSingleShot(True)
            self._split_watchdog_timer.timeout.connect(self._check_split_watchdog)
            self._split_watchdog_timer.start(DEFAULT_WATCHDOG_TIMEOUT_MS)
        except Exception:
            pass

    def _stop_watchdog(self) -> None:
        """ウォッチドッグタイマーを停止する。"""
        try:
            if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                self._split_watchdog_timer.stop()
        except Exception:
            pass

    def _check_split_watchdog(self) -> None:
        """分割ジョブがタイムアウトした場合の回復処理。"""
        if not getattr(self, 'range_retranscribing', False):
            return  # 既に完了済みなら何もしない
        logger.warning('[WATCHDOG] 分割再文字起こしがタイムアウトしました。処理を中断します。')
        self.range_retranscribing = False
        if hasattr(self, '_pending_rejobs'):
            self._pending_rejobs.clear()
        if hasattr(self, 'range_thread') and self.range_thread and self.range_thread.isRunning():
            try:
                self.range_thread.terminate()
            except Exception:
                pass
        self._rebuild_text_and_refresh()
        self.status_label.setText('分割再文字起こしがタイムアウトしました')
        self._update_split_button_state()

    def _append_log(self, message: str) -> None:
        """ログパネルにメッセージを追記し、表示中なら末尾にスクロール。"""
        if hasattr(self, 'log_text'):
            self.log_text.append(message)
            if self.log_text.isVisible():
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )

    def _update_table_row(self, row_index: int, seg: Segment) -> None:
        """指定行のテーブルセルを最新の Segment 内容で部分更新。"""
        if row_index < 0 or row_index >= self.transcription_table.rowCount():
            return
        p1 = seg.lang1_prob
        p2 = seg.lang2_prob
        item1 = QTableWidgetItem(f"{p1:.2f}")
        item2 = QTableWidgetItem(f"{p2:.2f}")
        txt_item = QTableWidgetItem(display_text(seg))
        if p1 >= p2:
            item1.setForeground(QColor(200, 0, 0))
            item2.setForeground(QColor(0, 0, 180))
        else:
            item2.setForeground(QColor(200, 0, 0))
            item1.setForeground(QColor(0, 0, 180))
        self.transcription_table.setItem(row_index, 2, item1)
        self.transcription_table.setItem(row_index, 3, item2)
        self.transcription_table.setItem(row_index, 4, txt_item)

    # ================================================================
    # テーブル操作
    # ================================================================

    def _seek_to_table_row(self, row: int, col: int) -> None:
        """テーブル行クリック: その行の開始位置にシーク。"""
        item = self.transcription_table.item(row, 0)
        if not item:
            return
        try:
            self.media_player.setPosition(parse_to_ms(item.text()))
        except Exception:
            pass

    def _play_row_on_doubleclick(self, row: int, col: int) -> None:
        """テーブル行をダブルクリック: 開始位置にシークして再生開始。"""
        if not self.current_video_path:
            return
        self._seek_to_table_row(row, col)
        if self.media_player.playbackState() != QMediaPlayer.PlayingState:
            self.media_player.play()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def sync_table_selection_with_position(self, position_ms: int) -> None:
        """再生位置 (ms) に対応するテーブル行を選択してスクロール。"""
        pos_sec = position_ms / 1000.0
        for row in range(self.transcription_table.rowCount()):
            si = self.transcription_table.item(row, 0)
            ei = self.transcription_table.item(row, 1)
            if not si or not ei:
                continue
            try:
                st = parse_to_ms(si.text()) / 1000.0
                ed = parse_to_ms(ei.text()) / 1000.0
                if st <= pos_sec < ed:
                    self.transcription_table.setCurrentCell(row, 0)
                    self.transcription_table.scrollToItem(
                        self.transcription_table.item(row, 0)
                    )
                    return
            except Exception:
                continue

    def _show_table_context_menu(self, pos) -> None:
        """右クリックコンテキストメニューを表示。"""
        if not getattr(self, 'transcription_result', None):
            return

        global_pos = self.transcription_table.viewport().mapToGlobal(pos)

        # 右クリック時は一時停止 (編集中の誤進行防止)
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                self.media_player.pause()
        except Exception:
            pass

        menu = QMenu(self)
        act_play    = menu.addAction("ここから再生")
        act_edit    = menu.addAction("このテキストを編集")
        act_re      = menu.addAction("これらの結合＆再文字起こし")
        act_dynamic = menu.addAction("現在位置で分割＆再文字起こし")
        act_delete  = menu.addAction("選択行を削除")
        
        menu.addSeparator()
        act_set_silence = menu.addAction("無音に設定")
        act_retrans_lang1 = menu.addAction(f"{self.lang1_combo.currentData() or 'LANG1'}で強制再認識")
        lang2 = self.lang2_combo.currentData()
        act_retrans_lang2 = menu.addAction(f"{lang2 or 'LANG2'}で強制再認識") if lang2 else None

        # 進行中はすべて無効
        if getattr(self, 'range_retranscribing', False):
            for a in menu.actions():
                a.setEnabled(False)
            menu.exec(global_pos)
            return

        # 右クリックした行が未選択なら単一選択へ
        item = self.transcription_table.itemAt(pos)
        if item:
            row = item.row()
            if row not in [i.row() for i in self.transcription_table.selectedItems()]:
                self.transcription_table.selectRow(row)

        rows = self._collect_selected_rows()
        segs = self.transcription_result.get('segments', [])

        # 編集: 単一行のみ
        if len(rows) != 1 or not (0 <= rows[0] < len(segs)):
            act_edit.setEnabled(False)

        # 動的分割: 1行選択時のみ
        can_dynamic = len(rows) == 1 and rows and 0 <= rows[0] < len(segs)
        if not can_dynamic:
            act_dynamic.setEnabled(False)

        # 再生: 先頭行が有効なら可
        if not rows or not (0 <= rows[0] < len(segs)):
            act_play.setEnabled(False)

        # 再文字起こし
        if not self._can_retranscribe_selection(rows):
            act_re.setEnabled(False)

        # 削除
        if not rows:
            act_delete.setEnabled(False)

        # 無音に設定、LANG1/LANG2強制再認識: 選択行があれば可能
        if not rows or not all(0 <= r < len(segs) for r in rows):
            act_set_silence.setEnabled(False)
            act_retrans_lang1.setEnabled(False)
            if act_retrans_lang2:
                act_retrans_lang2.setEnabled(False)

        chosen = menu.exec(global_pos)
        if chosen is None:
            return

        if chosen == act_play and act_play.isEnabled():
            r0 = rows[0]
            try:
                self.media_player.setPosition(int(float(segs[r0].get('start', 0.0)) * 1000))
                self.media_player.play()
            except Exception:
                pass

        elif chosen == act_edit and act_edit.isEnabled():
            self._invoke_edit_dialog()

        elif chosen == act_dynamic and act_dynamic.isEnabled():
            self.split_or_adjust_at_current_position()

        elif chosen == act_re and act_re.isEnabled():
            self.retranscribe_selected()

        elif chosen == act_delete and act_delete.isEnabled():
            self.delete_selected_segments()

        elif chosen == act_set_silence and act_set_silence.isEnabled():
            self.set_segments_silence(rows)

        elif chosen == act_retrans_lang1 and act_retrans_lang1.isEnabled():
            self.retranscribe_as_language(rows, self.lang1_combo.currentData() or 'ja')

        elif act_retrans_lang2 and chosen == act_retrans_lang2 and act_retrans_lang2.isEnabled():
            self.retranscribe_as_language(rows, self.lang2_combo.currentData())

    # ================================================================
    # 編集ダイアログ
    # ================================================================

    def _invoke_edit_dialog(self) -> None:
        """現在の単一選択行に対して EditDialog を開く。"""
        if not getattr(self, 'transcription_result', None):
            return
        if getattr(self, 'range_retranscribing', False):
            return
        rows = self._collect_selected_rows()
        if len(rows) != 1:
            return
        segs = self.transcription_result.get('segments', [])
        r = rows[0]
        if r < 0 or r >= len(segs):
            return
        self._open_edit_dialog_for_row(r)

    def _open_edit_dialog_for_row(self, row: int) -> None:
        """指定行の EditDialog を開き、結果を transcription_result に反映する。"""
        if not self.transcription_result:
            return
        segs = self.transcription_result.get('segments', [])
        if row < 0 or row >= len(segs):
            return
        seg = segs[row]

        lang1_code = self.lang1_combo.currentData() or 'ja'
        lang2_code = self.lang2_combo.currentData()  # may be None
        dlg = EditDialog(seg, parent=self, dialog_size=self.split_dialog_size,
                         lang1_code=lang1_code, lang2_code=lang2_code)
        if dlg.exec() != EditDialog.Accepted:
            return

        if dlg.mode == 'split':
            self.perform_segment_split(row, dlg.split_which, dlg.split_pos)

        elif dlg.mode == 'edit':
            seg['text_lang1'] = dlg.new_text_lang1
            seg['text_lang2'] = dlg.new_text_lang2
            seg['chosen_language'] = dlg.chosen_language
            # 表示用 text も更新
            seg['text'] = dlg.new_text_lang1 if dlg.chosen_language == lang1_code else dlg.new_text_lang2
            self._rebuild_text_and_refresh()
            self.status_label.setText("編集を保存しました")

    # ================================================================
    # 手動分割 / 境界調整 / 再文字起こし
    # ================================================================

    def perform_segment_split(self, row: int, which: str, pos: int) -> None:
        """指定行セグメントを文字位置で 2 分割し、前後半を再文字起こし。"""
        if not getattr(self, 'transcription_result', None):
            return
        original = self.transcription_result.get('segments', [])
        new_list = split_segment_at_position(original, row, which, pos)
        if new_list is original or len(new_list) == len(original):
            return
        self.transcription_result['segments'] = new_list
        front = new_list[row]
        back  = new_list[row + 1]
        self.range_retranscribing = True
        self._pending_rejobs = [
            ('front', front['start'], front['end']),
            ('back',  back['start'],  back['end']),
        ]
        self._split_row_base = row
        self._update_split_button_state()
        self._run_next_split_rejob()

    def split_or_adjust_at_current_position(self) -> None:
        """1行選択時のみ: 現在位置で時間基準に2分割し、前後区間を再文字起こし。"""
        if getattr(self, 'range_retranscribing', False):
            return
        if not getattr(self, 'transcription_result', None):
            return
        segs = as_segment_list(self.transcription_result.get('segments', []))
        rows = self._collect_selected_rows()
        if not rows or len(rows) != 1:
            return
        
        r = rows[0]
        if r < 0 or r >= len(segs):
            return
        
        try:
            cur_sec = self.media_player.position() / 1000.0
        except Exception:
            return

        # セグメント内部で時間分割
        new_list, front_index = dynamic_time_split(
            self.transcription_result.get('segments', []), r, cur_sec
        )
        if new_list is None:
            return
        self.transcription_result['segments'] = new_list

        # 分割直後は旧テキスト断片を消去してプレースホルダを表示
        for segx in (new_list[front_index], new_list[front_index + 1]):
            segx['text'] = segx['text_lang1'] = segx['text_lang2'] = PLACEHOLDER_PENDING
            segx['lang1_prob'] = segx['lang2_prob'] = 0.0
            segx['chosen_language'] = None

        # テーブルのプレースホルダを即時反映
        try:
            for idx_tmp in (front_index, front_index + 1):
                if 0 <= idx_tmp < self.transcription_table.rowCount():
                    item = self.transcription_table.item(idx_tmp, 4)
                    if item is None:
                        item = QTableWidgetItem(PLACEHOLDER_PENDING)
                        self.transcription_table.setItem(idx_tmp, 4, item)
                    else:
                        item.setText(PLACEHOLDER_PENDING)
        except Exception:
            pass

        self.range_retranscribing = True
        self._pending_rejobs = [
            ('front', new_list[front_index]['start'],     new_list[front_index]['end']),
            ('back',  new_list[front_index + 1]['start'], new_list[front_index + 1]['end']),
        ]
        self._split_row_base = front_index
        self._start_watchdog()
        self._run_next_split_rejob()

    def _run_next_split_rejob(self) -> None:
        """分割後のキューに従って順次 RangeTranscriptionThread を起動。"""
        if not getattr(self, '_pending_rejobs', None):
            # 全ジョブ完了
            self.range_retranscribing = False
            self._rebuild_text_and_refresh()
            self.status_label.setText("分割＆再文字起こし完了")
            self._stop_watchdog()
            return

        job = self._pending_rejobs.pop(0)
        kind, start_sec, end_sec = job
        half_label = '前半' if kind == 'front' else '後半'
        self.status_label.setText(f"{half_label}再文字起こし開始…")
        self.progress_bar.setValue(0)

        options = {
            'model':   self.model_combo.currentText(),
            'device':  self._get_selected_device(),
            'lang1_weight': self.lang1_weight_slider.value() / 10.0,
            'lang2_weight': self.lang2_weight_slider.value() / 10.0,
            'lang1': self.lang1_combo.currentData() or 'ja',
            'lang2': self.lang2_combo.currentData(),
        }
        lang1 = self.lang1_combo.currentData() or 'ja'
        lang2 = self.lang2_combo.currentData()
        options['languages'] = [lang1] + ([lang2] if lang2 else [])
        hidden = self.config.get('hidden', {}) if isinstance(self.config, dict) else {}
        for k in ('debug_prob_log',):
            if k in hidden:
                options.setdefault(k, hidden[k])
        self._active_split_kind = kind
        self.range_thread = RangeTranscriptionThread(
            self.current_video_path, start_sec, end_sec, options
        )
        self.range_thread.progress.connect(self._on_range_progress)
        self.range_thread.status.connect(self._on_range_status)
        self.range_thread.range_finished.connect(self._on_split_rejob_finished)
        self.range_thread.error.connect(self._on_split_rejob_error)
        self.range_thread.device_fallback_warning.connect(self._on_device_fallback)
        self.range_thread.start()

    @Slot(dict)
    def _on_split_rejob_finished(self, seg: dict) -> None:
        """分割後の個別(前半/後半)再文字起こし完了ハンドラ。"""
        if not getattr(self, 'transcription_result', None):
            return
        kind     = getattr(self, '_active_split_kind', None)
        base_row = getattr(self, '_split_row_base', None)
        if kind not in ('front', 'back') or base_row is None:
            return

        segs_obj = as_segment_list(self.transcription_result.get('segments', []))
        target_index = base_row if kind == 'front' else base_row + 1
        if 0 <= target_index < len(segs_obj):
            tgt = segs_obj[target_index]
            tgt.update({
                'text':     seg.get('text', ''),
                'text_lang1':  seg.get('text_lang1', ''),
                'text_lang2':  seg.get('text_lang2', ''),
                'lang1_prob':  float(seg.get('lang1_prob', 0.0)),
                'lang2_prob':  float(seg.get('lang2_prob', 0.0)),
                'lang1_code':  seg.get('lang1_code', tgt.get('lang1_code', 'ja')),
                'lang2_code':  seg.get('lang2_code', tgt.get('lang2_code', 'ru')),
                'chosen_language': (
                    seg.get('chosen_language') or seg.get('language')
                    or tgt.get('chosen_language')
                ),
            })
            # テキストが両方空なら明示ラベル
            if not tgt.text_lang1 and not tgt.text_lang2:
                tgt.text = tgt.text_lang1 = tgt.text_lang2 = '(空)'
                tgt.lang1_prob = tgt.lang2_prob = 0.0

        try:
            self.transcription_result['segments'] = [s.to_dict() for s in segs_obj]
        except Exception:
            pass

        # 対象行のみ部分更新
        try:
            self._update_table_row(target_index, segs_obj[target_index])
        except Exception:
            self._rebuild_text_and_refresh()

        # 次ジョブへ or 全完了
        if getattr(self, '_pending_rejobs', None):
            self._start_watchdog()  # タイムアウトを延長
            self._run_next_split_rejob()
            return

        self.range_retranscribing = False
        self._rebuild_text_and_refresh()
        self.status_label.setText('分割＆再文字起こし完了')
        self._update_split_button_state()
        self._stop_watchdog()
        self._active_split_kind = None

    @Slot(str)
    def _on_split_rejob_error(self, err: str) -> None:
        """分割後再文字起こしジョブでエラーが発生した場合の回復処理。"""
        self._stop_watchdog()
        self.status_label.setText(f"分割再文字起こし失敗: {err}")
        self.range_retranscribing = False
        if hasattr(self, '_pending_rejobs'):
            self._pending_rejobs.clear()
        self._rebuild_text_and_refresh()
        QMessageBox.critical(self, "エラー", f"分割後再文字起こしでエラーが発生しました:\n{err}")

    def delete_selected_segments(self) -> None:
        """選択行のテキスト列のみ空にして再描画。時間/ID/確率は保持する。"""
        if getattr(self, 'range_retranscribing', False):
            return
        if not getattr(self, 'transcription_result', None):
            return
        segs = self.transcription_result.get('segments', [])
        rows = self._collect_selected_rows()
        if not rows:
            return
        changed = 0
        for r in rows:
            if 0 <= r < len(segs):
                seg = segs[r]
                if any(seg.get(k) for k in ('text', 'text_lang1', 'text_lang2')):
                    seg['text'] = seg['text_lang1'] = seg['text_lang2'] = ''
                    changed += 1
        if changed == 0:
            return
        # テーブルの TEXT 列を直接更新 (全再描画コスト削減)
        try:
            for r in rows:
                if 0 <= r < self.transcription_table.rowCount():
                    item = self.transcription_table.item(r, 4)
                    if item is None:
                        self.transcription_table.setItem(r, 4, QTableWidgetItem(''))
                    else:
                        item.setText('')
        except Exception:
            self._rebuild_text_and_refresh()
        self.status_label.setText(f"{changed}行のテキストを消去しました")
        self._update_split_button_state()

    def retranscribe_selected(self) -> None:
        """選択行を再文字起こし。
        単一行: その区間のみ再推論。
        複数連続行 (30 秒以内): 結合して 1 行プレースホルダへ縮約後に再推論。
        """
        # 操作開始時に一時停止
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                self.media_player.pause()
        except Exception:
            pass

        if not self.transcription_result:
            return
        rows = self._collect_selected_rows()
        if not rows:
            QMessageBox.warning(self, "再文字起こし", "行を選択してください")
            return
        segs = as_segment_list(self.transcription_result.get('segments', []))
        try:
            target = [segs[i] for i in rows]
        except IndexError:
            QMessageBox.critical(self, "エラー", "内部インデックス不整合")
            return

        options = {
            'model':     self.model_combo.currentText(),
            'device':    self._get_selected_device(),
            'lang1_weight': self.lang1_weight_slider.value() / 10.0,
            'lang2_weight': self.lang2_weight_slider.value() / 10.0,
            'lang1': self.lang1_combo.currentData() or 'ja',
            'lang2': self.lang2_combo.currentData(),
        }
        lang1 = self.lang1_combo.currentData() or 'ja'
        lang2 = self.lang2_combo.currentData()
        options['languages'] = [lang1] + ([lang2] if lang2 else [])
        hidden = self.config.get('hidden', {}) if isinstance(self.config, dict) else {}
        for k in ('debug_prob_log',):
            if k in hidden:
                options.setdefault(k, hidden[k])
        start_sec = float(target[0].get('start', 0.0))
        end_sec   = float(target[-1].get('end', start_sec))

        # ---- 単一行モード ----
        if len(rows) == 1:
            idx = rows[0]
            self.status_label.setText(f"単一行再文字起こし中 (行 {idx})…")
            self.progress_bar.setValue(0)
            self.range_retranscribing = True
            self.range_thread = RangeTranscriptionThread(
                self.current_video_path, start_sec, end_sec, options
            )
            self.range_thread.progress.connect(self._on_range_progress)
            self.range_thread.status.connect(self._on_range_status)
            self.range_thread.device_fallback_warning.connect(self._on_device_fallback)

            def single_finished(seg_result: dict) -> None:
                try:
                    if 0 <= idx < len(segs):
                        segs[idx].update({
                            'text':     seg_result.get('text', ''),
                            'text_lang1':  seg_result.get('text_lang1', ''),
                            'text_lang2':  seg_result.get('text_lang2', ''),
                            'lang1_prob':  seg_result.get('lang1_prob', 0.0),
                            'lang2_prob':  seg_result.get('lang2_prob', 0.0),
                            'lang1_code':  seg_result.get('lang1_code', segs[idx].get('lang1_code', 'ja')),
                            'lang2_code':  seg_result.get('lang2_code', segs[idx].get('lang2_code')),
                            'chosen_language': seg_result.get(
                                'chosen_language', segs[idx].get('chosen_language')
                            ),
                        })
                        self.transcription_result['segments'] = [s.to_dict() for s in segs]
                finally:
                    self.range_retranscribing = False
                    self._rebuild_text_and_refresh()
                    self.status_label.setText("単一行再文字起こし完了")

            def single_error(err: str) -> None:
                QMessageBox.critical(self, "再文字起こし失敗", err)
                self.range_retranscribing = False
                self._rebuild_text_and_refresh()
                self.status_label.setText("単一行再文字起こし失敗")

            self.range_thread.range_finished.connect(single_finished)
            self.range_thread.error.connect(single_error)
            self.range_thread.start()
            return

        # ---- 複数行モード: 連続性 & 30 秒以内チェック ----
        merged_list, insert_index, sec_range = merge_contiguous_segments(
            self.transcription_result.get('segments', []), rows
        )
        if merged_list is None:
            QMessageBox.critical(self, "エラー", "連続でない行や不正な範囲が含まれています")
            return
        start_sec, end_sec = sec_range
        if (end_sec - start_sec) > 30.0:
            QMessageBox.critical(self, "エラー", "選択範囲が30秒を超えています")
            return

        # セグメント差し替え (プレースホルダで即時縮約表示)
        self.transcription_result['segments'] = merged_list
        try:
            ph = self.transcription_result['segments'][insert_index]
            ph['text'] = ph['text_lang1'] = ph['text_lang2'] = PLACEHOLDER_PENDING
            ph['lang1_prob'] = ph['lang2_prob'] = 0.0
            ph['chosen_language'] = None
        except Exception:
            pass
        self._rebuild_text_and_refresh()

        self.status_label.setText("再文字起こし準備中...")
        self.progress_bar.setValue(0)
        self.range_retranscribing = True
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.range_thread = RangeTranscriptionThread(
            self.current_video_path, start_sec, end_sec, options
        )
        self.range_thread.progress.connect(self._on_range_progress)
        self.range_thread.status.connect(self._on_range_status)
        orig_segs = segs  # クロージャでキャプチャ
        self.range_thread.range_finished.connect(
            lambda seg, rows_=rows, orig_=orig_segs: self._on_range_finished(seg, rows_, orig_)
        )
        self.range_thread.error.connect(self._on_range_error)
        self.range_thread.device_fallback_warning.connect(self._on_device_fallback)
        self.range_thread.start()

    @Slot(int)
    def _on_range_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    @Slot(str)
    def _on_range_status(self, message: str) -> None:
        self.status_label.setText(message)
        self._append_log(message)

    def _on_range_finished(
        self, new_seg: dict, rows_sorted: list[int], orig_segs: list
    ) -> None:
        QApplication.restoreOverrideCursor()
        self.range_retranscribing = False

        first = rows_sorted[0]
        new_segments: list[dict] = []
        for idx, s in enumerate(orig_segs):
            if idx == first:
                # BUG-2 修正: Segment オブジェクトか dict かどちらでも安全に取得
                s_dict = s.to_dict() if hasattr(s, 'to_dict') else s
                new_segments.append({
                    'start': new_seg['start'], 'end': new_seg['end'],
                    'text': new_seg['text'],
                    'text_lang1': new_seg.get('text_lang1', ''),
                    'text_lang2': new_seg.get('text_lang2', ''),
                    'chosen_language': new_seg['chosen_language'],
                    'id': s_dict.get('id', idx),
                    'lang1_prob': new_seg.get('lang1_prob', 0.0),
                    'lang2_prob': new_seg.get('lang2_prob', 0.0),
                    'lang1_code': new_seg.get('lang1_code',
                        s_dict.get('lang1_code', self.lang1_combo.currentData() or 'ja')),
                    'lang2_code': new_seg.get('lang2_code',
                        s_dict.get('lang2_code', self.lang2_combo.currentData())),
                })
            elif idx in rows_sorted[1:]:
                continue  # 縮約されたので除去
            else:
                new_segments.append(s.to_dict() if hasattr(s, 'to_dict') else s)

        self.transcription_result['segments'] = new_segments
        self._rebuild_text_and_refresh()
        self.status_label.setText("再文字起こし完了")

    def _on_range_error(self, err: str) -> None:
        QApplication.restoreOverrideCursor()
        self.range_retranscribing = False
        self.status_label.setText("再文字起こし失敗")
        QMessageBox.critical(self, "再文字起こし失敗", err)

    # ================================================================
    # 無音設定 & 言語強制再認識
    # ================================================================

    def set_segments_silence(self, rows: list[int]) -> None:
        """選択された行を無音セグメントに設定する。"""
        if not self.transcription_result:
            return
        if not rows:
            return
        
        segs = self.transcription_result.get('segments', [])
        for r in rows:
            if 0 <= r < len(segs):
                segs[r]['text'] = ''
                segs[r]['text_lang1'] = ''
                segs[r]['text_lang2'] = ''
                segs[r]['chosen_language'] = 'silence'
                segs[r]['lang1_prob'] = 0.0
                segs[r]['lang2_prob'] = 0.0
        
        self._rebuild_text_and_refresh()
        self.status_label.setText(f"{len(rows)}行を無音に設定しました")

    def retranscribe_as_language(self, rows: list[int], lang: str) -> None:
        """選択された行を指定言語で強制再認識する。
        
        Args:
            rows: 対象行のインデックスリスト
            lang: 'ja' または 'ru'
        """
        if not self.transcription_result:
            return
        if not rows:
            return
        if not self.current_video_path:
            QMessageBox.warning(self, "再認識", "動画ファイルが読み込まれていません")
            return
        
        # 操作開始時に一時停止
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                self.media_player.pause()
        except Exception:
            pass
        
        segs = as_segment_list(self.transcription_result.get('segments', []))
        
        # オプション準備 (合計1.0になるように設定)
        options = {
            'model': self.model_combo.currentText(),
            'device': self._get_selected_device(),
            'lang1_weight': 0.999 if lang == self.lang1_combo.currentData() else 0.001,
            'lang2_weight': 0.001 if lang == self.lang1_combo.currentData() else 0.999,
        }
        options['languages'] = [lang]
        hidden = self.config.get('hidden', {}) if isinstance(self.config, dict) else {}
        for k in ('debug_prob_log',):
            if k in hidden:
                options.setdefault(k, hidden[k])
        
        # 各行を個別に再認識
        self.range_retranscribing = True
        self._retranscribe_rows_as_language(rows, segs, lang, options, 0)

    def _retranscribe_rows_as_language(
        self, rows: list[int], segs: list, lang: str, options: dict, current_idx: int
    ) -> None:
        """rows内の各行を順次再認識する（再帰的処理）。"""
        if current_idx >= len(rows):
            # 全て完了
            self.range_retranscribing = False
            self._rebuild_text_and_refresh()
            lang_name = "日本語" if lang == 'ja' else "ロシア語"
            self.status_label.setText(f"{len(rows)}行を{lang_name}で再認識完了")
            return
        
        row_idx = rows[current_idx]
        if not (0 <= row_idx < len(segs)):
            # 次へ
            self._retranscribe_rows_as_language(rows, segs, lang, options, current_idx + 1)
            return
        
        seg = segs[row_idx]
        start_sec = float(seg.get('start', 0.0))
        end_sec = float(seg.get('end', start_sec))
        
        lang_name = "日本語" if lang == 'ja' else "ロシア語"
        self.status_label.setText(
            f"{lang_name}で再認識中 ({current_idx + 1}/{len(rows)})…"
        )
        self.progress_bar.setValue(int(100 * current_idx / len(rows)))
        
        self.range_thread = RangeTranscriptionThread(
            self.current_video_path, start_sec, end_sec, options
        )
        self.range_thread.progress.connect(self._on_range_progress)
        self.range_thread.status.connect(self._on_range_status)
        self.range_thread.device_fallback_warning.connect(self._on_device_fallback)
        
        def on_finished(result: dict) -> None:
            try:
                # 結果を反映（確率を100%に強制設定）
                if lang == self.lang1_combo.currentData():
                    segs[row_idx]['lang1_prob'] = 100.0
                    segs[row_idx]['lang2_prob'] = 0.0
                else:
                    segs[row_idx]['lang1_prob'] = 0.0
                    segs[row_idx]['lang2_prob'] = 100.0
                
                segs[row_idx]['text'] = result.get('text', '')
                segs[row_idx]['text_lang1'] = result.get('text_lang1', '')
                segs[row_idx]['text_lang2'] = result.get('text_lang2', '')
                segs[row_idx]['chosen_language'] = lang
                
                self.transcription_result['segments'] = [s.to_dict() for s in segs]
            finally:
                # 次の行へ
                self._retranscribe_rows_as_language(rows, segs, lang, options, current_idx + 1)
        
        def on_error(err: str) -> None:
            QMessageBox.warning(
                self, "再認識エラー", 
                f"行 {row_idx} の再認識に失敗しました: {err}\n\n次の行に進みます。"
            )
            # 次の行へ
            self._retranscribe_rows_as_language(rows, segs, lang, options, current_idx + 1)
        
        self.range_thread.range_finished.connect(on_finished)
        self.range_thread.error.connect(on_error)
        self.range_thread.start()

    # ================================================================
    # 書き出し
    # ================================================================

    def partial_export_selected(self) -> None:
        """選択行ごとに音声とテキストを指定フォルダへ書き出す。"""
        if not self.transcription_result or not self.current_video_path:
            QMessageBox.warning(self, "部分書き出し", "書き出し対象がありません")
            return
        rows = self._collect_selected_rows()
        if not rows:
            QMessageBox.warning(self, "部分書き出し", "行を選択してください")
            return
        segs = self.transcription_result.get('segments', [])
        try:
            target = [segs[i] for i in rows]
        except IndexError:
            QMessageBox.critical(self, "部分書き出し", "内部インデックス不整合")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "出力フォルダを選択",
            os.path.dirname(self.current_video_path),
        )
        if not out_dir:
            return

        format_items = [
            ("TXT + WAV", "txt", "wav"),
            ("SRT + WAV", "srt", "wav"),
            ("JSON + WAV", "json", "wav"),
            ("TXT + MP4", "txt", "mp4"),
            ("SRT + MP4", "srt", "mp4"),
            ("JSON + MP4", "json", "mp4"),
        ]
        fmt_labels = [label for label, _, _ in format_items]
        fmt_label, ok = QInputDialog.getItem(
            self,
            "部分書き出し",
            "出力形式を選択",
            fmt_labels,
            0,
            False,
        )
        if not ok:
            return
        sel_idx = fmt_labels.index(fmt_label)
        text_fmt, audio_fmt = format_items[sel_idx][1], format_items[sel_idx][2]

        def safe_filename(name: str) -> str:
            cleaned = re.sub(r"[\\/:*?\"<>|]", "_", name)
            cleaned = cleaned.strip().strip(".")
            return cleaned or "segment"

        used_names: dict[str, int] = {}
        errors: list[str] = []
        saved = 0
        for i, seg in enumerate(target):
            start_sec = float(seg.get('start', 0.0))
            end_sec = float(seg.get('end', start_sec))
            if end_sec <= start_sec:
                errors.append(f"行 {rows[i]}: 時間範囲が不正")
                continue

            text_name = display_text(seg)
            if not text_name or text_name == "[無音]":
                continue
            base_name = safe_filename(text_name)
            used_names[base_name] = used_names.get(base_name, 0) + 1
            if used_names[base_name] > 1:
                base_name = f"{base_name}_{used_names[base_name]}"

            audio_path = os.path.join(out_dir, f"{base_name}.{audio_fmt}")
            text_path = os.path.join(out_dir, f"{base_name}.{text_fmt}")
            if os.path.exists(audio_path) or os.path.exists(text_path):
                suffix = used_names[base_name] + 1
                while True:
                    candidate = f"{base_name}_{suffix}"
                    audio_path = os.path.join(out_dir, f"{candidate}.{audio_fmt}")
                    text_path = os.path.join(out_dir, f"{candidate}.{text_fmt}")
                    if not (os.path.exists(audio_path) or os.path.exists(text_path)):
                        base_name = candidate
                        break
                    suffix += 1

            try:
                if audio_fmt == "wav":
                    subprocess.run(
                        ['ffmpeg', '-y', '-i', self.current_video_path,
                         '-ss', f"{start_sec:.3f}", '-to', f"{end_sec:.3f}",
                         '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                    )
                else:
                    subprocess.run(
                        ['ffmpeg', '-y', '-i', self.current_video_path,
                         '-ss', f"{start_sec:.3f}", '-to', f"{end_sec:.3f}",
                         '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                    )
            except subprocess.CalledProcessError:
                errors.append(f"{base_name}: 音声抽出に失敗しました (ffmpeg)")
                continue

            try:
                seg_result = {'segments': [seg]}
                if text_fmt == "json":
                    payload = build_json_payload(
                        seg_result,
                        {
                            'video_path': self.current_video_path,
                            'model': self.model_combo.currentText(),
                            'device': self._get_selected_device(),
                        },
                    )
                    with open(text_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                else:
                    text_out = build_export_text(seg_result, text_fmt)
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text_out)
            except Exception as e:
                errors.append(f"{base_name}: テキスト保存失敗: {e}")
                continue

            saved += 1

        msg = (
            "部分書き出しが完了しました\n"
            "--------------------------------\n"
            f" 出力先   : {out_dir}\n"
            f" 保存件数 : {saved}\n"
            f" 失敗件数 : {len(errors)}\n"
            "--------------------------------"
        )
        self.status_label.setText("部分書き出し完了")
        self._append_log(msg)
        if errors:
            detail = "\n".join(errors[:10])
            QMessageBox.warning(self, "部分書き出し", msg + "\n\n" + detail)
        else:
            QMessageBox.information(self, "部分書き出し", msg)

    def export_transcription(self) -> None:
        """認識結果を JSON / TXT / SRT ファイルへ書き出す。"""
        if not getattr(self, 'transcription_result', None):
            self.status_label.setText("書き出し対象がありません")
            return
        if self.current_video_path:
            base = os.path.splitext(os.path.basename(self.current_video_path))[0]
            default_name = f"{base}.json"
        else:
            default_name = "transcription.json"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "書き出しファイルを保存", default_name,
            "JSON (*.json);;テキスト (*.txt);;SRT 字幕 (*.srt)"
        )
        if not file_path:
            return

        low = file_path.lower()
        if selected_filter.startswith("SRT") or low.endswith('.srt'):
            fmt = 'srt'
        elif selected_filter.startswith("テキスト") or low.endswith('.txt'):
            fmt = 'txt'
        else:
            fmt = 'json'

        try:
            if fmt == 'json':
                if not low.endswith('.json'):
                    file_path += '.json'
                payload = build_json_payload(
                    self.transcription_result,
                    {
                        'video_path': self.current_video_path,
                        'model':  self.model_combo.currentText(),
                        'device': self._get_selected_device(),
                    }
                )
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            else:
                text = build_export_text(self.transcription_result, fmt)
                if fmt == 'txt' and not low.endswith('.txt'):
                    file_path += '.txt'
                elif fmt == 'srt' and not low.endswith('.srt'):
                    file_path += '.srt'
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            self.status_label.setText(f"書き出し完了: {file_path}")
        except Exception as e:
            logger.exception("Export failed")
            QMessageBox.critical(self, "書き出しエラー", str(e))
            self.status_label.setText("書き出し失敗")

    # ================================================================
    # 上級者向け設定
    # ================================================================

    def _open_hidden_params_dialog(self) -> None:
        """上級者向け設定ダイアログを開き、設定を config.toml の [advanced] に保存する。"""
        # provide advanced (preferred) or fall back to legacy hidden section
        current_hidden = self.config.get("advanced", self.config.get("hidden", {}))
        
        dialog = HiddenParamsDialog(current_hidden, self)
        if dialog.exec() == HiddenParamsDialog.Accepted:
            new_values = dialog.get_values()
            
            # config.toml を更新 (tomlkit使用で堅牢化)
            try:
                import tomlkit
                
                cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
                cfg_path = os.path.normpath(cfg_path)
                
                # 既存のconfig.tomlを読み込み (コメント・書式を保持)
                with open(cfg_path, "r", encoding="utf-8") as f:
                    doc = tomlkit.load(f)
                
                # ensure [advanced] exists (preferred)
                if "advanced" not in doc:
                    doc["advanced"] = tomlkit.table()

                # ensure nested [advanced.logging] exists for logging-related keys
                if "logging" not in doc["advanced"] or not isinstance(doc["advanced"].get("logging"), dict):
                    doc["advanced"]["logging"] = tomlkit.table()

                # Map logging-related flat keys into the nested logging table
                LOG_KEY_MAP = {
                    "log_file_enabled": "file_enabled",
                    "log_level": "level",
                    "log_file_path": "file_path",
                    "log_max_bytes": "max_bytes",
                    "log_backup_count": "backup_count",
                }

                # 各パラメータを更新
                for key, val in new_values.items():
                    if key in LOG_KEY_MAP:
                        doc["advanced"]["logging"][LOG_KEY_MAP[key]] = val
                    else:
                        doc["advanced"][key] = val
                
                # ファイルに書き戻し (書式・コメント保持)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    tomlkit.dump(doc, f)
                
                # メモリ上のconfigも更新
                with open(cfg_path, "rb") as f:
                    self.config = _toml.load(f)
                
                QMessageBox.information(
                    self,
                    "保存完了",
                    "上級者向け設定を config.toml の [advanced] セクションに保存しました。\n次回の文字起こしから反映されます。"
                )
                logger.info(f"[ADVANCED] Updated advanced params: {new_values}")
                
            except Exception as e:
                logger.exception("Failed to save hidden params")
                QMessageBox.critical(
                    self,
                    "保存エラー",
                    f"上級者向け設定の保存に失敗しました: {e}"
                )
    
    def _toml_value(self, val) -> str:
        """Python値をTOML形式文字列に変換。"""
        if isinstance(val, bool):
            return "true" if val else "false"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, list):
            return "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in val) + "]"
        else:
            return str(val)

    # ================================================================
    # 文字起こし開始 / 制御
    # ================================================================

    def start_transcription(self) -> None:
        """設定値を収集して TranscriptionThread を起動する。"""
        if self.transcription_table.rowCount() > 0:
            resp = QMessageBox.question(
                self,
                "文字起こし開始",
                "現在の結果はリセットされます。続行しますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
            # OK直後にテキストをリセット
            self.transcription_table.setRowCount(0)
            self.transcription_result = {'text': '', 'segments': []}

        # 言語選択: lang1 は必須, lang2 は None ("なし") 時は单言語モード
        lang1 = self.lang1_combo.currentData()
        lang2 = self.lang2_combo.currentData()  # None if "なし"
        selected_langs = [lang1] + ([lang2] if lang2 else [])

        # バリデーション: lang1 と lang2 が同じになっていないか確認
        if lang2 is not None and lang1 == lang2:
            QMessageBox.critical(
                self,
                "言語設定エラー",
                "言語1と言語2に同じ言語が選択されています。別の言語を選択してください。",
            )
            return

        # 詳細設定ウィジェット値を収集
        detail_values: dict = {}
        for key, ctrl in self.detail_controls.items():
            if isinstance(ctrl, QCheckBox):
                detail_values[key] = ctrl.isChecked()
            elif isinstance(ctrl, (QSpinBox, QDoubleSpinBox)):
                detail_values[key] = ctrl.value()
            elif isinstance(ctrl, QLineEdit):
                detail_values[key] = ctrl.text()
            elif isinstance(ctrl, QTextEdit):
                detail_values[key] = ctrl.toPlainText()
            elif isinstance(ctrl, QComboBox):
                if key == "vad_level":
                    data = ctrl.currentData()
                    detail_values[key] = int(data) if data is not None else 2
                else:
                    detail_values[key] = ctrl.currentText()

        # VAD は常に有効 (ループ外へ移動 BUG-15付近の修正)
        detail_values["vad_filter"] = True

        options: dict = {
            "model":    self.model_combo.currentText(),
            "device":   self._get_selected_device(),
            "languages": selected_langs,
            "lang1_weight": self.lang1_weight_slider.value() / 10.0,
            "lang2_weight": self.lang2_weight_slider.value() / 10.0,
        }
        options.update(detail_values)

        # [hidden] セクションのパラメータを注入
        hidden = self.config.get('hidden', {}) if isinstance(self.config, dict) else {}
        for k in ('ambiguous_threshold', 'condition_on_previous_text',
                  'compression_ratio_threshold', 'log_prob_threshold',
                  'repetition_penalty', 'speech_pad_ms', 'duplicate_merge',
                  'phase1_beam_size', 'phase2_detect_beam_size', 'phase2_retranscribe_beam_size',
                  'debug_prob_log',):
            if k in hidden:
                options.setdefault(k, hidden[k])

        # UI をロック
        self.transcribe_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("文字起こし開始準備中...")
        self._cancelled_during_transcribe = False

        # 逐次追加用に結果初期化
        self.transcription_result = {'text': '', 'segments': []}

        self.transcription_thread = TranscriptionThread(self.current_video_path, options)
        self.transcription_thread.progress.connect(self._update_progress)
        self.transcription_thread.status.connect(self._update_status)
        self.transcription_thread.segment_ready.connect(self._on_segment_ready)
        self.transcription_thread.finished_transcription.connect(self._on_transcription_finished)
        self.transcription_thread.error.connect(self._on_transcription_error)
        self.transcription_thread.device_fallback_warning.connect(self._on_device_fallback)
        self.transcription_thread.start()

    def cancel_transcription(self) -> None:
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.status_label.setText("キャンセル要求送信中...")
            try:
                self.transcription_thread.request_cancel()
                self.cancel_button.setEnabled(False)
            except Exception as e:
                logger.error(f"Cancel failed: {e}")
            # 逐次表示を即クリア
            try:
                self.transcription_table.setRowCount(0)
            except Exception:
                pass
            self.transcription_result = {'text': '', 'segments': []}
            self._cancelled_during_transcribe = True

    @Slot(int)
    def _update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    @Slot(str)
    def _update_status(self, message: str) -> None:
        self.status_label.setText(message)
        self._append_log(message)

    @Slot(dict)
    def _on_transcription_finished(self, result: dict) -> None:
        self.transcription_result = result
        self._rebuild_text_and_refresh()
        self.transcribe_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.partial_export_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("文字起こし完了")

    @Slot(dict)
    def _on_segment_ready(self, seg_dict: dict) -> None:
        """逐次セグメント受信: テーブルへ 1 行追加。"""
        try:
            thr = getattr(self, 'transcription_thread', None)
            if thr is None or not thr.isRunning():
                return
            if getattr(self, '_cancelled_during_transcribe', False):
                return
            segs = self.transcription_result.get('segments', []) if self.transcription_result else []
            segs.append(seg_dict)
            line = display_text(seg_dict)  # BUG-15 修正: display_text() を使用
            if line:
                prev_text = self.transcription_result.get('text', '') if self.transcription_result else ''
                self.transcription_result['text'] = (
                    prev_text + ('\n' if prev_text else '') + line
                )
            row = self.transcription_table.rowCount()
            self.transcription_table.insertRow(row)
            st_str = format_ms(int(seg_dict.get('start', 0.0) * 1000))
            ed_str = format_ms(int(seg_dict.get('end',   0.0) * 1000))
            start_item = QTableWidgetItem(st_str)
            end_item = QTableWidgetItem(ed_str)
            p1 = seg_dict.get('lang1_prob', 0.0)
            p2 = seg_dict.get('lang2_prob', 0.0)
            item1 = QTableWidgetItem(f"{p1:.2f}")
            item2 = QTableWidgetItem(f"{p2:.2f}")
            text_item = QTableWidgetItem(line)  # BUG-15: display_text result
            for it in (start_item, end_item, item1, item2):
                it.setTextAlignment(Qt.AlignCenter)
            apply_prob_colors(self.transcription_table, item1, item2, p1, p2)
            self.transcription_table.setItem(row, 0, start_item)
            self.transcription_table.setItem(row, 1, end_item)
            self.transcription_table.setItem(row, 2, item1)
            self.transcription_table.setItem(row, 3, item2)
            self.transcription_table.setItem(row, 4, text_item)
            self.status_label.setText(f"文字起こし中... ({len(segs)})")
        except Exception:
            pass

    @Slot(str)
    def _on_transcription_error(self, error_message: str) -> None:
        logger.error(f"Transcription error: {error_message}")
        self.status_label.setText(f"エラー: {error_message}")
        self.transcribe_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(bool(getattr(self, 'transcription_result', None)))
        self.partial_export_button.setEnabled(bool(getattr(self, 'transcription_result', None)))
        self.progress_bar.setValue(0)

    @Slot()
    def _on_device_fallback(self) -> None:
        """GPU→CPUフォールバック警告"""
        from PySide6.QtWidgets import QMessageBox
        
        QMessageBox.warning(
            self,
            "デバイスフォールバック警告",
            "GPUが利用できないため、CPUモードで処理を実行しました。\n\n"
            "CPUモードでは処理速度が大幅に低下します。\n"
            "GPUを使用するには、CUDA Toolkit (NVIDIA の場合) または ROCm SDK (AMD の場合) "
            "が正しくインストールされていることを確認してください。"
        )

    # ================================================================
    # ファイル操作
    # ================================================================

    def open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "動画ファイルを選択", "",
            "動画ファイル (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;すべてのファイル (*.*)",
        )
        if file_path:
            self.current_video_path = file_path
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.transcribe_button.setEnabled(True)
            for attr in ('seek_back_10_btn', 'seek_back_1_btn',
                         'seek_fwd_1_btn', 'seek_fwd_10_btn'):
                getattr(self, attr).setEnabled(True)
            self.setWindowTitle(
                f"動画文字起こしエディタ - {os.path.basename(file_path)}"
            )

    # ================================================================
    # UI イベント
    # ================================================================

    def play_pause(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop_video(self) -> None:
        self.media_player.stop()

    def seek_relative(self, delta_ms: int) -> None:
        """現在位置から delta_ms だけ相対シーク (境界クリップ)。"""
        try:
            cur = self.media_player.position()
            dur = self.media_player.duration() or 0
            new_pos = max(0, cur + delta_ms)
            if dur > 0:
                new_pos = min(new_pos, dur)
            self.media_player.setPosition(new_pos)
        except Exception:
            pass

    def _media_state_changed(self, state) -> None:
        icon = (QStyle.SP_MediaPause
                if state == QMediaPlayer.PlayingState
                else QStyle.SP_MediaPlay)
        self.play_button.setIcon(self.style().standardIcon(icon))

    def _position_changed(self, position: int) -> None:
        self.position_slider.setValue(position)
        self.current_time_label.setText(format_ms(position))
        # 自動同期: チェック ON かつ再生中のみ
        if not self.auto_sync_check.isChecked():
            return
        if self.media_player.playbackState() != QMediaPlayer.PlayingState:
            return
        try:
            sel = self.transcription_table.selectionModel()
            if sel and len(sel.selectedRows()) <= 1:
                self.sync_table_selection_with_position(position)
        except Exception:
            pass

    def _duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, duration)
        self.total_time_label.setText(format_ms(duration))

    def _set_position(self, position: int) -> None:
        self.media_player.setPosition(position)

    def _update_table_headers(self) -> None:
        """言語設定変更時にテーブルヘッダーのコードを更新する。"""
        lang1 = self.lang1_combo.currentData() or 'LANG1'
        lang2 = self.lang2_combo.currentData()
        lbl2 = f"{lang2.upper()}%" if lang2 else "LANG2%"
        self.transcription_table.setHorizontalHeaderLabels(
            ["START", "END", f"{lang1.upper()}%", lbl2, "TEXT"]
        )

    def _on_auto_sync_toggled(self, checked: bool) -> None:
        """自動同期 ON 時: 再生中なら即座に現在位置で同期する。"""
        if not checked:
            return
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                pos = self.media_player.position()
                sel = self.transcription_table.selectionModel()
                if sel and len(sel.selectedRows()) <= 1:
                    self.sync_table_selection_with_position(pos)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        """Ensure background transcription threads are stopped and models
        are unloaded before the application exits. Shows a small waiting
        dialog while shutdown proceeds.
        """
        # Collect running transcription threads
        threads = []
        if getattr(self, 'transcription_thread', None) and self.transcription_thread.isRunning():
            threads.append(self.transcription_thread)
        if getattr(self, 'range_thread', None) and self.range_thread.isRunning():
            threads.append(self.range_thread)

        if threads:
            from PySide6.QtWidgets import QMessageBox
            dlg = QMessageBox(self)
            dlg.setWindowTitle('終了処理中')
            dlg.setText('処理を終了しています。しばらくお待ちください...')
            dlg.setStandardButtons(QMessageBox.NoButton)
            dlg.setModal(True)
            dlg.show()

            # Request cancel on threads
            for t in threads:
                try:
                    if hasattr(t, 'request_cancel'):
                        t.request_cancel()
                except Exception:
                    pass

            # Wait up to 20s for threads to finish, checking periodically
            total_wait = 0
            timeout_ms = 20000
            interval = 250
            while any(t.isRunning() for t in threads) and total_wait < timeout_ms:
                for t in threads:
                    try:
                        t.wait(interval)
                    except Exception:
                        pass
                QApplication.processEvents()
                total_wait += interval

            dlg.close()

        # Clear model cache now that threads are stopped
        try:
            import transcription.model_cache as model_cache
            model_cache.clear_cache()
        except Exception:
            logger.exception('Failed to clear model cache during shutdown')

        super().closeEvent(event)

    def _on_device_changed(self, *_args) -> None:
        """デバイス変更時の処理"""
        device = self._get_selected_device()
        is_cpu = device == "cpu"
        self.cpu_warning_label.setVisible(is_cpu)
        
        # CPUモードに変更した場合は警告ダイアログを表示（初期化中は除く）
        if is_cpu and not self._initializing:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "CPUモード警告",
                "CPUモードが選択されました。\n\n"
                "CPUモードでは処理速度が大幅に低下します。\n"
                "可能であればCUDAを使用することを推奨します。"
            )



    def display_transcription(self, result: dict) -> None:
        """テーブルにセグメントを表示する。"""
        if not result:
            return
        populate_table(self.transcription_table, result)
