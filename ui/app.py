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
import logging
import os
import subprocess
import sys
import time
import tomllib as _toml

import torch
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
    PLACEHOLDER_PENDING,
    DEFAULT_WATCHDOG_TIMEOUT_MS,
    MIN_SEGMENT_DUR,
    MAX_RANGE_SEC,
)
from core.exporter import build_export_text, build_json_payload
from core.logging_config import setup_logging, get_logger
from models.segment import Segment, as_segment_list
from services.retranscribe_ops import dynamic_time_split, merge_contiguous_segments
from services.segment_ops import split_segment_at_position, adjust_boundary
from transcription.threads import TranscriptionThread, RangeTranscriptionThread
from ui.edit_dialog import EditDialog
from ui.table_presenter import populate_table, rebuild_aggregate_text
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

        # 設定ロード
        try:
            self.config = self._load_config()
        except Exception as e:
            logger.error(f"設定ロード失敗: {e}")
            self.config = {"default": {}}

        # UI 初期化とシグナル接続
        self._init_ui()
        self._connect_signals()

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
            log_conf = data.get('logging') if isinstance(data, dict) else None
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

        # ---- ヘルプテキスト (ツールチップ用) ----
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
            "ja_weight": (
                "日本語言語判定スコアへの補正係数。\n"
                "最終確率 = 元スコア × weight を正規化後に比較。\n"
                "1.0 = 補正なし / >1.0 で日本語優遇 / <1.0 で抑制。"
            ),
            "ru_weight": (
                "ロシア語判定スコア補正係数 (ja_weight と同様の計算)。"
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
                "WebRTC VAD (Voice Activity Detection) 感度。\n"
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

        button_layout = QHBoxLayout()
        self.open_button = QPushButton("動画を開く")
        self.open_button.clicked.connect(self.open_file)
        button_layout.addWidget(self.open_button)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_pause)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.play_button)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        SEEK_BTN_WIDTH = 52

        def make_seek_btn(label: str, delta_ms: int, tooltip: str) -> QPushButton:
            btn = QPushButton(label)
            btn.setEnabled(False)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _=None, d=delta_ms: self.seek_relative(d))
            btn.setFixedWidth(SEEK_BTN_WIDTH)
            return btn

        self.seek_back_10_btn = make_seek_btn("◀10s", -10000, "10秒戻る")
        self.seek_back_1_btn  = make_seek_btn("◀1s",  -1000,  "1秒戻る")
        self.seek_fwd_1_btn   = make_seek_btn("1s▶",   1000,  "1秒進む")
        self.seek_fwd_10_btn  = make_seek_btn("10s▶", 10000,  "10秒進む")
        for btn in (self.seek_back_10_btn, self.seek_back_1_btn,
                    self.seek_fwd_1_btn, self.seek_fwd_10_btn):
            button_layout.addWidget(btn)
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        video_vlayout.addLayout(control_layout)

        # ---- テーブルコンテナ ----
        table_container = QWidget()
        table_vlayout = QVBoxLayout(table_container)
        table_vlayout.setContentsMargins(0, 0, 0, 0)
        table_vlayout.setSpacing(4)

        top_export_bar = QHBoxLayout()
        top_export_bar.setContentsMargins(0, 0, 0, 0)
        top_export_bar.addStretch()
        self.partial_export_button = QPushButton("部分書き出し")
        self.partial_export_button.setEnabled(False)
        self.partial_export_button.setToolTip(
            "選択された行範囲の音声(WAV)とテキスト(TXT)を ./output に書き出し"
        )
        self.partial_export_button.clicked.connect(self.partial_export_selected)
        top_export_bar.addWidget(self.partial_export_button)
        self.export_button = QPushButton("全文書き出し...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_transcription)
        top_export_bar.addWidget(self.export_button)
        table_vlayout.addLayout(top_export_bar)

        self.transcription_table = QTableWidget()
        self.transcription_table.setStyleSheet("""
            QTableWidget::item:selected { background: #1976d2; color: #ffffff; }
            QTableWidget::item:focus    { outline: none; }
            QTableWidget::item          { padding-top: 1px; padding-bottom: 1px; }
        """)
        self.transcription_table.setColumnCount(5)
        self.transcription_table.setHorizontalHeaderLabels(
            ["START", "END", "JA%", "RU%", "TEXT"]
        )
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
        fixed_settings_width = 350
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
        self.profiles = [k for k, v in self.config.items() if isinstance(v, dict)]
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
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        self.device_combo.addItems(devices)
        base_prof = self.config.get(self.current_profile_name, self.config.get('default', {}))
        default_device = base_prof.get("device", "cuda")
        if default_device not in devices:
            devices.append(default_device)
            self.device_combo.addItem(default_device)
        self.device_combo.setCurrentText(default_device)
        model_layout.addWidget(self.device_combo, 1, 1)
        model_layout.addWidget(help_label("デバイス"), 1, 2)

        model_layout.addWidget(QLabel("トランスクリプションモデル:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        default_model = base_prof.get("transcription_model", "large-v3")
        if default_model not in WHISPER_MODELS:
            self.model_combo.addItem(default_model)
        self.model_combo.setCurrentText(default_model)
        model_layout.addWidget(self.model_combo, 2, 1)
        model_layout.addWidget(help_label("トランスクリプションモデル"), 2, 2)

        model_layout.addWidget(QLabel("セグメンテーションモデル:"), 3, 0)
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.addItems(WHISPER_MODELS)
        default_seg_model = base_prof.get("segmentation_model", "turbo")
        if default_seg_model not in WHISPER_MODELS:
            self.segmentation_model_combo.addItem(default_seg_model)
        self.segmentation_model_combo.setCurrentText(default_seg_model)
        model_layout.addWidget(self.segmentation_model_combo, 3, 1)
        model_layout.addWidget(help_label("セグメンテーションモデル"), 3, 2)

        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)

        # ---- 言語設定 ----
        lang_group  = QGroupBox("言語設定")
        lang_layout = QVBoxLayout()

        def make_lang_row(code: str, label_text: str, weight_key: str, prof: dict):
            dlangs = prof.get("default_languages", ["ja", "ru"])
            if not isinstance(dlangs, list) or not dlangs:
                dlangs = ["ja"]
            row = QWidget()
            hl  = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            chk = QCheckBox(label_text)
            chk.setChecked(code in dlangs)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0); slider.setMaximum(300); slider.setSingleStep(5)
            slider.setFixedWidth(140)
            val = prof.get(weight_key, 1.0)
            if not isinstance(val, (int, float)):
                val = 1.0
            slider.setValue(int(round(val * 100)))
            value_label = QLabel(f"{val:.2f}")
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v/100:.2f}"))
            hl.addWidget(chk)
            hl.addWidget(QLabel("weight"))
            hl.addWidget(slider)
            hl.addWidget(value_label)
            hl.addWidget(help_label(weight_key))
            return chk, slider, value_label, row

        self.ja_check, self.ja_weight_slider, self.ja_weight_value_label, ja_row = \
            make_lang_row("ja", "JA", "ja_weight", base_prof)
        self.ru_check, self.ru_weight_slider, self.ru_weight_value_label, ru_row = \
            make_lang_row("ru", "RU", "ru_weight", base_prof)
        lang_layout.addWidget(ja_row)
        lang_layout.addWidget(ru_row)
        lang_group.setLayout(lang_layout)
        scroll_layout.addWidget(lang_group)

        # ---- プロファイル適用ロジック ----
        def apply_profile(name: str) -> None:
            if name not in self.config:
                return
            prof = self.config[name]
            self.current_profile_name = name

            dev = prof.get("device")
            if dev and dev in [self.device_combo.itemText(i)
                                for i in range(self.device_combo.count())]:
                self.device_combo.setCurrentText(dev)

            for combo, key in (
                (self.model_combo, "transcription_model"),
                (self.segmentation_model_combo, "segmentation_model"),
            ):
                val = prof.get(key)
                if val and val in [combo.itemText(i) for i in range(combo.count())]:
                    combo.setCurrentText(val)

            dlangs = prof.get("default_languages", [])
            if self.ja_check:
                self.ja_check.setChecked("ja" in dlangs or not dlangs)
            if self.ru_check:
                self.ru_check.setChecked("ru" in dlangs)

            for slider, key in (
                (self.ja_weight_slider, "ja_weight"),
                (self.ru_weight_slider, "ru_weight"),
            ):
                w = prof.get(key, 1.0)
                if isinstance(w, (int, float)):
                    slider.setValue(int(round(w * 100)))

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
        self.detail_controls: dict[str, QWidget] = {}

        # 基本設定グループで使用済みのキーは除外
        exclude_keys = {
            "device", "transcription_model", "segmentation_model",
            "default_languages", "ja_weight", "ru_weight",
            # 廃止済みオプション
            "dual_transcribe_all", "merge_refine", "enable_temp_fallback",
        }

        for key, value in base_prof.items():
            if key in exclude_keys:
                continue

            # initial_prompt は複数行テキストエリア
            if key == "initial_prompt":
                detail_layout.addWidget(QLabel("initial_prompt:"))
                txt = QTextEdit()
                txt.setMaximumHeight(100)
                txt.setPlainText(value or "")
                detail_layout.addWidget(txt)
                self.detail_controls[key] = txt
                continue

            row_container = QWidget()
            row_h = QHBoxLayout(row_container)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.addWidget(QLabel(f"{key}:"))

            ctrl: QWidget | None = None
            if isinstance(value, bool):
                ctrl = QCheckBox()
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
                    if key == "srt_max_line":
                        ctrl.setRange(1, 1000)
                    ctrl.setValue(value)
            elif isinstance(value, float):
                ctrl = QDoubleSpinBox()
                ctrl.setDecimals(4)
                ctrl.setRange(-1e9, 1e9)
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
        right_layout.addWidget(scroll_area)

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
        transcribe_layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        transcribe_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("待機中")
        self.status_label.setMinimumHeight(22)
        transcribe_layout.addWidget(self.status_label)

        # 折りたたみ可能ログパネル
        self.log_panel_container = QWidget()
        log_layout = QVBoxLayout(self.log_panel_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        toggle_row = QHBoxLayout()
        self.toggle_log_button = QPushButton("▼ ログ表示")
        self.toggle_log_button.setCheckable(True)
        self.toggle_log_button.setChecked(False)
        self.toggle_log_button.toggled.connect(self._toggle_log_panel)
        toggle_row.addWidget(self.toggle_log_button)
        toggle_row.addStretch()
        log_layout.addLayout(toggle_row)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)
        self.log_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size:11px; }"
        )
        log_layout.addWidget(self.log_text)
        transcribe_layout.addWidget(self.log_panel_container)
        right_layout.addLayout(transcribe_layout)

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
        self.ja_check.stateChanged.connect(self._ensure_language_selected)
        self.ru_check.stateChanged.connect(self._ensure_language_selected)
        self.auto_sync_check.toggled.connect(self._on_auto_sync_toggled)

    # ================================================================
    # ヘルパー
    # ================================================================

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
        ja_item  = QTableWidgetItem(f"{seg.ja_prob:.2f}")
        ru_item  = QTableWidgetItem(f"{seg.ru_prob:.2f}")
        txt_item = QTableWidgetItem(display_text(seg))
        if seg.ja_prob >= seg.ru_prob:
            ja_item.setForeground(QColor(200, 0, 0))
            ru_item.setForeground(QColor(0, 0, 180))
        else:
            ru_item.setForeground(QColor(200, 0, 0))
            ja_item.setForeground(QColor(0, 0, 180))
        self.transcription_table.setItem(row_index, 2, ja_item)
        self.transcription_table.setItem(row_index, 3, ru_item)
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
        act_set_start = menu.addAction("現在位置をSTARTにセット")
        act_set_end   = menu.addAction("現在位置をENDにセット")

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

        # 動的分割: 1 行 or 連続 2 行
        can_dynamic = False
        if len(rows) == 1 and rows and 0 <= rows[0] < len(segs):
            can_dynamic = True
        elif (len(rows) == 2 and rows[1] == rows[0] + 1
              and all(0 <= r < len(segs) for r in rows)):
            can_dynamic = True
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

        # START/END セット: 単一行かつ再生位置が妥当な範囲
        can_set = len(rows) == 1 and rows and 0 <= rows[0] < len(segs)
        if can_set:
            try:
                cur_sec = self.media_player.position() / 1000.0
                seg = segs[rows[0]]
                s = float(seg.get('start', 0.0))
                e = float(seg.get('end', s))
                if not (cur_sec < e - 0.01):
                    act_set_start.setEnabled(False)
                if not (cur_sec > s + 0.01):
                    act_set_end.setEnabled(False)
            except Exception:
                act_set_start.setEnabled(False)
                act_set_end.setEnabled(False)
        else:
            act_set_start.setEnabled(False)
            act_set_end.setEnabled(False)

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

        elif chosen == act_set_start and act_set_start.isEnabled():
            r0 = rows[0]
            try:
                cur_sec = self.media_player.position() / 1000.0
                seg = segs[r0]
                if cur_sec < float(seg.get('end', cur_sec)) - 0.01:
                    seg['start'] = cur_sec
                    self._rebuild_text_and_refresh()
                    self.status_label.setText(f"行 {r0} の START を {cur_sec:.3f}s に設定")
            except Exception:
                pass

        elif chosen == act_set_end and act_set_end.isEnabled():
            r0 = rows[0]
            try:
                cur_sec = self.media_player.position() / 1000.0
                seg = segs[r0]
                if cur_sec > float(seg.get('start', cur_sec)) + 0.01:
                    seg['end'] = cur_sec
                    self._rebuild_text_and_refresh()
                    self.status_label.setText(f"行 {r0} の END を {cur_sec:.3f}s に設定")
            except Exception:
                pass

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

        dlg = EditDialog(seg, parent=self, dialog_size=self.split_dialog_size)
        if dlg.exec() != EditDialog.Accepted:
            return

        if dlg.mode == 'split':
            self.perform_segment_split(row, dlg.split_which, dlg.split_pos)

        elif dlg.mode == 'edit':
            seg['text_ja'] = dlg.new_text_ja
            seg['text_ru'] = dlg.new_text_ru
            seg['chosen_language'] = dlg.chosen_language
            seg['text'] = dlg.new_text_ja if dlg.chosen_language == 'ja' else dlg.new_text_ru
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
        """1 行選択: 現在位置で時間基準に 2 分割。
        2 行連続選択: 境界を現在位置に移動。
        いずれも前後区間を再文字起こし。
        """
        if getattr(self, 'range_retranscribing', False):
            return
        if not getattr(self, 'transcription_result', None):
            return
        segs = as_segment_list(self.transcription_result.get('segments', []))
        rows = self._collect_selected_rows()
        if not rows:
            return
        rows = sorted(rows)
        try:
            cur_sec = self.media_player.position() / 1000.0
        except Exception:
            return

        # ---- 1 行選択: セグメント内部で時間分割 ----
        if len(rows) == 1:
            r = rows[0]
            if r < 0 or r >= len(segs):
                return
            new_list, front_index = dynamic_time_split(
                self.transcription_result.get('segments', []), r, cur_sec
            )
            if new_list is None:
                return
            self.transcription_result['segments'] = new_list

            # 分割直後は旧テキスト断片を消去してプレースホルダを表示
            for segx in (new_list[front_index], new_list[front_index + 1]):
                segx['text'] = segx['text_ja'] = segx['text_ru'] = PLACEHOLDER_PENDING
                segx['ja_prob'] = segx['ru_prob'] = 0.0
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
            return

        # ---- 2 行連続選択: 境界時刻を現在位置に調整 ----
        if len(rows) == 2 and rows[1] == rows[0] + 1:
            r1, r2 = rows
            if r1 < 0 or r2 >= len(segs):
                return
            seg1 = segs[r1]
            seg2 = segs[r2]
            start = float(seg1.get('start', 0.0))
            end   = float(seg2.get('end', start))
            if not (start < cur_sec < end):
                return
            if (cur_sec - start < MIN_SEGMENT_DUR) or (end - cur_sec < MIN_SEGMENT_DUR):
                return
            updated = adjust_boundary(
                self.transcription_result.get('segments', []), r1, cur_sec, MIN_SEGMENT_DUR
            )
            self.transcription_result['segments'] = updated
            self.range_retranscribing = True
            self._pending_rejobs = [
                ('front', updated[r1]['start'], updated[r1]['end']),
                ('back',  updated[r2]['start'], updated[r2]['end']),
            ]
            self._split_row_base = r1
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
            'device':  self.device_combo.currentText(),
            'ja_weight': self.ja_weight_slider.value() / 100.0,
            'ru_weight': self.ru_weight_slider.value() / 100.0,
        }
        self._active_split_kind = kind
        self.range_thread = RangeTranscriptionThread(
            self.current_video_path, start_sec, end_sec, options
        )
        self.range_thread.progress.connect(self._on_range_progress)
        self.range_thread.status.connect(self._on_range_status)
        self.range_thread.range_finished.connect(self._on_split_rejob_finished)
        self.range_thread.error.connect(self._on_split_rejob_error)
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
                'text_ja':  seg.get('text_ja', ''),
                'text_ru':  seg.get('text_ru', ''),
                'ja_prob':  float(seg.get('ja_prob', 0.0)),
                'ru_prob':  float(seg.get('ru_prob', 0.0)),
                'chosen_language': (
                    seg.get('chosen_language') or seg.get('language')
                    or tgt.get('chosen_language')
                ),
            })
            # テキストが両方空なら明示ラベル
            if not tgt.text_ja and not tgt.text_ru:
                tgt.text = tgt.text_ja = tgt.text_ru = '(空)'
                tgt.ja_prob = tgt.ru_prob = 0.0

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
                if any(seg.get(k) for k in ('text', 'text_ja', 'text_ru')):
                    seg['text'] = seg['text_ja'] = seg['text_ru'] = ''
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
            'device':    self.device_combo.currentText(),
            'ja_weight': self.ja_weight_slider.value() / 100.0,
            'ru_weight': self.ru_weight_slider.value() / 100.0,
        }
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

            def single_finished(seg_result: dict) -> None:
                try:
                    if 0 <= idx < len(segs):
                        segs[idx].update({
                            'text':     seg_result.get('text', ''),
                            'text_ja':  seg_result.get('text_ja', ''),
                            'text_ru':  seg_result.get('text_ru', ''),
                            'ja_prob':  seg_result.get('ja_prob', 0.0),
                            'ru_prob':  seg_result.get('ru_prob', 0.0),
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
            ph['text'] = ph['text_ja'] = ph['text_ru'] = PLACEHOLDER_PENDING
            ph['ja_prob'] = ph['ru_prob'] = 0.0
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
                new_segments.append({
                    'start': new_seg['start'], 'end': new_seg['end'],
                    'text': new_seg['text'], 'text_ja': new_seg['text_ja'],
                    'text_ru': new_seg['text_ru'],
                    'chosen_language': new_seg['chosen_language'],
                    'id': s.get('id', idx),
                    'ja_prob': new_seg['ja_prob'], 'ru_prob': new_seg['ru_prob'],
                })
            elif idx in rows_sorted[1:]:
                continue  # 縮約されたので除去
            else:
                new_segments.append(s)

        self.transcription_result['segments'] = new_segments
        self._rebuild_text_and_refresh()
        self.status_label.setText("再文字起こし完了")

    def _on_range_error(self, err: str) -> None:
        QApplication.restoreOverrideCursor()
        self.range_retranscribing = False
        self.status_label.setText("再文字起こし失敗")
        QMessageBox.critical(self, "再文字起こし失敗", err)

    # ================================================================
    # 書き出し
    # ================================================================

    def partial_export_selected(self) -> None:
        """選択行の音声 (WAV) とテキスト (TXT) を ./output に書き出す。"""
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

        start_sec = float(target[0].get('start', 0.0))
        end_sec   = float(target[-1].get('end', start_sec))
        if end_sec <= start_sec:
            QMessageBox.critical(self, "部分書き出し", "時間範囲が不正です")
            return

        text_out  = '\n'.join(display_text(s) for s in target)
        base_name = os.path.splitext(os.path.basename(self.current_video_path))[0]

        def ts(sec: float) -> str:
            h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
            return f"{h:02d}{m:02d}{s:02d}"

        out_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        os.makedirs(out_dir, exist_ok=True)
        stem     = f"{base_name}_{ts(start_sec)}_{ts(end_sec)}"
        wav_path = os.path.join(out_dir, f"{stem}.wav")
        txt_path = os.path.join(out_dir, f"{stem}.txt")

        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', self.current_video_path,
                 '-ss', f"{start_sec:.3f}", '-to', f"{end_sec:.3f}",
                 '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "部分書き出し", "音声抽出に失敗しました (ffmpeg)")
            return

        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_out)
        except Exception as e:
            QMessageBox.critical(self, "部分書き出し", f"テキスト保存失敗: {e}")
            return

        try:
            cwd    = os.getcwd()
            rel_wav = os.path.relpath(wav_path, cwd)
            rel_txt = os.path.relpath(txt_path, cwd)
        except Exception:
            rel_wav, rel_txt = wav_path, txt_path

        duration = end_sec - start_sec
        msg = (
            "部分書き出しが完了しました\n"
            "--------------------------------\n"
            f" 対象行数 : {len(target)}\n"
            f" 時間範囲 : {start_sec:,.3f}s 〜 {end_sec:,.3f}s (Δ {duration:.3f}s)\n"
            f" 出力WAV : {rel_wav}\n"
            f" 出力TXT : {rel_txt}\n"
            "--------------------------------"
        )
        self.status_label.setText("部分書き出し完了")
        self._append_log(msg)
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
                        'device': self.device_combo.currentText(),
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
    # 文字起こし開始 / 制御
    # ================================================================

    def start_transcription(self) -> None:
        """設定値を収集して TranscriptionThread を起動する。"""
        selected_langs = []
        if self.ja_check.isChecked():
            selected_langs.append("ja")
        if self.ru_check.isChecked():
            selected_langs.append("ru")

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

        options: dict = {
            "model":  self.model_combo.currentText(),
            "device": self.device_combo.currentText(),
            "segmentation_model_size": self.segmentation_model_combo.currentText(),
            "seg_mode": "hybrid" if len(selected_langs) >= 2 else "normal",
            "ja_weight": self.ja_weight_slider.value() / 100.0,
            "ru_weight": self.ru_weight_slider.value() / 100.0,
        }
        options.update(detail_values)

        # 重複マージ制御 ([debug] セクション参照)
        try:
            dbg = self.config.get('debug', {}) if isinstance(self.config, dict) else {}
            options['duplicate_merge'] = bool(dbg.get('duplicate_merge', True))
            options['duplicate_debug'] = bool(dbg.get('duplicate_debug', True))
        except Exception:
            options['duplicate_merge'] = True
            options['duplicate_debug'] = True

        # プロファイルの無音抑制設定
        try:
            prof = self.config.get(self.current_profile_name, {})
            for k in ('silence_rms_threshold', 'min_voice_ratio', 'max_silence_repeat'):
                options[k] = prof.get(k)
        except Exception:
            pass

        if selected_langs:
            options["language"] = selected_langs[0]

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
            line = display_text(seg_dict)
            if line:
                prev_text = self.transcription_result.get('text', '') if self.transcription_result else ''
                self.transcription_result['text'] = (
                    prev_text + ('\n' if prev_text else '') + line
                )
            row = self.transcription_table.rowCount()
            self.transcription_table.insertRow(row)
            st_str = format_ms(int(seg_dict.get('start', 0.0) * 1000))
            ed_str = format_ms(int(seg_dict.get('end',   0.0) * 1000))
            self.transcription_table.setItem(row, 0, QTableWidgetItem(st_str))
            self.transcription_table.setItem(row, 1, QTableWidgetItem(ed_str))
            self.transcription_table.setItem(
                row, 2, QTableWidgetItem(f"{seg_dict.get('ja_prob', 0.0):.2f}")
            )
            self.transcription_table.setItem(
                row, 3, QTableWidgetItem(f"{seg_dict.get('ru_prob', 0.0):.2f}")
            )
            self.transcription_table.setItem(
                row, 4, QTableWidgetItem(seg_dict.get('text', ''))
            )
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

    def _ensure_language_selected(self) -> None:
        """少なくとも 1 言語が選択された状態を維持する。"""
        QTimer.singleShot(0, self._ensure_language_selected_late)

    def _ensure_language_selected_late(self) -> None:
        if not (self.ja_check.isChecked() or self.ru_check.isChecked()):
            self.ja_check.setChecked(True)

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

    def _toggle_log_panel(self, checked: bool) -> None:
        self.log_text.setVisible(checked)
        self.toggle_log_button.setText("▲ ログ非表示" if checked else "▼ ログ表示")

    def display_transcription(self, result: dict) -> None:
        """テーブルにセグメントを表示する。"""
        if not result:
            return
        populate_table(self.transcription_table, result)
