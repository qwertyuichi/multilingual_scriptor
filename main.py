import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QStyle,
    QSizePolicy,
    QGroupBox,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QTextEdit,
    QSplitter,
    QProgressBar,
    QComboBox,
    QCheckBox,
    QScrollArea,
    QGridLayout,
)
from PySide6.QtCore import Qt, QUrl, Slot, QLoggingCategory, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from transcriber import TranscriptionThread
import torch
import os
import tomllib as _toml

# Qt Multimedia FFmpegログを非表示
QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg=false")


class VideoTranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 設定読み込み
        self.config = self.load_config()
        self.setWindowTitle("動画文字起こしエディタ")
        self.setGeometry(0, 0, 1400, 800)

        # メディアプレイヤーの初期化
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # 文字起こしスレッド
        self.transcription_thread = None
        self.current_video_path = None
        self.transcription_result = None

        # UIの初期化
        self.init_ui()

        # シグナルの接続
        self.connect_signals()

    def load_config(self):
        """config.toml を読み込み。無ければエラーを送出。フォールバック挿入は行わない。"""
        cfg_path = os.path.join(os.path.dirname(__file__), "config.toml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("config.toml not found.")
        try:
            with open(cfg_path, "rb") as f:
                return _toml.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config.toml: {e}")

    def init_ui(self):
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ヘルプテキスト定義
        HELP_TEXTS = {
            "設定項目": (
                "config.toml の [default] セクションに存在するキー一覧を表示します。\n"
                "現段階では選択しても処理内容には影響しません (将来フィルタ/ジャンプ用途予定)。\n"
                "・用途: 設定ファイル構成の把握 / 今後の編集UI拡張の基礎"),
            "デバイス": (
                "Whisper を実行する計算デバイス。\n"
                "cuda: GPU 使用 (推奨 / 速度向上)\ncpu: CPU フォールバック (低速)\n"
                "・CUDA が利用不可の場合は自動的に cpu のみ選択可\n"
                "・マシンに複数 GPU がある場合の個別指定は今後拡張余地"),
            "トランスクリプションモデル": (
                "音声→文字変換に使用する Whisper モデルサイズ。\n"
                "小さなモデル: 高速 / 低精度・短文向き\n大きなモデル: 低速 / 高精度・多言語安定\n"
                "・large-v3 推奨 (精度バランス)\n・低スペック環境: base / small で試行"),
            "セグメンテーションモデル": (
                "音声区間の切り出し(セグメント化)に用いるモデル。\n"
                "turbo: 高速 (精度より速度優先)\nlarge 系: 精度は高いが遅い\n"
                "・実際の文字起こしモデルとは別個にロード\n・精度が不要に低い場合のみサイズ拡大を検討"),
            "ja_weight": (
                "日本語言語判定スコアへの補正係数。\n"
                "最終確率 = 元スコア × weight を正規化後に比較。\n"
                "1.0 = 補正なし / >1.0 で日本語優遇 / <1.0 で抑制。\n"
                "・多言語混在で日本語が取りこぼされる場合は 1.1～1.3 程度を試す"),
            "ru_weight": (
                "ロシア語判定スコア補正係数 (ja_weight と同様の計算)。\n"
                "・日本語優勢すぎてロシア語が検出されにくい場合に 1.1～1.4 など調整"),
            "en_weight": (
                "英語判定スコア補正係数。\n"
                "・英語が断片的に含まれる配信で取りこぼしを防ぎたい場合に引き上げ"),
            "min_seg_dur": (
                "1 つのセグメントがこれ未満秒なら分割候補から除外 (過分割防止)。\n"
                "推奨: 0.4～0.8 秒。短すぎるとノイズ増 / 長すぎると応答遅延"),
            "mix_threshold": (
                "主要2言語(JA/RU) の確信度差 |JA-RU| がこの値未満なら [MIX] マークを付与。\n"
                "・値を小さく: MIX 判定減\n・値を大きく: MIX 判定増 (雑多表示の恐れ)"),
            "ambiguous_threshold": (
                "主要言語間の確信度差がこの値未満で 'あいまい' とみなし、\n"
                "両(複数)言語再トライ比較を行う境界。\n"
                "・大きくすると再トライ頻度 ↑ (精度↑/速度↓)\n"
                "・小さくすると再トライ減 (速度↑/誤判定リスク↑)"),
            "vad_level": (
                "WebRTC VAD (Voice Activity Detection) 感度。\n"
                "0: もっとも寛容 (雑音も音声扱い)\n3: 最も厳格 (静音判定が増える)\n"
                "・無音除去の厳しさを調整し、短いノイズを避けたい場合は 2～3"),
            "gap_threshold": (
                "前セグメント終端と次セグメント開始の間隔がこの秒数以上なら '無音ギャップ' として扱う。\n"
                "include_silent 有効時のみログ出力。\n"
                "・MC/トーク番組: 0.3～0.7\n・間が長い朗読: 1.0 以上も検討"),
            "include_silent": (
                "True で無音/スキップ理由 (短すぎる, silence 等) を出力ログに残す。\n"
                "解析/デバッグ用途向け。通常運用では False でログ簡潔化。"),
            "srt_max_line": (
                "SRT 生成時、1 セグメントを強制的に改行分割する際の行数上限目安。\n"
                "・視認性を優先: 2～4\n・長文字幕許容: 6～8"),
            "initial_prompt": (
                "最初の推論呼び出しに与えるコンテキスト文字列。\n"
                "固有名詞/話題/口調を誘導したい場合に使用。\n"
                "空文字なら無視。過剰に長いと逆効果の可能性。"),
            "output_format": (
                "保存出力フォーマット。\n"
                "txt: 単純テキスト\nsrt: 字幕フォーマット (タイムコード付き)\njson: 構造化データ (後処理用)"),
            "default_languages": (
                "起動時に ON にする言語コード配列。\n"
                "最低 1 つ必須。ここで指定されない言語は最初は未選択。\n"
                "多言語配信が確実な場合のみ 2 つ以上を推奨 (速度僅かに低下)"),
        }

        def help_label(key: str):
            lbl = QLabel("?")
            lbl.setFixedWidth(16)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("QLabel { color: #007acc; font-weight: bold; border:1px solid #007acc; border-radius:8px; }")
            if key in HELP_TEXTS:
                lbl.setToolTip(HELP_TEXTS[key])
            else:
                lbl.setToolTip(key)
            return lbl

        # 左側：ビデオプレイヤーと文字起こし結果
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # ビデオウィジェット
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.media_player.setVideoOutput(self.video_widget)
        left_layout.addWidget(self.video_widget, 2)

        # コントロールパネル
        control_layout = QVBoxLayout()

        # シークバー
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        control_layout.addWidget(self.position_slider)

        # 時間ラベル
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        control_layout.addLayout(time_layout)

        # ボタンコントロール
        button_layout = QHBoxLayout()

        # ファイル選択ボタン
        self.open_button = QPushButton("動画を開く")
        self.open_button.clicked.connect(self.open_file)
        button_layout.addWidget(self.open_button)

        # 再生/一時停止ボタン
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_pause)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.play_button)

        # 停止ボタン
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        button_layout.addStretch()
        control_layout.addLayout(button_layout)

        left_layout.addLayout(control_layout)

        # 文字起こし結果表示エリア
        self.transcription_text = QTextEdit()
        self.transcription_text.setPlaceholderText("文字起こし結果がここに表示されます")
        left_layout.addWidget(self.transcription_text, 1)

        # 右側：文字起こし設定
        right_widget = QWidget()
        right_widget.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_widget)

        # スクロールエリア
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 基本設定
        model_group = QGroupBox("基本設定")
        model_layout = QGridLayout()

        label_setting_item = QLabel("設定項目:")
        model_layout.addWidget(label_setting_item, 0, 0)
        self.setting_item_combo = QComboBox()
        setting_items = list(self.config["default"].keys())
        self.setting_item_combo.addItems(setting_items)
        model_layout.addWidget(self.setting_item_combo, 0, 1)
        model_layout.addWidget(help_label("設定項目"), 0, 2)

        dev_label = QLabel("デバイス:")
        model_layout.addWidget(dev_label, 1, 0)
        self.device_combo = QComboBox()
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        self.device_combo.addItems(devices)
        default_device = self.config["default"].get("device", "cuda")
        if default_device not in devices:
            devices.append(default_device)
        self.device_combo.setCurrentText(default_device)
        model_layout.addWidget(self.device_combo, 1, 1)
        model_layout.addWidget(help_label("デバイス"), 1, 2)

        tmodel_label = QLabel("トランスクリプションモデル:")
        model_layout.addWidget(tmodel_label, 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
        ])
        default_model = self.config["default"].get("transcription_model", "large-v3")
        if default_model not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.addItem(default_model)
        self.model_combo.setCurrentText(default_model)
        model_layout.addWidget(self.model_combo, 2, 1)
        model_layout.addWidget(help_label("トランスクリプションモデル"), 2, 2)

        seg_label = QLabel("セグメンテーションモデル:")
        model_layout.addWidget(seg_label, 3, 0)
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.addItems([
            "turbo","small","base","tiny","medium","large","large-v2","large-v3"
        ])
        default_segmentation_model = self.config["default"].get("segmentation_model", "turbo")
        if default_segmentation_model not in [self.segmentation_model_combo.itemText(i) for i in range(self.segmentation_model_combo.count())]:
            self.segmentation_model_combo.addItem(default_segmentation_model)
        self.segmentation_model_combo.setCurrentText(default_segmentation_model)
        model_layout.addWidget(self.segmentation_model_combo, 3, 1)
        model_layout.addWidget(help_label("セグメンテーションモデル"), 3, 2)

        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)

        # 言語設定
        lang_group = QGroupBox("言語設定")
        lang_layout = QVBoxLayout()

        # 言語 + weight スライダー
        dft = self.config.get("default", {})
        dlangs = dft.get("default_languages", ["ja", "ru"])
        if not isinstance(dlangs, list) or not dlangs:
            dlangs = ["ja"]

        def make_lang_row(code: str, label_text: str, default_weight_key: str):
            row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0,0,0,0)
            chk = QCheckBox(label_text)
            chk.setChecked(code in dlangs)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0); slider.setMaximum(300); slider.setSingleStep(5)
            slider.setFixedWidth(140)
            val = dft.get(default_weight_key, 1.0)
            if not isinstance(val, (int,float)):
                val = 1.0
            slider.setValue(int(round(val*100)))  # 1.00 -> 100
            value_label = QLabel(f"{val:.2f}")
            def on_change(v):
                value_label.setText(f"{v/100:.2f}")
            slider.valueChanged.connect(on_change)
            hl.addWidget(chk)
            weight_lbl = QLabel("weight")
            hl.addWidget(weight_lbl)
            hl.addWidget(slider)
            hl.addWidget(value_label)
            # ヘルプマーク
            hl.addWidget(help_label(default_weight_key))
            return chk, slider, value_label, row

        self.ja_check, self.ja_slider_label_dummy, self.ja_weight_value_label_dummy = None, None, None
        self.ja_check, self.ja_weight_slider_label_dummy, self.ja_weight_value_label_dummy, ja_row = make_lang_row("ja", "JA", "ja_weight")
        self.ja_weight_slider = ja_row.findChildren(QSlider)[0]
        self.ja_weight_value_label = ja_row.findChildren(QLabel)[-1]
        lang_layout.addWidget(ja_row)

        self.ru_check, _, _, ru_row = make_lang_row("ru", "RU", "ru_weight")
        self.ru_weight_slider = ru_row.findChildren(QSlider)[0]
        self.ru_weight_value_label = ru_row.findChildren(QLabel)[-1]
        lang_layout.addWidget(ru_row)

        self.en_check, _, _, en_row = make_lang_row("en", "EN", "en_weight")
        self.en_weight_slider = en_row.findChildren(QSlider)[0]
        self.en_weight_value_label = en_row.findChildren(QLabel)[-1]
        lang_layout.addWidget(en_row)

        lang_group.setLayout(lang_layout)
        scroll_layout.addWidget(lang_group)

        # 詳細設定 (config.toml の default セクションから、基本設定で使ったキーを除外して動的生成)
        detail_group = QGroupBox("詳細設定")
        detail_layout = QVBoxLayout()

        self.detail_controls = {}
        exclude_keys = {"device", "transcription_model", "segmentation_model", "default_languages", "ja_weight", "ru_weight", "en_weight"}

        # キー毎に適切なウィジェットを生成
        for key, value in dft.items():
            if key in exclude_keys:
                continue
            # initial_prompt は複数行テキスト
            if key == "initial_prompt":
                lbl = QLabel("initial_prompt:")
                detail_layout.addWidget(lbl)
                txt = QTextEdit()
                txt.setMaximumHeight(100)
                txt.setPlainText(value or "")
                detail_layout.addWidget(txt)
                self.detail_controls[key] = txt
                continue
            row_container = QWidget()
            row_layout = QHBoxLayout(row_container)
            row_layout.setContentsMargins(0, 0, 0, 0)
            key_label = QLabel(f"{key}:")
            row_layout.addWidget(key_label)
            ctrl = None
            if isinstance(value, bool):
                cb = QCheckBox()
                cb.setChecked(bool(value))
                ctrl = cb
            elif isinstance(value, int) and not isinstance(value, bool):
                if key == "vad_level":
                    combo = QComboBox()
                    for i in range(0, 4):
                        combo.addItem(str(i), i)
                    # 存在しない値なら 2 をデフォルト
                    idx = combo.findText(str(value))
                    if idx < 0:
                        idx = combo.findText("2")
                    combo.setCurrentIndex(idx)
                    ctrl = combo
                else:
                    sp = QSpinBox()
                    sp.setRange(-999999, 999999)
                    sp.setValue(int(value))
                    if key == "srt_max_line":
                        sp.setRange(1, 1000)
                    ctrl = sp
            elif isinstance(value, float):
                dsp = QDoubleSpinBox()
                dsp.setDecimals(4)
                dsp.setRange(-1e9, 1e9)
                dsp.setSingleStep(0.05)
                dsp.setValue(float(value))
                # よく使う閾値系はもう少し細かく
                if key in {"min_seg_dur", "gap_threshold"}:
                    dsp.setRange(0.0, 60.0)
                    dsp.setSingleStep(0.05)
                if key in {"ja_weight", "ru_weight", "en_weight"}:
                    dsp.setRange(0.0, 10.0)
                    dsp.setSingleStep(0.05)
                if key in {"mix_threshold", "ambiguous_threshold"}:
                    dsp.setRange(0.0, 100.0)
                    dsp.setSingleStep(0.5)
                ctrl = dsp
            elif isinstance(value, str):
                if key == "output_format":
                    combo = QComboBox()
                    # 想定される出力形式候補
                    candidates = ["txt", "srt", "json"]
                    if value not in candidates:
                        candidates.append(value)
                    combo.addItems(candidates)
                    combo.setCurrentText(value)
                    ctrl = combo
                else:
                    le = QLineEdit()
                    le.setText(value)
                    ctrl = le
            else:
                # 未知タイプは文字列化して LineEdit
                le = QLineEdit()
                le.setText(str(value))
                ctrl = le
            if ctrl is not None:
                row_layout.addWidget(ctrl)
                # ヘルプマークがあれば追加
                row_layout.addWidget(help_label(key))
                self.detail_controls[key] = ctrl
                detail_layout.addWidget(row_container)

        detail_group.setLayout(detail_layout)
        scroll_layout.addWidget(detail_group)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)

        # 文字起こし実行ボタンとプログレスバー
        transcribe_layout = QVBoxLayout()

        self.transcribe_button = QPushButton("文字起こしを開始")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        transcribe_layout.addWidget(self.transcribe_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        transcribe_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        transcribe_layout.addWidget(self.status_label)

        right_layout.addLayout(transcribe_layout)

        # スプリッターで左右を分割
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def connect_signals(self):
        # メディアプレイヤーのシグナル
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.playbackStateChanged.connect(self.media_state_changed)

        # スライダーのシグナル
        self.position_slider.sliderMoved.connect(self.set_position)

        # 言語チェックボックスのシグナル
        self.ja_check.stateChanged.connect(self.ensure_language_selected)
        self.ru_check.stateChanged.connect(self.ensure_language_selected)
        self.en_check.stateChanged.connect(self.ensure_language_selected)

    def open_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "動画ファイルを選択",
            "",
            "動画ファイル (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;すべてのファイル (*.*)",
        )

        if file_path:
            self.current_video_path = file_path
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.transcribe_button.setEnabled(True)
            self.setWindowTitle(f"動画文字起こしエディタ - {file_path}")

    def ensure_language_selected(self):
        # 状態変更イベントの連続発火で一時的に全チェックが外れることがあるため
        # イベントループの末尾に保護処理を遅延実行して最終状態を確認する
        QTimer.singleShot(0, self._ensure_language_selected_late)

    def _ensure_language_selected_late(self):
        # 言語が一つも選択されていない場合、JAを自動選択
        if not (
            self.ja_check.isChecked()
            or self.ru_check.isChecked()
            or self.en_check.isChecked()
        ):
            self.ja_check.setChecked(True)

    def start_transcription(self):
        # 言語選択 (seg_mode 判定用)
        selected_langs = []
        if self.ja_check.isChecked():
            selected_langs.append("ja")
        if self.ru_check.isChecked():
            selected_langs.append("ru")
        if self.en_check.isChecked():
            selected_langs.append("en")

        # 詳細設定ウィジェット値を収集 (言語 weight はここでは扱わない)
        detail_values = {}
        for key, ctrl in self.detail_controls.items():
            from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QComboBox
            if isinstance(ctrl, QCheckBox):
                detail_values[key] = ctrl.isChecked()
            elif isinstance(ctrl, (QSpinBox, QDoubleSpinBox)):
                detail_values[key] = ctrl.value()
            elif isinstance(ctrl, (QLineEdit,)):
                detail_values[key] = ctrl.text()
            elif isinstance(ctrl, QTextEdit):
                txt = ctrl.toPlainText()
                detail_values[key] = txt if txt else ""
            elif isinstance(ctrl, QComboBox):
                detail_values[key] = ctrl.currentText()

        # オプション構築 (高度モードのみ運用)
        options = {
            "advanced": True,
            "model": self.model_combo.currentText(),
            "device": self.device_combo.currentText(),
            "segmentation_model_size": self.segmentation_model_combo.currentText(),
            "seg_mode": "hybrid" if len(selected_langs) >= 2 else "normal",
        }
        # detail_values を統合
        options.update(detail_values)
        # 言語 weight スライダー値を追加 (0-300 -> 0.00-3.00)
        options["ja_weight"] = self.ja_weight_slider.value() / 100.0
        options["ru_weight"] = self.ru_weight_slider.value() / 100.0
        options["en_weight"] = self.en_weight_slider.value() / 100.0
        # 互換用 language (未使用だが将来拡張向け)
        if selected_langs:
            options["language"] = selected_langs[0]

        # UIを無効化
        self.transcribe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 文字起こしスレッドを開始
        self.transcription_thread = TranscriptionThread(
            self.current_video_path, options
        )
        self.transcription_thread.progress.connect(self.update_progress)
        self.transcription_thread.status.connect(self.update_status)
        self.transcription_thread.finished_transcription.connect(
            self.on_transcription_finished
        )
        self.transcription_thread.error.connect(self.on_transcription_error)
        self.transcription_thread.start()

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(message)

    @Slot(dict)
    def on_transcription_finished(self, result):
        self.transcription_result = result
        self.display_transcription(result)
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("文字起こし完了")

    @Slot(str)
    def on_transcription_error(self, error_message):
        self.status_label.setText(f"エラー: {error_message}")
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def display_transcription(self, result):
        # セグメントごとに時間と文字起こし結果を表示
        text = ""
        for segment in result["segments"]:
            start_time = self.format_time(int(segment["start"] * 1000))
            end_time = self.format_time(int(segment["end"] * 1000))
            text += f"[{start_time} - {end_time}]\n{segment['text']}\n\n"

        self.transcription_text.setPlainText(text)

    def play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop_video(self):
        self.media_player.stop()

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))

    def set_position(self, position):
        self.media_player.setPosition(position)

    def format_time(self, milliseconds):
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        hours = minutes // 60
        minutes = minutes % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


def main():
    app = QApplication(sys.argv)
    window = VideoTranscriptionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
