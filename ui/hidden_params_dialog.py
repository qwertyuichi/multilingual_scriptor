"""高度な設定ダイアログ。

config.toml の [advanced] セクションを GUI から編集可能にします。
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QWidget,
)
from PySide6.QtCore import Qt

from core.constants import WHISPER_LANGUAGES, WHISPER_LANGUAGES_JA


class HiddenParamsDialog(QDialog):
    """高度な設定編集ダイアログ。"""

    def __init__(self, current_hidden: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("高度な設定")
        # Reduce minimum height so dialog can shrink to fit content
        self.setMinimumSize(980, 594)
        
        self.current_hidden = current_hidden.copy()
        self.widgets = {}
        self.lang_checks: dict[str, QCheckBox] = {}
        
        self._init_ui()
    
    def _init_ui(self):
        """UI構築。"""
        main_layout = QVBoxLayout(self)
        
        def help_mark(tooltip: str) -> QLabel:
            mark = QLabel("?")
            mark.setFixedWidth(16)
            mark.setAlignment(Qt.AlignCenter)
            mark.setStyleSheet(
                "QLabel { color: #007acc; font-weight: bold; "
                "border:1px solid #007acc; border-radius:8px; }"
            )
            mark.setToolTip(tooltip)
            return mark

        def row_with_help(control: QWidget, tooltip: str, extra: QWidget | None = None) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(control)
            if extra is not None:
                h.addWidget(extra)
            h.addStretch()
            h.addWidget(help_mark(tooltip))
            return w

        # 説明ラベル
        info = QLabel(
            "高度なパラメータです。変更後は「適用」を押すと config.toml の [advanced] セクションに保存されます。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { color: #888; margin-bottom: 10px; }")
        main_layout.addWidget(info)
        
        # スクロールエリア
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Allow the scroll area to expand/shrink to avoid leaving empty space below
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        # Ensure inner widget expands with the scroll area
        scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        columns_container = QWidget()
        columns_layout = QHBoxLayout(columns_container)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        columns_layout.addLayout(left_col, 1)
        columns_layout.addLayout(right_col, 1)
        
        # ---- 言語判定設定 (Beam Size + あいまい判定を統合) ----
        # Shared height for language-related groups so they align exactly
        # Restoring to original value per request.
        SHARED_LANG_HEIGHT = 260
        lang_detect_group = QGroupBox("言語判定設定")
        lang_detect_layout = QFormLayout()

        # Beam sizes (Phase1 / Phase2 detection / Phase2 retranscribe)
        params = [
            ("phase1_beam_size", "Phase1 Beam", 1, 10, 5,
             "全体転写時の beam_size。大きいほど精度は上がるが処理時間が増える。"),
            ("phase2_detect_beam_size", "Phase2 検出 Beam", 1, 10, 1,
             "クリップ単位の言語検出時に使う beam_size。通常は 1 で十分。"),
            ("phase2_retranscribe_beam_size", "Phase2 再転写 Beam", 1, 10, 5,
             "再転写時の beam_size。あいまいな場合はやや大きめを検討。"),
        ]

        for key, label, min_val, max_val, default, tooltip in params:
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(int(self.current_hidden.get(key, default)))
            spin.setToolTip(tooltip)  # ヘルプ側には単位を記載しない
            self.widgets[key] = spin
            extra = QLabel(f"（デフォルト: {default}）")
            lang_detect_layout.addRow(f"{label}:", row_with_help(spin, tooltip, extra))

        # ambiguous_threshold (help without units; UI shows unit)
        ambig_spin = QDoubleSpinBox()
        ambig_spin.setRange(0.0, 100.0)
        ambig_spin.setDecimals(1)
        ambig_spin.setSingleStep(0.1)
        ambig_val = float(self.current_hidden.get("ambiguous_threshold", 70.0))
        ambig_spin.setValue(ambig_val)
        ambig_spin.setToolTip(
            "主要言語間の確信度差がこの値未満なら 'あいまい' と判定します。"
        )
        self.widgets["ambiguous_threshold"] = ambig_spin
        ambig_extra = QLabel(f"pt（デフォルト: {ambig_val}）")
        lang_detect_layout.addRow(
            "あいまい判定しきい値:",
            row_with_help(ambig_spin, ambig_spin.toolTip(), ambig_extra),
        )

        lang_detect_group.setLayout(lang_detect_layout)
        # Make height identical to the language candidates group
        lang_detect_group.setFixedHeight(SHARED_LANG_HEIGHT)
        left_col.addWidget(lang_detect_group)
        
        # ---- 言語候補 ----
        lang_group = QGroupBox("言語候補")
        lang_outer = QVBoxLayout()

        lang_scroll = QScrollArea()
        lang_scroll.setWidgetResizable(True)
        # Make language candidate area height follow the shared height (with padding)
        # Give slightly more bottom padding to avoid touching the groupbox border.
        lang_scroll.setFixedHeight(SHARED_LANG_HEIGHT - 30)
        lang_widget = QWidget()
        lang_grid = QGridLayout(lang_widget)
        lang_grid.setContentsMargins(4, 4, 4, 4)
        lang_grid.setHorizontalSpacing(12)
        lang_grid.setVerticalSpacing(4)

        current_list = self.current_hidden.get("available_languages", [])
        if not isinstance(current_list, list):
            current_list = []
        current_set = {str(x) for x in current_list}

        items = [(code, WHISPER_LANGUAGES_JA.get(code, WHISPER_LANGUAGES[code]))
                 for code in WHISPER_LANGUAGES.keys()]
        items.sort(key=lambda x: x[0])
        cols = 2
        for i, (code, name) in enumerate(items):
            chk = QCheckBox(f"{code} - {name}")
            chk.setChecked(code in current_set)
            self.lang_checks[code] = chk
            r = i // cols
            c = i % cols
            lang_grid.addWidget(chk, r, c)

        lang_scroll.setWidget(lang_widget)
        lang_outer.addWidget(lang_scroll)
        lang_group.setLayout(lang_outer)
        # Set the same fixed height so the two columns align perfectly
        lang_group.setFixedHeight(SHARED_LANG_HEIGHT)
        lang_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_col.addWidget(lang_group)
        
        # ---- その他のパラメータグループ ----
        other_group = QGroupBox("その他")
        other_layout = QFormLayout()
        
        # condition_on_previous_text
        cond_check = QCheckBox()
        cond_check.setChecked(self.current_hidden.get("condition_on_previous_text", True))
        cond_check.setToolTip(
            "前セグメントのテキストを次セグメントの文脈として使用。\n"
            "繰り返しの誤認識が多い場合は false にすると改善する場合がある。"
        )
        self.widgets["condition_on_previous_text"] = cond_check
        other_layout.addRow(
            "前区間を文脈に使用:",
            row_with_help(cond_check, cond_check.toolTip()),
        )
        
        # compression_ratio_threshold
        comp_spin = QDoubleSpinBox()
        comp_spin.setRange(0.0, 10.0)
        comp_spin.setDecimals(1)
        comp_spin.setSingleStep(0.1)
        comp_spin.setValue(self.current_hidden.get("compression_ratio_threshold", 2.4))
        comp_spin.setToolTip(
            "圧縮比（出力テキストの繰り返し度合い）がこの値を超えるセグメントを検出します。\n"
            "範囲: 0.0〜10.0"
        )
        self.widgets["compression_ratio_threshold"] = comp_spin
        other_layout.addRow(
            "圧縮比しきい値:",
            row_with_help(comp_spin, comp_spin.toolTip()),
        )
        
        # log_prob_threshold
        log_spin = QDoubleSpinBox()
        log_spin.setRange(-10.0, 0.0)
        log_spin.setDecimals(1)
        log_spin.setSingleStep(0.1)
        log_spin.setValue(self.current_hidden.get("log_prob_threshold", -1.0))
        log_spin.setToolTip(
            "対数確率がこの値未満のセグメントを除外。\n"
            "範囲: -10.0〜0.0"
        )
        self.widgets["log_prob_threshold"] = log_spin
        other_layout.addRow(
            "対数確率しきい値:",
            row_with_help(log_spin, log_spin.toolTip()),
        )
        
        # repetition_penalty
        rep_spin = QDoubleSpinBox()
        rep_spin.setRange(1.0, 2.0)
        rep_spin.setDecimals(2)
        rep_spin.setSingleStep(0.05)
        rep_spin.setValue(self.current_hidden.get("repetition_penalty", 1.0))
        rep_spin.setToolTip(
            "繰り返し単語へのペナルティ（1.0=なし、大きいほど抑制）。\n"
            "範囲: 1.0〜2.0"
        )
        self.widgets["repetition_penalty"] = rep_spin
        other_layout.addRow(
            "繰り返しペナルティ:",
            row_with_help(rep_spin, rep_spin.toolTip()),
        )
        
        # speech_pad_ms
        pad_spin = QSpinBox()
        pad_spin.setRange(0, 2000)
        pad_spin.setSingleStep(50)
        pad_spin.setValue(self.current_hidden.get("speech_pad_ms", 400))
        pad_spin.setToolTip(
            "VAD検出した音声区間の前後に追加するパディング（ms）。\n"
            "範囲: 0〜2000"
        )
        self.widgets["speech_pad_ms"] = pad_spin
        other_layout.addRow(
            "音声パディング (ms):",
            row_with_help(pad_spin, pad_spin.toolTip()),
        )
        
        # duplicate_merge
        dup_check = QCheckBox()
        dup_check.setChecked(self.current_hidden.get("duplicate_merge", True))
        dup_check.setToolTip("完全一致する重複セグメントを自動的にマージ。")
        self.widgets["duplicate_merge"] = dup_check
        other_layout.addRow(
            "重複セグメント自動マージ:",
            row_with_help(dup_check, dup_check.toolTip()),
        )
        
        other_group.setLayout(other_layout)
        right_col.addWidget(other_group)

        # ---- ログ設定 ----
        log_group = QGroupBox("ログ設定")
        log_layout = QFormLayout()

        # log_level
        level_combo = QComboBox()
        level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        level_combo.setCurrentText(self.current_hidden.get("log_level", "INFO"))
        level_combo.setToolTip(
            "ログレベルを設定します。各レベルの意味:\n"
            "DEBUG: 開発向けの詳細ログ。内部デバッグ情報を含む。\n"
            "INFO: 通常運用向けの情報ログ。処理の大まかな進行を記録。\n"
            "WARNING: 想定外だが継続可能な問題を記録。\n"
            "ERROR: 処理に失敗したエラーを記録。\n"
            "CRITICAL: 致命的な状態を示します."
        )
        self.widgets["log_level"] = level_combo
        log_layout.addRow(
            "ログレベル:",
            row_with_help(level_combo, level_combo.toolTip()),
        )

        # log_file_path
        path_edit = QLineEdit()
        path_edit.setText(self.current_hidden.get("log_file_path", "app.log"))
        path_edit.setToolTip("ログファイルの出力先。相対パス可。")
        self.widgets["log_file_path"] = path_edit
        log_layout.addRow(
            "ログ出力パス:",
            row_with_help(path_edit, path_edit.toolTip()),
        )

        # log_max_bytes: UI 表示は KB 単位、ヘルプには単位を含めない
        max_kb_spin = QSpinBox()
        max_kb_spin.setRange(0, 100_000)  # 単位: KB (0 = 無制限)
        max_kb_spin.setSingleStep(100)
        # current_hidden may store bytes; convert to KB for display
        raw_mb = int(self.current_hidden.get("log_max_bytes", 1_048_576))
        display_kb = max(0, raw_mb // 1024)
        if display_kb == 0 and raw_mb > 0:
            display_kb = 1
        max_kb_spin.setValue(display_kb)
        max_kb_spin.setToolTip("ログローテーションのサイズ上限を KB 単位で指定します。(0=無制限)")
        self.widgets["log_max_bytes"] = max_kb_spin
        log_layout.addRow(
            "ログ最大サイズ:",
            row_with_help(max_kb_spin, max_kb_spin.toolTip(), QLabel(f"KB（デフォルト: {display_kb}）")),
        )

        # log_backup_count
        backup_spin = QSpinBox()
        backup_spin.setRange(0, 20)
        backup_spin.setValue(self.current_hidden.get("log_backup_count", 5))
        backup_spin.setToolTip("ログ世代数。0でローテーション無効。")
        self.widgets["log_backup_count"] = backup_spin
        log_layout.addRow(
            "ログ世代数:",
            row_with_help(backup_spin, backup_spin.toolTip()),
        )

        log_group.setLayout(log_layout)
        # Mirror size policy so left and right columns keep similar height
        log_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_col.addWidget(log_group)

        # ---- ログ関連フラグをログ設定に統合 ----
        # 確率デバッグログ
        prob_log_check = QCheckBox()
        prob_log_check.setChecked(self.current_hidden.get("debug_prob_log", False))
        prob_log_check.setToolTip("各クリップの言語確率を INFO レベルで出力します。")
        self.widgets["debug_prob_log"] = prob_log_check
        log_layout.addRow(
            "確率デバッグログ:",
            row_with_help(prob_log_check, prob_log_check.toolTip()),
        )

        # ファイルログ出力
        log_file_check = QCheckBox()
        log_file_check.setChecked(self.current_hidden.get("log_file_enabled", False))
        log_file_check.setToolTip("ログファイルへの出力を有効化します。")
        self.widgets["log_file_enabled"] = log_file_check
        log_layout.addRow(
            "ファイルログ出力:",
            row_with_help(log_file_check, log_file_check.toolTip()),
        )

        log_group.setLayout(log_layout)
        
        left_col.addStretch()
        right_col.addStretch()
        scroll_layout.addWidget(columns_container)
        scroll.setWidget(scroll_widget)
        # Give the scroll area stretch so it fills available vertical space
        main_layout.addWidget(scroll, 1)
        
        # ---- ボタン行 ----
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        reset_btn = QPushButton("デフォルトに戻す")
        reset_btn.clicked.connect(self._reset_to_defaults)
        btn_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("適用")
        apply_btn.clicked.connect(self.accept)
        apply_btn.setDefault(True)
        btn_layout.addWidget(apply_btn)
        
        main_layout.addLayout(btn_layout)
    
    def _reset_to_defaults(self):
        """すべてデフォルト値に戻す。"""
        defaults = {
            "phase1_beam_size": 5,
            "phase2_detect_beam_size": 1,
            "phase2_retranscribe_beam_size": 5,
            "ambiguous_threshold": 70.0,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "repetition_penalty": 1.0,
            "speech_pad_ms": 400,
            "duplicate_merge": True,
            "debug_prob_log": False,

            "log_file_enabled": False,
            "log_level": "INFO",
            "log_file_path": "app.log",
            # UI では KB 単位で扱う (ここでは KB をデフォルトとして保持)
            "log_max_bytes": 1024,
            "log_backup_count": 5,
            "available_languages": [
                "en", "es", "it", "ja", "de", "zh",
                "ru", "ko", "pt", "fr", "pl", "nl",
            ],
        }
        
        for key, default_val in defaults.items():
            widget = self.widgets.get(key)
            if widget:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(default_val)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(default_val))
                elif key == "available_languages":
                    default_set = {str(x) for x in (default_val or [])}
                    for code, chk in self.lang_checks.items():
                        chk.setChecked(code in default_set)
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(str(default_val))
                else:
                    widget.setValue(default_val)
    
    def get_values(self) -> dict:
        """現在のウィジェット値を辞書で返す。"""
        result = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif key == "available_languages":
                result[key] = [code for code, chk in self.lang_checks.items() if chk.isChecked()]
            elif isinstance(widget, QLineEdit):
                result[key] = widget.text()
            elif isinstance(widget, QComboBox):
                result[key] = widget.currentText()
            else:
                # 特殊処理: UI 上は log_max_bytes を KB 単位で扱うが、内部/設定は bytes を期待する
                if key == "log_max_bytes":
                    try:
                        kb = int(widget.value())
                        result[key] = kb * 1024
                    except Exception:
                        result[key] = int(widget.value())
                else:
                    result[key] = widget.value()
        return result
