"""隠しパラメータ設定ダイアログ。

config.toml の [hidden] セクションを GUI から編集可能にします。
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
    QPushButton,
    QGroupBox,
    QFormLayout,
    QScrollArea,
    QSizePolicy,
    QWidget,
)
from PySide6.QtCore import Qt


class HiddenParamsDialog(QDialog):
    """隠しパラメータ編集ダイアログ。"""

    def __init__(self, current_hidden: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("隠しパラメータ設定")
        self.setMinimumSize(600, 550)
        
        self.current_hidden = current_hidden.copy()
        self.widgets = {}
        
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
            "上級者向けパラメータです。変更後は「適用」を押すと config.toml の [hidden] セクションに保存されます。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { color: #888; margin-bottom: 10px; }")
        main_layout.addWidget(info)
        
        # スクロールエリア
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # ---- Beam Size グループ ----
        beam_group = QGroupBox("Beam Size 設定")
        beam_layout = QFormLayout()
        
        params = [
            ("phase1_beam_size", "Phase 1（全体転写）", 1, 10, 5, 
               "全体転写時のbeam_size。大きいほど高精度だが処理時間増加。\n"
               "範囲: 1〜10"),
            ("phase2_detect_beam_size", "Phase 2（言語検出）", 1, 10, 1,
               "クリップ単位の言語検出時のbeam_size。通常は1で十分。\n"
               "範囲: 1〜10"),
            ("phase2_retranscribe_beam_size", "Phase 2（再転写）", 1, 10, 5,
               "RU判定クリップやあいまいクリップの再転写時のbeam_size。\n"
               "範囲: 1〜10"),
        ]
        
        for key, label, min_val, max_val, default, tooltip in params:
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(self.current_hidden.get(key, default))
            spin.setToolTip(tooltip)
            self.widgets[key] = spin
            
            extra = QLabel(f"（デフォルト: {default}）")
            beam_layout.addRow(f"{label}:", row_with_help(spin, tooltip, extra))
        
        beam_group.setLayout(beam_layout)
        scroll_layout.addWidget(beam_group)
        
        # ---- 言語判定グループ ----
        lang_group = QGroupBox("言語判定")
        lang_layout = QFormLayout()
        
        # ambiguous_threshold
        ambig_spin = QDoubleSpinBox()
        ambig_spin.setRange(0.0, 100.0)
        ambig_spin.setDecimals(1)
        ambig_spin.setSingleStep(0.1)
        ambig_spin.setValue(self.current_hidden.get("ambiguous_threshold", 70.0))
        ambig_spin.setToolTip(
            "JA/RU確率差がこの値未満なら両言語で転写（あいまい判定）。\n"
            "大きいほど両言語転写する頻度が上がるが、処理時間も増加。\n"
            "単位: pt / 範囲: 0.0〜100.0"
        )
        self.widgets["ambiguous_threshold"] = ambig_spin
        
        ambig_extra = QLabel("pt（デフォルト: 70.0）")
        lang_layout.addRow(
            "あいまい判定しきい値:",
            row_with_help(ambig_spin, ambig_spin.toolTip(), ambig_extra),
        )
        
        lang_group.setLayout(lang_layout)
        scroll_layout.addWidget(lang_group)
        
        # ---- その他のパラメータグループ ----
        other_group = QGroupBox("その他")
        other_layout = QFormLayout()
        
        # condition_on_previous_text
        cond_check = QCheckBox()
        cond_check.setChecked(self.current_hidden.get("condition_on_previous_text", True))
        cond_check.setToolTip(
            "前セグメントのテキストを次セグメントの文脈として使用。\n"
            "繰り返しハルシネーションが多い場合は false にすると改善する場合がある。"
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
            "圧縮比がこの値を超えるセグメントをハルシネーション判定。\n"
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
        scroll_layout.addWidget(other_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
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
        }
        
        for key, default_val in defaults.items():
            widget = self.widgets.get(key)
            if widget:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(default_val)
                else:
                    widget.setValue(default_val)
    
    def get_values(self) -> dict:
        """現在のウィジェット値を辞書で返す。"""
        result = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            else:
                result[key] = widget.value()
        return result
