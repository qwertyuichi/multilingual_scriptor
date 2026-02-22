"""テキスト編集ダイアログ。

`EditDialog` はセグメント 1 件に対して、テキスト (LANG1/LANG2) の編集を行うモーダルダイアログ。

使用例::

    dlg = EditDialog(seg, parent=self, lang1_code='ja', lang2_code='ru')
    if dlg.exec() == QDialog.Accepted:
        seg['text_lang1'] = dlg.new_text_lang1
        seg['text_lang2'] = dlg.new_text_lang2
        seg['chosen_language'] = dlg.chosen_language
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QWidget,
    QMessageBox,
)
from PySide6.QtCore import Qt, QObject, QEvent


class _FocusFilter(QObject):
    """focusInEvent をモンキーパッチせず QObject.installEventFilter で処理 (BUG-5 修正)。"""

    def __init__(self, radio: QRadioButton, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._radio = radio

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.FocusIn:
            self._radio.setChecked(True)
        return False  # イベントを伝播させる


class EditDialog(QDialog):
    """セグメントのテキスト編集を行うダイアログ。

    Attributes
    ----------
    new_text_lang1 : str
        編集後の言語1テキスト。
    new_text_lang2 : str
        編集後の言語2テキスト。
    chosen_language : str
        表示言語コード (lang1_code or lang2_code)。
    """

    def __init__(
        self,
        seg: dict,
        parent=None,
        dialog_size: tuple[int, int] = (640, 200),
        lang1_code: str = 'ja',
        lang2_code: str | None = 'ru',
    ):
        super().__init__(parent)
        self.setWindowTitle("編集画面")
        self.resize(*dialog_size)

        self._lang1_code = lang1_code
        self._lang2_code = lang2_code

        # 結果フィールド (accept 後に参照する)
        self.new_text_lang1: str = seg.get('text_lang1', '') or ''
        self.new_text_lang2: str = seg.get('text_lang2', '') or ''
        self.chosen_language: str = self._resolve_chosen_language(seg)

        self._seg = seg
        self._build_ui()

    # ------------------------------------------------------------------ #
    # 内部ヘルパー                                                          #
    # ------------------------------------------------------------------ #

    def _resolve_chosen_language(self, seg: dict) -> str:
        lang = seg.get('chosen_language')
        if lang == self._lang1_code:
            return self._lang1_code
        if self._lang2_code and lang == self._lang2_code:
            return self._lang2_code
        # fallback: prob
        p1 = seg.get('lang1_prob', 0.0)
        p2 = seg.get('lang2_prob', 0.0)
        return self._lang1_code if p1 >= p2 else (self._lang2_code or self._lang1_code)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 操作説明
        guide = QLabel("この画面でテキストを編集できます。")
        layout.addWidget(guide)

        lbl1 = self._lang1_code.upper()
        lbl2 = self._lang2_code.upper() if self._lang2_code else None

        # ---- LANG1 ブロック ----
        block1 = QWidget()
        bl1 = QVBoxLayout(block1)
        bl1.setContentsMargins(0, 0, 0, 0)
        self._lang1_radio = QRadioButton(lbl1)
        bl1.addWidget(self._lang1_radio, alignment=Qt.AlignLeft)
        self._lang1_edit = QTextEdit()
        self._lang1_edit.setPlainText(self._seg.get('text_lang1', '') or '')
        self._lang1_edit.setMaximumHeight(120)
        bl1.addWidget(self._lang1_edit)

        # ---- LANG2 ブロック (lang2 があるときのみ) ----
        self._lang2_radio: QRadioButton | None = None
        self._lang2_edit: QTextEdit | None = None
        block2: QWidget | None = None
        if lbl2:
            block2 = QWidget()
            bl2 = QVBoxLayout(block2)
            bl2.setContentsMargins(0, 0, 0, 0)
            self._lang2_radio = QRadioButton(lbl2)
            bl2.addWidget(self._lang2_radio, alignment=Qt.AlignLeft)
            self._lang2_edit = QTextEdit()
            self._lang2_edit.setPlainText(self._seg.get('text_lang2', '') or '')
            self._lang2_edit.setMaximumHeight(120)
            bl2.addWidget(self._lang2_edit)

        # ラジオグループ
        bg_lang = QButtonGroup(self)
        bg_lang.addButton(self._lang1_radio)
        if self._lang2_radio:
            bg_lang.addButton(self._lang2_radio)

        if self._lang2_radio and self.chosen_language == self._lang2_code:
            self._lang2_radio.setChecked(True)
        else:
            self._lang1_radio.setChecked(True)

        layout.addWidget(block1)
        if block2 is not None:
            layout.addWidget(block2)

        # BUG-5 修正: focusInEvent モンキーパッチ → QObject.installEventFilter
        self._ff1 = _FocusFilter(self._lang1_radio, self)
        self._lang1_edit.installEventFilter(self._ff1)
        if self._lang2_edit and self._lang2_radio:
            self._ff2 = _FocusFilter(self._lang2_radio, self)
            self._lang2_edit.installEventFilter(self._ff2)



        # ---- OK / キャンセル ----
        btn_layout = QHBoxLayout()
        ok_btn     = QPushButton("OK")
        cancel_btn = QPushButton("キャンセル")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # ---- シグナル接続 ----
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)



    def _on_ok(self) -> None:
        """OK ボタン押下時の処理。編集内容を確定して accept。"""
        self.new_text_lang1 = self._lang1_edit.toPlainText()
        self.new_text_lang2 = self._lang2_edit.toPlainText() if self._lang2_edit else ''
        if self._lang2_radio and self._lang2_radio.isChecked():
            self.chosen_language = self._lang2_code or self._lang1_code
        else:
            self.chosen_language = self._lang1_code
        self.accept()
