"""テキスト編集 / カーソル位置分割ダイアログ。

`EditDialog` はセグメント 1 件に対して:
  1. テキスト (LANG1/LANG2) の編集
  2. カーソル位置を基準とした 2 分割

のいずれかを行うモーダルダイアログ。

使用例::

    dlg = EditDialog(seg, parent=self, lang1_code='ja', lang2_code='ru')
    if dlg.exec() == QDialog.Accepted:
        if dlg.mode == 'edit':
            seg['text_lang1'] = dlg.new_text_lang1
            seg['text_lang2'] = dlg.new_text_lang2
            seg['chosen_language'] = dlg.chosen_language
        elif dlg.mode == 'split':
            perform_split(dlg.split_which, dlg.split_pos)
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
    """セグメントのテキスト編集または分割を行うダイアログ。

    Attributes
    ----------
    mode : str | None
        'edit' または 'split'。accept 前は None。
    new_text_lang1 : str
        編集後の言語1テキスト (mode='edit' 時のみ有効)。
    new_text_lang2 : str
        編集後の言語2テキスト (mode='edit' 時のみ有効)。
    chosen_language : str
        表示言語コード (lang1_code or lang2_code)。
    split_pos : int
        分割文字位置 (mode='split' 時のみ有効)。
    split_which : str
        分割基準言語 ('lang1' or 'lang2')。
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
        self.mode: str | None = None
        self.new_text_lang1: str = seg.get('text_lang1', '') or ''
        self.new_text_lang2: str = seg.get('text_lang2', '') or ''
        self.chosen_language: str = self._resolve_chosen_language(seg)
        self.split_pos: int = 0
        self.split_which: str = 'lang1'

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
        guide = QLabel(
            "この画面では 1) テキスト編集  2) カーソル位置で分割 のいずれかを行えます。\n"
            "分割モード選択時はカーソル位置で 2 つに分割 (先頭/末尾は不可)。"
        )
        layout.addWidget(guide)

        # カーソル位置情報ラベル (分割モード時に表示)
        self._cursor_info_lbl = QLabel("")
        self._cursor_info_lbl.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(self._cursor_info_lbl)

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

        # ---- 操作モード選択 ----
        mode_box = QWidget()
        mode_h = QHBoxLayout(mode_box)
        mode_h.setContentsMargins(0, 0, 0, 0)
        self._rb_edit  = QRadioButton("編集")
        self._rb_split = QRadioButton("カーソル位置で分割")
        self._rb_edit.setChecked(True)
        bg_mode = QButtonGroup(self)
        bg_mode.addButton(self._rb_edit)
        bg_mode.addButton(self._rb_split)
        mode_h.addWidget(self._rb_edit)
        mode_h.addWidget(self._rb_split)
        mode_h.addStretch()
        layout.addWidget(mode_box)

        # ---- OK / キャンセル ----
        btn_layout = QHBoxLayout()
        ok_btn     = QPushButton("OK")
        cancel_btn = QPushButton("キャンセル")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # ---- シグナル接続 ----
        self._rb_edit.toggled.connect(self._update_editability)
        self._rb_split.toggled.connect(self._update_editability)
        self._rb_split.toggled.connect(self._update_cursor_info)

        # カーソル移動でタイム推定を更新
        self._lang1_edit.cursorPositionChanged.connect(self._update_cursor_info)
        self._lang1_edit.textChanged.connect(self._update_cursor_info)
        self._lang1_radio.toggled.connect(self._update_cursor_info)
        if self._lang2_edit and self._lang2_radio:
            self._lang2_edit.cursorPositionChanged.connect(self._update_cursor_info)
            self._lang2_edit.textChanged.connect(self._update_cursor_info)
            self._lang2_radio.toggled.connect(self._update_cursor_info)

        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)

        self._update_editability()
        self._update_cursor_info()

    def _update_editability(self) -> None:
        """分割モード時はテキストエディタを読み取り専用に切り替える。"""
        is_split = self._rb_split.isChecked()
        editors = [self._lang1_edit]
        if self._lang2_edit:
            editors.append(self._lang2_edit)
        for ed in editors:
            if is_split:
                ed.setReadOnly(True)
                ed.setTextInteractionFlags(
                    Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
                )
            else:
                ed.setReadOnly(False)
                ed.setTextInteractionFlags(Qt.TextEditorInteraction)

    def _active_edit(self) -> tuple[QTextEdit, str]:
        """現在フォーカスまたはラジオで選択中のエディタと言語ID ('lang1'/'lang2') を返す。"""
        if self._lang1_edit.hasFocus():
            return self._lang1_edit, 'lang1'
        if self._lang2_edit and self._lang2_edit.hasFocus():
            return self._lang2_edit, 'lang2'
        # フォーカスなし → ラジオで判断
        if self._lang2_radio and self._lang2_radio.isChecked():
            return (self._lang2_edit or self._lang1_edit), 'lang2'
        return self._lang1_edit, 'lang1'

    def _update_cursor_info(self) -> None:
        """分割モード時、カーソル位置から推定時刻を算出してラベルに表示。"""
        if not self._rb_split.isChecked():
            self._cursor_info_lbl.setText("")
            return
        edit, which = self._active_edit()
        base = edit.toPlainText()
        if not base:
            self._cursor_info_lbl.setText("カーソル: (テキスト空)")
            return
        pos = edit.textCursor().position()
        start = float(self._seg.get('start', 0.0))
        end   = float(self._seg.get('end', start))
        dur   = max(0.0, end - start)
        if pos <= 0:
            t = start
        elif pos >= len(base):
            t = end
        else:
            t = start + dur * (pos / len(base))
        self._cursor_info_lbl.setText(
            f"カーソル位置: {pos}/{len(base)}  推定時間: {t:.2f}s  (基準: {which.upper()})"
        )

    def _on_ok(self) -> None:
        """OK ボタン押下時の処理。モードに応じて結果フィールドをセットして accept。"""
        if self._rb_split.isChecked():
            edit, which = self._active_edit()
            pos = edit.textCursor().position()
            full = edit.toPlainText()
            if pos <= 0 or pos >= len(full):
                QMessageBox.warning(self, "分割不可", "先頭/末尾では分割できません。")
                return
            self.mode        = 'split'
            self.split_pos   = pos
            self.split_which = which  # 'lang1' or 'lang2'
        else:
            self.mode            = 'edit'
            self.new_text_lang1  = self._lang1_edit.toPlainText()
            self.new_text_lang2  = self._lang2_edit.toPlainText() if self._lang2_edit else ''
            if self._lang2_radio and self._lang2_radio.isChecked():
                self.chosen_language = self._lang2_code or self._lang1_code
            else:
                self.chosen_language = self._lang1_code
        self.accept()
