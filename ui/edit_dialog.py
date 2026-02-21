"""テキスト編集 / カーソル位置分割ダイアログ。

`EditDialog` はセグメント 1 件に対して:
  1. テキスト (JA/RU) の編集
  2. カーソル位置を基準とした 2 分割

のいずれかを行うモーダルダイアログ。

使用例::

    dlg = EditDialog(seg, parent=self)
    if dlg.exec() == QDialog.Accepted:
        if dlg.mode == 'edit':
            seg['text_ja'] = dlg.new_text_ja
            seg['text_ru'] = dlg.new_text_ru
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
from PySide6.QtCore import Qt


class EditDialog(QDialog):
    """セグメントのテキスト編集または分割を行うダイアログ。

    Attributes
    ----------
    mode : str | None
        'edit' または 'split'。accept 前は None。
    new_text_ja : str
        編集後の日本語テキスト (mode='edit' 時のみ有効)。
    new_text_ru : str
        編集後のロシア語テキスト (mode='edit' 時のみ有効)。
    chosen_language : str
        表示言語コード ('ja' or 'ru')。
    split_pos : int
        分割文字位置 (mode='split' 時のみ有効)。
    split_which : str
        分割基準言語 ('ja' or 'ru')。
    """

    def __init__(self, seg: dict, parent=None, dialog_size: tuple[int, int] = (640, 200)):
        super().__init__(parent)
        self.setWindowTitle("編集画面")
        self.resize(*dialog_size)

        # 結果フィールド (accept 後に参照する)
        self.mode: str | None = None
        self.new_text_ja: str = seg.get('text_ja', '') or ''
        self.new_text_ru: str = seg.get('text_ru', '') or ''
        self.chosen_language: str = self._resolve_chosen_language(seg)
        self.split_pos: int = 0
        self.split_which: str = 'ja'

        self._seg = seg
        self._build_ui()

    # ------------------------------------------------------------------ #
    # 内部ヘルパー                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_chosen_language(seg: dict) -> str:
        lang = seg.get('chosen_language')
        if lang in ('ja', 'ru'):
            return lang
        return 'ja' if seg.get('ja_prob', 0.0) >= seg.get('ru_prob', 0.0) else 'ru'

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

        # ---- JA ブロック ----
        ja_block = QWidget()
        ja_bl = QVBoxLayout(ja_block)
        ja_bl.setContentsMargins(0, 0, 0, 0)
        self._ja_radio = QRadioButton("JA")
        ja_bl.addWidget(self._ja_radio, alignment=Qt.AlignLeft)
        self._ja_edit = QTextEdit()
        self._ja_edit.setPlainText(self._seg.get('text_ja', '') or '')
        self._ja_edit.setMaximumHeight(120)
        ja_bl.addWidget(self._ja_edit)

        # ---- RU ブロック ----
        ru_block = QWidget()
        ru_bl = QVBoxLayout(ru_block)
        ru_bl.setContentsMargins(0, 0, 0, 0)
        self._ru_radio = QRadioButton("RU")
        ru_bl.addWidget(self._ru_radio, alignment=Qt.AlignLeft)
        self._ru_edit = QTextEdit()
        self._ru_edit.setPlainText(self._seg.get('text_ru', '') or '')
        self._ru_edit.setMaximumHeight(120)
        ru_bl.addWidget(self._ru_edit)

        # ラジオグループ
        bg_lang = QButtonGroup(self)
        bg_lang.addButton(self._ja_radio)
        bg_lang.addButton(self._ru_radio)
        if self.chosen_language == 'ru':
            self._ru_radio.setChecked(True)
        else:
            self._ja_radio.setChecked(True)

        layout.addWidget(ja_block)
        layout.addWidget(ru_block)

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

        # フォーカス移動で言語ラジオを追随
        orig_ja_focus = self._ja_edit.focusInEvent
        orig_ru_focus = self._ru_edit.focusInEvent
        def ja_focus(ev):
            orig_ja_focus(ev)
            self._ja_radio.setChecked(True)
        def ru_focus(ev):
            orig_ru_focus(ev)
            self._ru_radio.setChecked(True)
        self._ja_edit.focusInEvent = ja_focus
        self._ru_edit.focusInEvent = ru_focus

        # カーソル移動でタイム推定を更新
        self._ja_edit.cursorPositionChanged.connect(self._update_cursor_info)
        self._ru_edit.cursorPositionChanged.connect(self._update_cursor_info)
        self._ja_edit.textChanged.connect(self._update_cursor_info)
        self._ru_edit.textChanged.connect(self._update_cursor_info)
        self._ja_radio.toggled.connect(self._update_cursor_info)
        self._ru_radio.toggled.connect(self._update_cursor_info)

        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)

        self._update_editability()
        self._update_cursor_info()

    def _update_editability(self) -> None:
        """分割モード時はテキストエディタを読み取り専用に切り替える。"""
        is_split = self._rb_split.isChecked()
        for ed in (self._ja_edit, self._ru_edit):
            if is_split:
                ed.setReadOnly(True)
                ed.setTextInteractionFlags(
                    Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
                )
            else:
                ed.setReadOnly(False)
                ed.setTextInteractionFlags(Qt.TextEditorInteraction)

    def _active_edit(self) -> tuple[QTextEdit, str]:
        """現在フォーカスまたはラジオで選択中のエディタと言語コードを返す。"""
        if self._ja_edit.hasFocus():
            return self._ja_edit, 'ja'
        if self._ru_edit.hasFocus():
            return self._ru_edit, 'ru'
        # フォーカスなし → ラジオで判断
        if self._ru_radio.isChecked():
            return self._ru_edit, 'ru'
        return self._ja_edit, 'ja'

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
            self.mode       = 'split'
            self.split_pos  = pos
            self.split_which = which
        else:
            self.mode           = 'edit'
            self.new_text_ja    = self._ja_edit.toPlainText()
            self.new_text_ru    = self._ru_edit.toPlainText()
            self.chosen_language = 'ru' if self._ru_radio.isChecked() else 'ja'
        self.accept()
