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
    QSplitter,
    QProgressBar,
    QComboBox,
    QCheckBox,
    QScrollArea,
    QGridLayout,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
)
from PySide6.QtCore import Qt, QUrl, Slot, QLoggingCategory, QTimer
import logging
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from transcriber import TranscriptionThread
from models.segment import Segment, as_segment_list
import torch
from utils.timefmt import format_ms, parse_to_ms, to_srt_timestamp
from utils.segment_utils import display_text, normalize_segment_id
from exporter import build_export_text, build_json_payload
from ui.table_presenter import rebuild_aggregate_text, populate_table
from services.segment_ops import split_segment_at_position, adjust_boundary
from services.retranscribe_ops import dynamic_time_split, merge_contiguous_segments


def get_whisper_model_names():
    """利用可能なWhisper公式モデル名一覧を返す (インライン定義)。
    以前は `whisper_model_list.py` からインポートしていたが簡素化のため統合。
    必要ならここで並び順を調整可能。
    """
    return [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
        "turbo",
    ]


import os
import tomllib as _toml

# Qt Multimedia FFmpegログを非表示
QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg=false")


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
        # 手動分割ダイアログ既定サイズ (幅, 高さ) 必要に応じて変更
        self.split_dialog_size = (640, 200)

        # 設定ロード
        try:
            self.config = self.load_config()
        except Exception as e:
            # 最低限のフォールバック設定
            logging.getLogger(__name__).error(f"設定ロード失敗: {e}")
            self.config = {"default": {}}

        # UI 初期化とシグナル接続
        self.init_ui()
        self.connect_signals()
        # テーブル行クリック: シークのみ
        self.transcription_table.cellClicked.connect(self.seek_to_table_row)
        # ダブルクリック: 行開始位置にシークして再生開始（編集ダイアログは右クリック/ボタンからのみ）
        self.transcription_table.cellDoubleClicked.connect(self._play_row_on_doubleclick)
        # 選択変更で編集ボタンの有効/無効を更新
        self.transcription_table.itemSelectionChanged.connect(self._update_split_button_state)

        # 初期ウィンドウサイズと位置（最小サイズ設定でレイアウト崩れ防止）
        self.setMinimumSize(800, 600)
        self.resize(1280, 800)
        # 画面左上へ移動 (0,0)
        self.move(0, 0)

    def seek_to_table_row(self, row, col):
        # START列の値を取得し、hh:mm:ss→秒に変換してシーク
        start_item = self.transcription_table.item(row, 0)
        if not start_item:
            return
        start_str = start_item.text()
        ms = parse_to_ms(start_str)
        try:
            self.media_player.setPosition(ms)
        except Exception:
            pass

    def _play_row_on_doubleclick(self, row: int, col: int):
        """テーブルをダブルクリックした行の開始位置へシークし、再生を開始する。"""
        if not self.current_video_path:
            return
        # まずシーク
        self.seek_to_table_row(row, col)
        # 再生状態でなければ再生
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            if self.media_player.playbackState() != QMediaPlayer.PlayingState:
                self.media_player.play()
                # 再生ボタンのアイコン更新
                if hasattr(self, 'play_button'):
                    self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        except Exception:
            pass

    # ---------------- 手動分割機能 ここから ----------------
    def open_split_dialog_for_row(self, row: int, col: int):
        """(旧API) 編集画面を開くラッパー。互換のため残置。"""
        return self.open_edit_dialog_for_row(row, col)

    def open_edit_dialog_for_row(self, row: int, col: int):
        """編集画面: テキスト編集 / カーソル位置分割。表示言語は JA/RU ラジオで選択しフォーカスとも連動。"""
        if not self.transcription_result:
            return
        segs = self.transcription_result.get('segments', [])
        if row < 0 or row >= len(segs):
            return
        seg = segs[row]
        text_ja = seg.get('text_ja', '')
        text_ru = seg.get('text_ru', '')
        # 以前は「どちらも空なら編集不要」として return していたが、
        # テキスト消去後に再入力したいケースを許容するため空でも編集ダイアログを開く。

        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QMessageBox, QCheckBox
        from PySide6.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("編集画面")
        # サイズ指定（高さを抑えたい場合は self.split_dialog_size を調整）
        try:
            if hasattr(self, 'split_dialog_size') and self.split_dialog_size:
                w, h = self.split_dialog_size
                dlg.resize(w, h)
        except Exception:
            pass
        layout = QVBoxLayout(dlg)
        info_lbl = QLabel("この画面では 1) テキスト編集  2) カーソル位置で分割 のいずれかを行えます。\n分割モード選択時はカーソル位置で2つに分割 (先頭/末尾は不可)。")
        layout.addWidget(info_lbl)
        cursor_info_lbl = QLabel("")
        cursor_info_lbl.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(cursor_info_lbl)
        # JA/RU テキスト + 個別ラジオ（各テキスト上）
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        # ラッパー
        lang_container = QWidget(); lang_container_layout = QVBoxLayout(lang_container)
        lang_container_layout.setContentsMargins(0,0,0,0)

        # JA ブロック
        ja_block = QWidget(); ja_block_layout = QVBoxLayout(ja_block); ja_block_layout.setContentsMargins(0,0,0,0)
        ja_radio = QRadioButton("JA")
        ja_block_layout.addWidget(ja_radio, alignment=Qt.AlignLeft)
        ja_edit = QTextEdit(); ja_edit.setPlainText(text_ja); ja_edit.setMaximumHeight(120)
        ja_block_layout.addWidget(ja_edit)

        # RU ブロック
        ru_block = QWidget(); ru_block_layout = QVBoxLayout(ru_block); ru_block_layout.setContentsMargins(0,0,0,0)
        ru_radio = QRadioButton("RU")
        ru_block_layout.addWidget(ru_radio, alignment=Qt.AlignLeft)
        ru_edit = QTextEdit(); ru_edit.setPlainText(text_ru); ru_edit.setMaximumHeight(120)
        ru_block_layout.addWidget(ru_edit)

        # ボタングループ化
        bg_lang = QButtonGroup(dlg); bg_lang.addButton(ja_radio); bg_lang.addButton(ru_radio)
        chosen_lang = seg.get('chosen_language')
        if chosen_lang not in ('ja','ru'):
            chosen_lang = 'ja' if seg.get('ja_prob',0.0) >= seg.get('ru_prob',0.0) else 'ru'
        (ja_radio if chosen_lang=='ja' else ru_radio).setChecked(True)

        lang_container_layout.addWidget(ja_block)
        lang_container_layout.addWidget(ru_block)
        layout.addWidget(lang_container)

        # モード（編集 / カーソル位置で分割）
        mode_box = QWidget(); mode_h = QHBoxLayout(mode_box); mode_h.setContentsMargins(0,0,0,0)
        rb_mode_edit = QRadioButton("編集")
        rb_mode_split = QRadioButton("カーソル位置で分割")
        rb_mode_edit.setChecked(True)
        bg_mode = QButtonGroup(dlg); bg_mode.addButton(rb_mode_edit); bg_mode.addButton(rb_mode_split)
        mode_h.addWidget(rb_mode_edit); mode_h.addWidget(rb_mode_split); mode_h.addStretch()
        layout.addWidget(mode_box)

        btn_layout = QHBoxLayout()
        action_btn = QPushButton("OK")
        cancel_btn = QPushButton("キャンセル")
        btn_layout.addWidget(action_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # 状態保持
        action_state = {'mode': None, 'split_pos': None, 'split_which': None}

        def update_editability():
            if rb_mode_split.isChecked():
                for ed in (ja_edit, ru_edit):
                    ed.setReadOnly(True)
                    ed.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            else:
                for ed in (ja_edit, ru_edit):
                    ed.setReadOnly(False)
                    ed.setTextInteractionFlags(Qt.TextEditorInteraction)
        rb_mode_edit.toggled.connect(lambda _: update_editability())
        rb_mode_split.toggled.connect(lambda _: update_editability())
        update_editability()

        # フォーカス時に言語ラジオを追随
        def focus_in_factory(radio_btn):
            def _slot():
                radio_btn.setChecked(True)
            return _slot
        ja_edit.focusInEvent_orig = ja_edit.focusInEvent
        ru_edit.focusInEvent_orig = ru_edit.focusInEvent
        def ja_focus(ev):
            ja_edit.focusInEvent_orig(ev)
            ja_radio.setChecked(True)
        def ru_focus(ev):
            ru_edit.focusInEvent_orig(ev)
            ru_radio.setChecked(True)
        ja_edit.focusInEvent = ja_focus
        ru_edit.focusInEvent = ru_focus

        # ラジオ切替時も背景はデフォルト（指定解除）
        def clear_highlight():
            ja_edit.setStyleSheet("")
            ru_edit.setStyleSheet("")
        ja_radio.toggled.connect(clear_highlight)
        ru_radio.toggled.connect(clear_highlight)
        clear_highlight()

        # 分割モード時カーソル→時間換算表示
        def update_cursor_time():
            if not rb_mode_split.isChecked():
                cursor_info_lbl.setText("")
                return
            edit = ja_edit if ja_edit.hasFocus() else (ru_edit if ru_edit.hasFocus() else ja_edit)
            which = 'ja' if edit is ja_edit else 'ru'
            base = edit.toPlainText()
            if not base:
                cursor_info_lbl.setText("カーソル: (テキスト空)")
                return
            pos = edit.textCursor().position()
            # 時間推定
            start = float(seg.get('start',0.0)); end = float(seg.get('end',start))
            dur = max(0.0, end-start)
            if pos <= 0:
                t = start
            elif pos >= len(base):
                t = end
            else:
                ratio = pos/len(base)
                t = start + dur*ratio
            cursor_info_lbl.setText(f"カーソル位置: {pos}/{len(base)}  推定時間: {t:.2f}s (基準:{which.upper()})")
        # イベント接続（キャレット移動）
        def install_cursor_tracker(edit):
            edit.cursorPositionChanged.connect(update_cursor_time)
        install_cursor_tracker(ja_edit)
        install_cursor_tracker(ru_edit)
        rb_mode_split.toggled.connect(update_cursor_time)
        ja_edit.textChanged.connect(update_cursor_time)
        ru_edit.textChanged.connect(update_cursor_time)
        ja_radio.toggled.connect(update_cursor_time)
        ru_radio.toggled.connect(update_cursor_time)
        # 初期
        update_cursor_time()

        def do_action():
            if rb_mode_split.isChecked():
                edit = ja_edit if ja_edit.hasFocus() else (ru_edit if ru_edit.hasFocus() else ja_edit)
                which = 'ja' if edit is ja_edit else 'ru'
                cursor = edit.textCursor()
                pos = cursor.position()
                full_text = edit.toPlainText()
                if pos <= 0 or pos >= len(full_text):
                    QMessageBox.warning(dlg, "分割不可", "先頭/末尾では分割できません。")
                    return
                action_state['mode'] = 'split'
                action_state['split_pos'] = pos
                action_state['split_which'] = which
                dlg.accept()
                return
            # 編集保存: 選択ラジオ言語に合わせて表示言語決定
            new_ja = ja_edit.toPlainText()
            new_ru = ru_edit.toPlainText()
            seg['text_ja'] = new_ja
            seg['text_ru'] = new_ru
            if ja_radio.isChecked():
                seg['chosen_language'] = 'ja'
                seg['text'] = new_ja
            elif ru_radio.isChecked():
                seg['chosen_language'] = 'ru'
                seg['text'] = new_ru
            self._rebuild_text_and_refresh()
            self.status_label.setText("編集を保存しました")
            action_state['mode'] = 'saved'
            dlg.accept()

        action_btn.clicked.connect(do_action)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return
        if action_state['mode'] == 'split':
            pos = action_state['split_pos']
            which = action_state['split_which']
            if pos is not None:
                # 分割後は常に前後半を再文字起こし
                self.perform_segment_split(row, which, pos)
        return

    def perform_segment_split(self, row: int, which: str, pos: int):
        """選択行セグメントを pos 位置で 2 分割し 前半/後半を再文字起こし。"""
        if not getattr(self, 'transcription_result', None):
            return
        original = self.transcription_result.get('segments', [])
        new_list = split_segment_at_position(original, row, which, pos)
        if new_list is original or len(new_list) == len(original):
            return
        self.transcription_result['segments'] = new_list
        front = new_list[row]
        back = new_list[row+1]
        re_jobs = [('front', front['start'], front['end']), ('back', back['start'], back['end'])]
        # 非同期連鎖実行
        self.range_retranscribing = True
        self._pending_rejobs = re_jobs
        self._split_row_base = row  # 後でどこに反映するか参照
        self._run_next_split_rejob()
        self._update_split_button_state()
    # ---------------- 手動分割機能 ここまで ----------------

    def _update_split_button_state(self):
        """分割/動的分割/削除ボタンの活性状態を更新。"""
        if not hasattr(self, 'split_button'):
            return
        busy = getattr(self, 'range_retranscribing', False)
        result = getattr(self, 'transcription_result', None)
        if not result:
            self.split_button.setEnabled(False)
            if hasattr(self, 'dynamic_split_button'):
                self.dynamic_split_button.setEnabled(False)
            if hasattr(self, 'delete_button'):
                self.delete_button.setEnabled(False)
            return
        segs = result.get('segments', [])
        rows = self._collect_selected_rows()
        # 分割: 1行選択 & 非処理中
        if busy or len(rows) != 1:
            self.split_button.setEnabled(False)
        else:
            r = rows[0]
            self.split_button.setEnabled(0 <= r < len(segs))
        # 動的分割/境界調整: 1行または2行連続
        if hasattr(self, 'dynamic_split_button'):
            can_dyn = False
            if not busy and rows:
                if len(rows) == 1 and 0 <= rows[0] < len(segs):
                    can_dyn = True
                elif len(rows) == 2 and rows[1] == rows[0] + 1:
                    r1, r2 = rows
                    if 0 <= r1 < len(segs) and 0 <= r2 < len(segs):
                        can_dyn = True
            self.dynamic_split_button.setEnabled(can_dyn)
        # 削除ボタン
        if hasattr(self, 'delete_button'):
            self.delete_button.setEnabled(bool(rows) and not busy and bool(segs))

    def _run_next_split_rejob(self):
        """分割後のキューに従って順次 RangeTranscriptionThread を起動。"""
        if not hasattr(self, '_pending_rejobs') or not self._pending_rejobs:
            # 全完了
            self.range_retranscribing = False
            self._rebuild_text_and_refresh()
            self.status_label.setText("分割＆再文字起こし完了")
            # ウォッチドッグ停止
            try:
                if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                    self._split_watchdog_timer.stop()
            except Exception:
                pass
            return
        job = self._pending_rejobs.pop(0)
        kind, start_sec, end_sec = job
        self.status_label.setText(f"{('前半' if kind=='front' else '後半')}再文字起こし開始…")
        self.progress_bar.setValue(0)
        from transcriber import RangeTranscriptionThread
        model_size = self.model_combo.currentText()
        device = self.device_combo.currentText()
        ja_w = self.ja_weight_slider.value()/100.0
        ru_w = self.ru_weight_slider.value()/100.0
        options = {
            'model': model_size,
            'device': device,
            'ja_weight': ja_w,
            'ru_weight': ru_w,
        }
        self._active_split_kind = kind
        self.range_thread = RangeTranscriptionThread(self.current_video_path, start_sec, end_sec, options)
        self.range_thread.progress.connect(self.on_range_progress)
        self.range_thread.status.connect(self.on_range_status)
        # range_finished に直接接続（旧 finished 廃止）
        self.range_thread.range_finished.connect(self._on_split_rejob_finished)
        self.range_thread.error.connect(self._on_split_rejob_error)
        self.range_thread.start()

    def _on_split_rejob_finished(self, seg: dict):
        """分割後の個別(前半/後半)再文字起こし完了ハンドラ。

        不具合: 以前の実装は (1) transcription_result への書き戻し、(2) テーブルセル更新、
        (3) 次ジョブ起動 / 最終確定 が行われずジョブキューが停止し `[再解析中]` が残留した。

        対応: 上記全処理を追加し、ウォッチドッグは最終完了時のみ停止する。
        """
        if not getattr(self, 'transcription_result', None):
            return
        kind = getattr(self, '_active_split_kind', None)
        base_row = getattr(self, '_split_row_base', None)
        if kind not in ('front', 'back') or base_row is None:
            return
        # セグメントリスト(オブジェクト化)
        segs_obj = as_segment_list(self.transcription_result.get('segments', []))
        target_index = base_row if kind == 'front' else base_row + 1
        if 0 <= target_index < len(segs_obj):
            tgt = segs_obj[target_index]
            tgt.update({
                'text': seg.get('text', ''),
                'text_ja': seg.get('text_ja', ''),
                'text_ru': seg.get('text_ru', ''),
                'ja_prob': float(seg.get('ja_prob', 0.0)),
                'ru_prob': float(seg.get('ru_prob', 0.0)),
                'chosen_language': seg.get('chosen_language') or seg.get('language') or tgt.get('chosen_language'),
            })
            # 空結果なら明示ラベル
            if not tgt.text_ja and not tgt.text_ru:
                tgt.text = tgt.text_ja = tgt.text_ru = '(空)'
                tgt.ja_prob = 0.0
                tgt.ru_prob = 0.0
        # 書き戻し (他箇所は dict を前提)
        try:
            self.transcription_result['segments'] = [s.to_dict() for s in segs_obj]
        except Exception:
            pass
        # 対象行だけセルを部分更新（全再描画コスト削減）
        try:
            if 0 <= target_index < self.transcription_table.rowCount():
                from PySide6.QtWidgets import QTableWidgetItem
                from PySide6.QtGui import QColor
                from utils.segment_utils import display_text
                row = target_index
                tgt = segs_obj[target_index]
                # 新規アイテム生成（既存オブジェクト再利用しない）
                ja_item = QTableWidgetItem(f"{tgt.ja_prob:.2f}")
                ru_item = QTableWidgetItem(f"{tgt.ru_prob:.2f}")
                if tgt.ja_prob >= tgt.ru_prob:
                    ja_item.setForeground(QColor(200,0,0))
                    ru_item.setForeground(QColor(0,0,180))
                else:
                    ru_item.setForeground(QColor(200,0,0))
                    ja_item.setForeground(QColor(0,0,180))
                txt_item = QTableWidgetItem(display_text(tgt))
                self.transcription_table.setItem(row, 2, ja_item)
                self.transcription_table.setItem(row, 3, ru_item)
                self.transcription_table.setItem(row, 4, txt_item)
        except Exception:
            try:
                self._rebuild_text_and_refresh()
            except Exception:
                pass
        # 次ジョブ有無
        if getattr(self, '_pending_rejobs', None):
            # まだ残り → 次へ (ウォッチドッグ継続: タイマー延長)
            try:
                from PySide6.QtCore import QTimer as _QTimer
                if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                    self._split_watchdog_timer.stop()
                self._split_watchdog_timer = _QTimer(self)
                self._split_watchdog_timer.setSingleShot(True)
                self._split_watchdog_timer.timeout.connect(self._check_split_watchdog)
                self._split_watchdog_timer.start(15000)
            except Exception:
                pass
            self._run_next_split_rejob()
            return
        # 全ジョブ完了 → 集約テキスト再構築 & ステータス更新
        try:
            self.range_retranscribing = False
            self._rebuild_text_and_refresh()
            self.status_label.setText('分割＆再文字起こし完了')
            self._update_split_button_state()
        except Exception:
            pass
        # ウォッチドッグ停止
        try:
            if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                self._split_watchdog_timer.stop()
        except Exception:
            pass
        # 後片付け
        self._active_split_kind = None
        try:
            del self.range_thread
        except Exception:
            pass

    def _on_split_rejob_error(self, err: str):
        """分割後再文字起こしジョブでエラー発生時の回復処理。"""
        try:
            if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                self._split_watchdog_timer.stop()
        except Exception:
            pass
        # ステータス表示とフラグ解除
        self.status_label.setText(f"分割再文字起こし失敗: {err}")
        self.range_retranscribing = False
        # 残りジョブは破棄
        if hasattr(self, '_pending_rejobs'):
            self._pending_rejobs.clear()
        # UI 再描画
        try:
            self._rebuild_text_and_refresh()
        except Exception:
            pass
        # ユーザー通知
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "エラー", f"分割後再文字起こしでエラーが発生しました:\n{err}")
        except Exception:
            pass

    def delete_selected_segments(self):
        """選択行のテキスト列を空にする（時間/ID/確率は保持）。"""
        if getattr(self, 'range_retranscribing', False):
            return
        if not getattr(self, 'transcription_result', None):
            return
        segs = self.transcription_result.get('segments', [])
        if not segs:
            return
        rows = self._collect_selected_rows()
        if not rows:
            return
        changed = 0
        for r in rows:
            if 0 <= r < len(segs):
                seg = segs[r]
                if any(seg.get(k) for k in ('text','text_ja','text_ru')):
                    seg['text'] = ''
                    seg['text_ja'] = ''
                    seg['text_ru'] = ''
                    changed += 1
        if changed == 0:
            return
        # 集約テキスト再構築
        try:
            from transcriber import rebuild_aggregate_text
            self.split_button.setEnabled(False)
        except Exception:
            pass
        # テーブル部分更新
        try:
            for r in rows:
                if 0 <= r < self.transcription_table.rowCount():
                    item = self.transcription_table.item(r, 4)
                    if item is None:
                        from PySide6.QtWidgets import QTableWidgetItem
                        item = QTableWidgetItem('')
                        self.transcription_table.setItem(r, 4, item)
                    else:
                        item.setText('')
        except Exception:
            self.display_transcription(self.transcription_result)
        self.status_label.setText(f"{changed}行のテキストを消去しました")
        if hasattr(self, 'delete_button'):
            self.delete_button.setEnabled(False)
        self._update_split_button_state()
        # 削除ボタン: 何か選択されていれば有効 (処理中除く)
        if hasattr(self, 'delete_button'):
            can_del = bool(rows) and not getattr(self, 'range_retranscribing', False) and bool(segs)
            self.delete_button.setEnabled(can_del)

    # 旧 GAP 機能は廃止済み

    def split_or_adjust_at_current_position(self):
        segs = as_segment_list(self.transcription_result.get('segments', [])) if getattr(self, 'transcription_result', None) else []
        """1行選択: その行を現在の再生位置で時間を基準に二分。
        2行連続選択: 境界(前行.end / 後行.start)を現在位置に移動。
        いずれも前後2区間を再文字起こし (前→後)。"""
        if getattr(self, 'range_retranscribing', False):
            return
        if not getattr(self, 'transcription_result', None):
            return
        # segs は Segment オブジェクト
        rows = self._collect_selected_rows()
        if not rows:
            return
        rows = sorted(rows)
        try:
            cur_ms = self.media_player.position()
        except Exception:
            return
        cur_sec = cur_ms / 1000.0
        # 1行選択: そのセグメント内でのみ有効
        if len(rows) == 1:
            r = rows[0]
            if r < 0 or r >= len(segs):
                return
            new_list, front_index = dynamic_time_split(self.transcription_result.get('segments', []), r, cur_sec)
            if new_list is None:
                return
            self.transcription_result['segments'] = new_list
            # 再文字起こしキュー設定（perform_segment_split と同等手順）
            front = new_list[front_index]
            back = new_list[front_index + 1]
            # 分割直後は旧テキスト断片を保持せず一旦空にしてプレースホルダ表示
            for segx in (front, back):
                # プレースホルダ明示: 完了前でも UI/編集ダイアログでわかるようにする
                segx['text'] = '[再解析中]'
                segx['text_ja'] = '[再解析中]'
                segx['text_ru'] = '[再解析中]'
                segx['ja_prob'] = 0.0
                segx['ru_prob'] = 0.0
                # 言語未確定扱い（確定前に display_text が ja/ru の空文字優先で消えるのを防ぐ）
                segx['chosen_language'] = None
            # 表示を即更新（[再解析中] プレースホルダ）
            try:
                # プレースホルダ文字列を TEXT 列へ直接入れる
                for idx_tmp, segx in ((front_index, front), (front_index+1, back)):
                    if 0 <= idx_tmp < self.transcription_table.rowCount():
                        from PySide6.QtWidgets import QTableWidgetItem
                        item = self.transcription_table.item(idx_tmp, 4)
                        if item is None:
                            item = QTableWidgetItem('[再解析中]')
                            self.transcription_table.setItem(idx_tmp, 4, item)
                        else:
                            item.setText('[再解析中]')
            except Exception:
                pass
            # 集約テキスト再構築
            try:
                from transcriber import rebuild_aggregate_text
                rebuild_aggregate_text(self.transcription_result)
            except Exception:
                pass
            self.range_retranscribing = True
            self._pending_rejobs = [
                ('front', front['start'], front['end']),
                ('back', back['start'], back['end'])
            ]
            self._split_row_base = front_index
            # ウォッチドッグ開始
            try:
                import time as _time
                self._split_watchdog_start = _time.time()
                from PySide6.QtCore import QTimer as _QTimer
                if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                    self._split_watchdog_timer.stop()
                self._split_watchdog_timer = _QTimer(self)
                self._split_watchdog_timer.setSingleShot(True)
                self._split_watchdog_timer.timeout.connect(self._check_split_watchdog)
                self._split_watchdog_timer.start(15000)
            except Exception:
                pass
            self._run_next_split_rejob()
            return
        # 2行連続選択: 境界移動
        if len(rows) == 2 and rows[1] == rows[0] + 1:
            r1, r2 = rows
            if r1 < 0 or r2 >= len(segs):
                return
            seg1 = segs[r1]; seg2 = segs[r2]
            start = float(seg1.get('start', 0.0))
            mid = float(seg1.get('end', start))
            end = float(seg2.get('end', mid))
            if not (start < cur_sec < end):
                return
            MIN_DUR = 0.2
            new_mid = cur_sec
            if new_mid - start < MIN_DUR or end - new_mid < MIN_DUR:
                return
            updated = adjust_boundary(self.transcription_result.get('segments', []), r1, new_mid, MIN_DUR)
            self.transcription_result['segments'] = updated
            # 再文字起こしキュー
            self.range_retranscribing = True
            self._pending_rejobs = [
                ('front', updated[r1]['start'], updated[r1]['end']),
                ('back', updated[r2]['start'], updated[r2]['end'])
            ]
            self._split_row_base = r1
            try:
                import time as _time
                self._split_watchdog_start = _time.time()
                from PySide6.QtCore import QTimer as _QTimer
                if hasattr(self, '_split_watchdog_timer') and self._split_watchdog_timer:
                    self._split_watchdog_timer.stop()
                self._split_watchdog_timer = _QTimer(self)
                self._split_watchdog_timer.setSingleShot(True)
                self._split_watchdog_timer.timeout.connect(self._check_split_watchdog)
                self._split_watchdog_timer.start(15000)
            except Exception:
                pass
            self._run_next_split_rejob()
            return
        return

    def sync_table_selection_with_position(self, position_ms):
        # 現在位置（ms→秒）に該当する行を選択＆スクロール
        pos_sec = position_ms / 1000.0
        row_to_select = None
        for row in range(self.transcription_table.rowCount()):
            start_item = self.transcription_table.item(row, 0)
            end_item = self.transcription_table.item(row, 1)
            if not start_item or not end_item:
                continue
            try:
                st_ms = parse_to_ms(start_item.text())
                ed_ms = parse_to_ms(end_item.text())
                st = st_ms / 1000.0
                ed = ed_ms / 1000.0
                if st <= pos_sec < ed:
                    row_to_select = row
                    break
            except Exception:
                continue
        if row_to_select is not None:
            self.transcription_table.setCurrentCell(row_to_select, 0)
            self.transcription_table.scrollToItem(
                self.transcription_table.item(row_to_select, 0)
            )

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
                "・用途: 設定ファイル構成の把握 / 今後の編集UI拡張の基礎"
            ),
            "デバイス": (
                "Whisper を実行する計算デバイス。\n"
                "cuda: GPU 使用 (推奨 / 速度向上)\ncpu: CPU フォールバック (低速)\n"
                "・CUDA が利用不可の場合は自動的に cpu のみ選択可"
            ),
            "トランスクリプションモデル": (
                "音声→文字変換に使用する Whisper モデルサイズ。\n"
            ),
            "セグメンテーションモデル": (
                "音声区間の切り出し(セグメント化)に用いるモデル。"
            ),
            "ja_weight": (
                "日本語言語判定スコアへの補正係数。\n"
                "最終確率 = 元スコア × weight を正規化後に比較。\n"
                "1.0 = 補正なし / >1.0 で日本語優遇 / <1.0 で抑制。\n"
                "・多言語混在で日本語が取りこぼされる場合は 1.1～1.3 程度を試す"
            ),
            "ru_weight": (
                "ロシア語判定スコア補正係数 (ja_weight と同様の計算)。\n"
                "・日本語優勢すぎてロシア語が検出されにくい場合に 1.1～1.4 など調整"
            ),
            "en_weight": (
                "英語判定スコア補正係数。\n"
                "・英語が断片的に含まれる配信で取りこぼしを防ぎたい場合に引き上げ"
            ),
            "min_seg_dur": (
                "1 つのセグメントがこれ未満秒なら分割候補から除外 (過分割防止)。\n"
                "推奨: 0.4～0.8 秒。短すぎるとノイズ増 / 長すぎると応答遅延"
            ),
            "mix_threshold": (
                "主要2言語(JA/RU) の確信度差 |JA-RU| がこの値未満なら [MIX] マークを付与。\n"
                "・値を小さく: MIX 判定減\n・値を大きく: MIX 判定増 (雑多表示の恐れ)"
            ),
            "ambiguous_threshold": (
                "主要言語間の確信度差がこの値未満で 'あいまい' とみなし、\n"
                "両(複数)言語再トライ比較を行う境界。\n"
                "・大きくすると再トライ頻度 ↑ (精度↑/速度↓)\n"
                "・小さくすると再トライ減 (速度↑/誤判定リスク↑)"
            ),
            "vad_level": (
                "WebRTC VAD (Voice Activity Detection) 感度。\n"
                "0: もっとも寛容 (雑音も音声扱い)\n3: 最も厳格 (静音判定が増える)\n"
                "・無音除去の厳しさを調整し、短いノイズを避けたい場合は 2～3"
            ),
            "gap_threshold": (
                "前セグメント終端と次セグメント開始の間隔がこの秒数以上なら '無音ギャップ' として扱う。\n"
                "include_silent 有効時のみログ出力。\n"
                "・MC/トーク番組: 0.3～0.7\n・間が長い朗読: 1.0 以上も検討"
            ),
            "include_silent": (
                "True で無音/スキップ理由 (短すぎる, silence 等) を出力ログに残す。\n"
                "解析/デバッグ用途向け。通常運用では False でログ簡潔化。"
            ),
            "srt_max_line": (
                "SRT 生成時、1 セグメントを強制的に改行分割する際の行数上限目安。\n"
                "・視認性を優先: 2～4\n・長文字幕許容: 6～8"
            ),
            "initial_prompt": (
                "最初の推論呼び出しに与えるコンテキスト文字列。\n"
                "固有名詞/話題/口調を誘導したい場合に使用。\n"
                "空文字なら無視。過剰に長いと逆効果の可能性。"
            ),
            "output_format": (
                "保存出力フォーマット。\n"
                "txt: 単純テキスト\nsrt: 字幕フォーマット (タイムコード付き)\njson: 構造化データ (後処理用)"
            ),
            "default_languages": (
                "起動時に ON にする言語コード配列。\n"
                "最低 1 つ必須。ここで指定されない言語は最初は未選択。\n"
                "多言語配信が確実な場合のみ 2 つ以上を推奨 (速度僅かに低下)"
            ),
            "設定項目": (
                "プロファイル (= 設定プリセット) を切り替えます。\n"
                "例: 'kapra' は配信向けチューニング、'default' は汎用。\n"
                "切替時に: デバイス / モデル / 言語初期選択 / ウェイト / しきい値類 が再読込されます。\n"
                "config.toml に新しいセクションを追加して独自プリセットを作成できます。"
            ),
        }

        def help_label(key: str):
            lbl = QLabel("?")
            lbl.setFixedWidth(16)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "QLabel { color: #007acc; font-weight: bold; border:1px solid #007acc; border-radius:8px; }"
            )
            if key in HELP_TEXTS:
                lbl.setToolTip(HELP_TEXTS[key])
            else:
                lbl.setToolTip(key)
            return lbl

        # 左側：ビデオ + コントロール (上) と テーブル (下) を縦方向 QSplitter で可変化
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        from PySide6.QtWidgets import QSplitter as _QSplitter
        vsplitter = _QSplitter(Qt.Vertical)
        # 上側コンテナ
        video_container = QWidget()
        video_vlayout = QVBoxLayout(video_container)
        video_vlayout.setContentsMargins(0,0,0,0)
        video_vlayout.setSpacing(4)
        # ビデオウィジェット
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.media_player.setVideoOutput(self.video_widget)
        video_vlayout.addWidget(self.video_widget, 1)
        # コントロールパネル
        control_layout = QVBoxLayout()

        # シークバー
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        control_layout.addWidget(self.position_slider)

        # 時間ラベル + 自動同期トグル
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00:00.000")
        self.total_time_label = QLabel("00:00:00.000")
        self.auto_sync_check = QCheckBox("自動同期")
        self.auto_sync_check.setChecked(True)
        self.auto_sync_check.setToolTip("再生中は現在位置に応じて行を自動選択します。停止/一時停止中は固定。")
        time_layout.addWidget(self.current_time_label)
        time_layout.addWidget(self.auto_sync_check)
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

        # 10秒戻る / 1秒戻る / 1秒進む / 10秒進む ボタン
        SEEK_BTN_WIDTH = 52
        def make_seek_btn(label: str, delta_ms: int, tooltip: str):
            btn = QPushButton(label)
            btn.setEnabled(False)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _=None, d=delta_ms: self.seek_relative(d))
            btn.setFixedWidth(SEEK_BTN_WIDTH)
            return btn
        self.seek_back_10_btn = make_seek_btn("◀10s", -10000, "10秒戻る")
        self.seek_back_1_btn = make_seek_btn("◀1s", -1000, "1秒戻る")
        self.seek_fwd_1_btn = make_seek_btn("1s▶", 1000, "1秒進む")
        self.seek_fwd_10_btn = make_seek_btn("10s▶", 10000, "10秒進む")
        button_layout.addWidget(self.seek_back_10_btn)
        button_layout.addWidget(self.seek_back_1_btn)
        button_layout.addWidget(self.seek_fwd_1_btn)
        button_layout.addWidget(self.seek_fwd_10_btn)
        # 右寄せ余白
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        video_vlayout.addLayout(control_layout)

        # 下側（テーブル）
        table_container = QWidget()
        table_vlayout = QVBoxLayout(table_container)
        table_vlayout.setContentsMargins(0, 0, 0, 0)
        table_vlayout.setSpacing(4)

        # テーブル上部 右寄せエクスポートバー
        top_export_bar = QHBoxLayout()
        top_export_bar.setContentsMargins(0, 0, 0, 0)
        top_export_bar.addStretch()
        self.partial_export_button = QPushButton("部分書き出し")
        self.partial_export_button.setEnabled(False)
        self.partial_export_button.setToolTip("選択された行範囲の音声(WAV)とテキスト(TXT)を ./output に書き出し")
        self.partial_export_button.clicked.connect(self.partial_export_selected)
        top_export_bar.addWidget(self.partial_export_button)
        self.export_button = QPushButton("全文書き出し...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_transcription)
        top_export_bar.addWidget(self.export_button)
        table_vlayout.addLayout(top_export_bar)

    # 旧: 編集/分割/再文字起こし/テキスト消去バーは削除（コンパクト化）

        self.transcription_table = QTableWidget()
        table_style = """
        QTableWidget::item:selected {
            background: #1976d2;
            color: #ffffff;
        }
        QTableWidget::item:focus {
            outline: none;
        }
        /* 行間圧縮: 上下パディング最小化 */
        QTableWidget::item {
            padding-top: 1px;
            padding-bottom: 1px;
        }
        """
        self.transcription_table.setStyleSheet(table_style)
        self.transcription_table.setColumnCount(5)
        self.transcription_table.setHorizontalHeaderLabels(["START", "END", "JA%", "RU%", "TEXT"])
        self.transcription_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transcription_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transcription_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.transcription_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.transcription_table.customContextMenuRequested.connect(self.show_table_context_menu)
        self.transcription_table.setColumnWidth(0, 80)
        self.transcription_table.setColumnWidth(1, 80)
        self.transcription_table.setColumnWidth(2, 60)
        self.transcription_table.setColumnWidth(3, 60)
        self.transcription_table.setColumnWidth(4, 400)
        header = self.transcription_table.horizontalHeader()
        for i in range(5):
            item = self.transcription_table.horizontalHeaderItem(i)
            if item:
                item.setTextAlignment(Qt.AlignCenter)
        header.setDefaultAlignment(Qt.AlignCenter)
        table_vlayout.addWidget(self.transcription_table, 1)
        # 行高（行間）をコンパクト化: フォント高さ + 4px
        vh = self.transcription_table.verticalHeader()
        compact_height = max(18, self.transcription_table.fontMetrics().height() + 4)
        vh.setDefaultSectionSize(compact_height)
        vh.setMinimumSectionSize(compact_height)

        # 最小高さ & 初期希望高さ設定
        video_container.setMinimumHeight(180)

        # vsplitter 組み立て
        vsplitter.addWidget(video_container)
        vsplitter.addWidget(table_container)
        vsplitter.setStretchFactor(0, 1)
        vsplitter.setStretchFactor(1, 4)
        try:
            vsplitter.setSizes([2000, 1200])
        except Exception:
            pass

        # ハンドル視認性向上
        vsplitter.setHandleWidth(6)
        vsplitter.setStyleSheet(
            """
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                              stop:0 #555, stop:0.5 #666, stop:1 #555);
                border-left: 1px solid #404040;
                border-right: 1px solid #404040;
                margin: 0px;
            }
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                              stop:0 #666, stop:0.5 #777, stop:1 #666);
            }
            QSplitter::handle:pressed {
                background: #2d89ef;
            }
            """
        )
        left_layout.addWidget(vsplitter, 1)

        # 右側：文字起こし設定
        right_widget = QWidget()
        # 幅固定 (横スクロールバー抑止のため上限=下限) ※必要に応じて値を調整
        fixed_settings_width = 350
        right_widget.setMinimumWidth(fixed_settings_width)
        right_widget.setMaximumWidth(fixed_settings_width)
        right_layout = QVBoxLayout(right_widget)

        # スクロールエリア
        scroll_area = QScrollArea()
        # 横スクロールバーを常に非表示
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # スクロール領域自体の幅を親に合わせる
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 基本設定
        model_group = QGroupBox("基本設定")
        model_layout = QGridLayout()

        label_setting_item = QLabel("プロファイル:")
        model_layout.addWidget(label_setting_item, 0, 0)
        self.profile_combo = QComboBox()
        # config の第一階層セクションをプロファイル候補とする
        self.profiles = [k for k, v in self.config.items() if isinstance(v, dict)]
        # default を先頭、それ以外はアルファベット順
        # 'kapra' を優先して先頭に、次に 'default'、その後アルファベット順
        ordered_profiles = [p for p in self.profiles if p == 'kapra'] + \
                           [p for p in self.profiles if p == 'default'] + \
                           sorted([p for p in self.profiles if p not in ('kapra','default')])
        self.profiles = ordered_profiles
        self.profile_combo.addItems(self.profiles)
        model_layout.addWidget(self.profile_combo, 0, 1)
        model_layout.addWidget(help_label("設定項目"), 0, 2)
        # 既定を 'kapra' に（無ければ 'default'）
        self.current_profile_name = 'kapra' if 'kapra' in self.profiles else 'default'
        self.profile_combo.setCurrentText(self.current_profile_name)

        dev_label = QLabel("デバイス:")
        model_layout.addWidget(dev_label, 1, 0)
        self.device_combo = QComboBox()
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        self.device_combo.addItems(devices)
        base_profile_dict = self.config.get(self.current_profile_name, self.config.get('default', {}))
        default_device = base_profile_dict.get("device", "cuda")
        if default_device not in devices:
            devices.append(default_device)
        self.device_combo.setCurrentText(default_device)
        model_layout.addWidget(self.device_combo, 1, 1)
        model_layout.addWidget(help_label("デバイス"), 1, 2)

        tmodel_label = QLabel("トランスクリプションモデル:")
        model_layout.addWidget(tmodel_label, 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_whisper_model_names())
        default_model = base_profile_dict.get("transcription_model", "large-v3")
        if default_model not in [
            self.model_combo.itemText(i) for i in range(self.model_combo.count())
        ]:
            self.model_combo.addItem(default_model)
        self.model_combo.setCurrentText(default_model)
        model_layout.addWidget(self.model_combo, 2, 1)
        model_layout.addWidget(help_label("トランスクリプションモデル"), 2, 2)

        seg_label = QLabel("セグメンテーションモデル:")
        model_layout.addWidget(seg_label, 3, 0)
        self.segmentation_model_combo = QComboBox()
        # トランスクリプションモデルと同一の公式モデル集合を利用
        self.segmentation_model_combo.addItems(get_whisper_model_names())
        default_segmentation_model = base_profile_dict.get(
            "segmentation_model", "turbo"
        )
        if default_segmentation_model not in [
            self.segmentation_model_combo.itemText(i)
            for i in range(self.segmentation_model_combo.count())
        ]:
            self.segmentation_model_combo.addItem(default_segmentation_model)
        self.segmentation_model_combo.setCurrentText(default_segmentation_model)
        model_layout.addWidget(self.segmentation_model_combo, 3, 1)
        model_layout.addWidget(help_label("セグメンテーションモデル"), 3, 2)

        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)

        # 言語設定
        dft = self.config.get(self.current_profile_name, {})  # プロファイル辞書再取得 (kapra 優先)
        lang_group = QGroupBox("言語設定")
        lang_layout = QVBoxLayout()

        def build_language_rows(profile_dict):
            # 既存行をクリアする場合は将来対応（現状は初期化時のみ呼ぶ）
            dlangs_local = profile_dict.get("default_languages", ["ja", "ru"])
            if not isinstance(dlangs_local, list) or not dlangs_local:
                dlangs_local = ["ja"]

            def make_lang_row(code: str, label_text: str, default_weight_key: str):
                row = QWidget()
                hl = QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                chk = QCheckBox(label_text)
                chk.setChecked(code in dlangs_local)
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(300)
                slider.setSingleStep(5)
                slider.setFixedWidth(140)
                val = profile_dict.get(default_weight_key, 1.0)
                if not isinstance(val, (int, float)):
                    val = 1.0
                slider.setValue(int(round(val * 100)))
                value_label = QLabel(f"{val:.2f}")

                def on_change(v):
                    value_label.setText(f"{v/100:.2f}")

                slider.valueChanged.connect(on_change)
                hl.addWidget(chk)
                weight_lbl = QLabel("weight")
                hl.addWidget(weight_lbl)
                hl.addWidget(slider)
                hl.addWidget(value_label)
                hl.addWidget(help_label(default_weight_key))
                return chk, slider, value_label, row

            (
                self.ja_check,
                self.ja_weight_slider_label_dummy,
                self.ja_weight_value_label_dummy,
                ja_row,
            ) = make_lang_row("ja", "JA", "ja_weight")
            self.ja_weight_slider = ja_row.findChildren(QSlider)[0]
            self.ja_weight_value_label = ja_row.findChildren(QLabel)[-1]
            lang_layout.addWidget(ja_row)

            self.ru_check, _, _, ru_row = make_lang_row("ru", "RU", "ru_weight")
            self.ru_weight_slider = ru_row.findChildren(QSlider)[0]
            self.ru_weight_value_label = ru_row.findChildren(QLabel)[-1]
            lang_layout.addWidget(ru_row)

        # 初期プロファイルで構築
        build_language_rows(dft)
        lang_group.setLayout(lang_layout)
        scroll_layout.addWidget(lang_group)

        # プロファイル適用ロジック
        def apply_profile(name: str):
            if name not in self.config:
                return
            prof = self.config.get(name, {})
            self.current_profile_name = name
            # デバイス
            dev = prof.get("device")
            if dev and dev in [self.device_combo.itemText(i) for i in range(self.device_combo.count())]:
                self.device_combo.setCurrentText(dev)
            # モデル
            tmodel = prof.get("transcription_model")
            if tmodel and tmodel in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                self.model_combo.setCurrentText(tmodel)
            # セグメンテーションモデル
            smodel = prof.get("segmentation_model")
            if smodel and smodel in [self.segmentation_model_combo.itemText(i) for i in range(self.segmentation_model_combo.count())]:
                self.segmentation_model_combo.setCurrentText(smodel)
            # 言語チェック & weight
            dlangs_prof = prof.get("default_languages", [])
            if self.ja_check:
                self.ja_check.setChecked("ja" in dlangs_prof or (not dlangs_prof and True))
            if self.ru_check:
                self.ru_check.setChecked("ru" in dlangs_prof)
            if self.ja_weight_slider:
                ja_w = prof.get("ja_weight", 1.0)
                if isinstance(ja_w, (int, float)):
                    self.ja_weight_slider.setValue(int(round(ja_w * 100)))
            if self.ru_weight_slider:
                ru_w = prof.get("ru_weight", 1.0)
                if isinstance(ru_w, (int, float)):
                    self.ru_weight_slider.setValue(int(round(ru_w * 100)))
            # 詳細設定ウィジェット (初期化後に detail_controls が埋まる)
            for key, ctrl in getattr(self, 'detail_controls', {}).items():
                if key not in prof:
                    continue
                val = prof[key]
                from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QComboBox
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
                    # 候補に無ければ追加
                    if ctrl.findText(val) < 0:
                        ctrl.addItem(val)
                    ctrl.setCurrentText(val)

        # プロファイル変更シグナル
        self.profile_combo.currentTextChanged.connect(apply_profile)

        # 詳細設定 (config.toml の default セクションから、基本設定で使ったキーを除外して動的生成)
        detail_group = QGroupBox("詳細設定")
        detail_layout = QVBoxLayout()

        self.detail_controls = {}
        exclude_keys = {
            "device",
            "transcription_model",
            "segmentation_model",
            "default_languages",
            "ja_weight",
            "ru_weight",
            # GUI から除去: 旧オプション (backend で固定挙動化済)
            "dual_transcribe_all",
            "merge_refine",
            "enable_temp_fallback",
        }

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
                if key in {"ja_weight", "ru_weight"}:
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

        # 文字起こし実行ボタン / プログレス / ステータス / ログ
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
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
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
        self.toggle_log_button.toggled.connect(self.toggle_log_panel)
        toggle_row.addWidget(self.toggle_log_button)
        toggle_row.addStretch()
        log_layout.addLayout(toggle_row)
        from PySide6.QtWidgets import QTextEdit as _QTextEdit

        self.log_text = _QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)
        self.log_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size:11px; }"
        )
        log_layout.addWidget(self.log_text)
        transcribe_layout.addWidget(self.log_panel_container)

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
        # 自動同期トグル
        if hasattr(self, 'auto_sync_check'):
            self.auto_sync_check.toggled.connect(self.on_auto_sync_toggled)

    def _collect_selected_rows(self):
        rows = set()
        for r in self.transcription_table.selectedRanges():
            for i in range(r.topRow(), r.bottomRow()+1):
                rows.add(i)
        return sorted(rows)

    def _rebuild_text_and_refresh(self):
        """集約テキスト再構築とテーブル再描画を一括実行。

        以前存在していたメソッドが消えていたため再追加。複数行結合後の
        即時縮約反映や分割後の再描画で使用する。
        """
        try:
            from ui.table_presenter import rebuild_aggregate_text, populate_table
            rebuild_aggregate_text(self.transcription_result)
            populate_table(self.transcription_table, self.transcription_result)
        except Exception:
            pass

    def _can_retranscribe_selection(self, rows):
        if not self.transcription_result:
            return False
        segs = self.transcription_result.get('segments', [])
        rows_sorted = sorted(rows)
        if not rows_sorted:
            return False
    # 対象取得
        try:
            target = [segs[i] for i in rows_sorted]
        except Exception:
            return False
        # 単一行: 無条件で許可（時間長すぎチェック不要）
        if len(rows_sorted) == 1:
            return True
        # 複数行: 連続性 & 30秒以内
        for a,b in zip(rows_sorted, rows_sorted[1:]):
            if b != a+1:
                return False
        start_sec = float(target[0].get('start',0.0))
        end_sec = float(target[-1].get('end', start_sec))
        return (end_sec - start_sec) <= 30.0

    def show_table_context_menu(self, pos):
        if not getattr(self, 'transcription_result', None):
            return
        from PySide6.QtWidgets import QMenu
        global_pos = self.transcription_table.viewport().mapToGlobal(pos)
        # 右クリックしたら再生を一時停止（編集中の誤進行防止）
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                self.media_player.pause()
        except Exception:
            pass
        # 進行中ならメニューは表示するが操作不可
        if getattr(self, 'range_retranscribing', False):
            # 実行中は全アクション無効化したメニューのみ表示（順序は要求仕様に合わせる）
            menu = QMenu(self)
            act_play_from_here = menu.addAction("ここから再生")
            act_edit = menu.addAction("このテキストを編集")
            # 並びを逆転: 結合→再分割
            act_re = menu.addAction("これらの結合＆再文字起こし")
            act_dynamic = menu.addAction("現在位置で分割＆再文字起こし")
            act_delete = menu.addAction("選択行を削除")
            for a in [act_play_from_here, act_edit, act_dynamic, act_re, act_delete]:
                a.setEnabled(False)
            menu.exec(global_pos)
            return
        # 右クリックした位置の行を追加選択（未選択ならその行を単一選択）
        item = self.transcription_table.itemAt(pos)
        if item:
            row = item.row()
            if row not in [i.row() for i in self.transcription_table.selectedItems()]:
                self.transcription_table.selectRow(row)
        rows = self._collect_selected_rows()
        menu = QMenu(self)
        # メニュー順序要求(更新): ここから再生 / このテキストを編集 / これらの結合＆再文字起こし / 現在位置で分割＆再文字起こし / 選択行を削除
        act_play_from_here = menu.addAction("ここから再生")
        act_edit = menu.addAction("このテキストを編集")
        act_re = menu.addAction("これらの結合＆再文字起こし")
        act_dynamic = menu.addAction("現在位置で分割＆再文字起こし")
        act_delete = menu.addAction("選択行を削除")
        # 追加: 現在位置を START/END にセット
        act_set_start = menu.addAction("現在位置をSTARTにセット")
        act_set_end = menu.addAction("現在位置をENDにセット")

        # 可否判定
        segs = self.transcription_result.get('segments', [])
        # 編集可: 単一行
        can_edit = False
        if len(rows) == 1:
            r = rows[0]
            if 0 <= r < len(segs):
                can_edit = True
        if not can_edit:
            act_edit.setEnabled(False)
        # 動的分割/境界調整可: (1行) or (2行連続)
        can_dynamic = False
        if len(rows) == 1 and can_edit:
            can_dynamic = True
        elif len(rows) == 2:
            r1, r2 = rows
            if abs(r1 - r2) == 1:
                if all(0 <= r < len(segs) for r in (r1, r2)):
                    can_dynamic = True
        if not can_dynamic:
            act_dynamic.setEnabled(False)
        # 再生は単一／複数問わず、最初の行の start にシークできる場合のみ有効
        can_play = False
        if rows:
            r0 = rows[0]
            if 0 <= r0 < len(segs):
                try:
                    float(segs[r0].get('start', 0.0))
                    can_play = True
                except Exception:
                    pass
        if not can_play:
            act_play_from_here.setEnabled(False)
        # 再文字起こし（複数連続 or 単一行）
        if not self._can_retranscribe_selection(rows):
            act_re.setEnabled(False)
        # 削除は何か選択されていれば有効
        if not rows:
            act_delete.setEnabled(False)
        # START/END セット可否: 単一行選択 & 再文字起こし処理中でない
        can_set_bounds = (len(rows) == 1 and not getattr(self, 'range_retranscribing', False))
        if not can_set_bounds:
            act_set_start.setEnabled(False)
            act_set_end.setEnabled(False)
        else:
            # 位置取得できなければ無効化
            try:
                cur_sec = self.media_player.position()/1000.0
                r0 = rows[0]
                segs = self.transcription_result.get('segments', [])
                if not (0 <= r0 < len(segs)):
                    raise ValueError
                seg = segs[r0]
                s = float(seg.get('start',0.0)); e = float(seg.get('end', s))
                # START 更新可能条件: cur < e (僅差許容) / END 更新可能条件: cur > s
                if not (cur_sec < e - 0.01):
                    act_set_start.setEnabled(False)
                if not (cur_sec > s + 0.01):
                    act_set_end.setEnabled(False)
            except Exception:
                act_set_start.setEnabled(False)
                act_set_end.setEnabled(False)
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen == act_play_from_here and act_play_from_here.isEnabled():
            # 最初の選択行の start へシークして再生
            if rows:
                segs = self.transcription_result.get('segments', [])
                r0 = rows[0]
                if 0 <= r0 < len(segs):
                    try:
                        start_sec = float(segs[r0].get('start', 0.0))
                        self.media_player.setPosition(int(start_sec * 1000))
                        self.media_player.play()
                    except Exception:
                        pass
        elif 'act_edit' in locals() and chosen == act_edit and act_edit.isEnabled():
            self.invoke_edit_dialog()
        elif chosen == act_dynamic and act_dynamic.isEnabled():
            self.split_or_adjust_at_current_position()
        elif chosen == act_re and act_re.isEnabled():
            self.retranscribe_selected()
        elif chosen == act_delete and act_delete.isEnabled():
            self.delete_selected_segments()
        elif chosen == act_set_start and act_set_start.isEnabled():
            # 単一行 start を現在位置に変更
            rows2 = rows
            if len(rows2) == 1:
                r0 = rows2[0]
                segs2 = self.transcription_result.get('segments', [])
                if 0 <= r0 < len(segs2):
                    try:
                        cur_sec = self.media_player.position()/1000.0
                        seg = segs2[r0]
                        old_end = float(seg.get('end', cur_sec))
                        if cur_sec < old_end - 0.01:  # 最低幅確保
                            seg['start'] = cur_sec
                            if cur_sec > old_end:
                                seg['end'] = cur_sec + 0.01
                            self._rebuild_text_and_refresh()
                            self.status_label.setText(f"行 {r0} の START を {cur_sec:.3f}s に設定")
                    except Exception:
                        pass
        elif chosen == act_set_end and act_set_end.isEnabled():
            rows2 = rows
            if len(rows2) == 1:
                r0 = rows2[0]
                segs2 = self.transcription_result.get('segments', [])
                if 0 <= r0 < len(segs2):
                    try:
                        cur_sec = self.media_player.position()/1000.0
                        seg = segs2[r0]
                        old_start = float(seg.get('start', cur_sec))
                        if cur_sec > old_start + 0.01:
                            seg['end'] = cur_sec
                            if cur_sec < old_start:
                                seg['start'] = cur_sec - 0.01
                            self._rebuild_text_and_refresh()
                            self.status_label.setText(f"行 {r0} の END を {cur_sec:.3f}s に設定")
                    except Exception:
                        pass

    def on_auto_sync_toggled(self, checked: bool):
        """自動同期トグル: ON かつ再生中なら即座に現在位置で同期する。"""
        if not checked:
            return
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                pos = self.media_player.position()
                sel_model = self.transcription_table.selectionModel()
                if sel_model and len(sel_model.selectedRows()) <= 1:
                    self.sync_table_selection_with_position(pos)
        except Exception:
            pass

    def invoke_edit_dialog(self):
        """現在の単一選択行に対して編集ダイアログを開く。"""
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
        # 行存在チェックのみ
        self.open_edit_dialog_for_row(r, 0)

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
            # シーク補助ボタン有効化
            for attr in ('seek_back_10_btn','seek_back_1_btn','seek_fwd_1_btn','seek_fwd_10_btn'):
                if hasattr(self, attr):
                    getattr(self, attr).setEnabled(True)
            self.setWindowTitle(f"動画文字起こしエディタ - {file_path}")

    def ensure_language_selected(self):
        # 状態変更イベントの連続発火で一時的に全チェックが外れることがあるため
        # イベントループの末尾に保護処理を遅延実行して最終状態を確認する
        QTimer.singleShot(0, self._ensure_language_selected_late)

    def _ensure_language_selected_late(self):
        # 言語が一つも選択されていない場合、JAを自動選択
        if not (self.ja_check.isChecked() or self.ru_check.isChecked()):
            self.ja_check.setChecked(True)

    def start_transcription(self):
        # 言語選択 (seg_mode 判定用)
        selected_langs = []
        if self.ja_check.isChecked():
            selected_langs.append("ja")
        if self.ru_check.isChecked():
            selected_langs.append("ru")
            selected_langs.append("en")

        # 詳細設定ウィジェット値を収集 (言語 weight はここでは扱わない)
        detail_values = {}
        for key, ctrl in self.detail_controls.items():
            from PySide6.QtWidgets import (
                QCheckBox,
                QSpinBox,
                QDoubleSpinBox,
                QLineEdit,
                QTextEdit,
                QComboBox,
            )

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
                # vad_level は整数値が必要
                if key == "vad_level":
                    data = ctrl.currentData()
                    if data is not None:
                        detail_values[key] = int(data)
                    else:
                        txtv = ctrl.currentText()
                        try:
                            detail_values[key] = int(txtv)
                        except ValueError:
                            detail_values[key] = 2  # フォールバック
                else:
                    detail_values[key] = ctrl.currentText()

        # オプション構築（advanced_process_video専用）
        options = {
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
        # アクセント補正 (config の値を使う: GUI 未露出)
        cfg_default = self.config.get("default", {})
        # ru_accent_boost 廃止: 旧設定があっても無視
        # 互換用 language (未使用だが将来拡張向け)
        if selected_langs:
            options["language"] = selected_langs[0]

        # UIを無効化
        self.transcribe_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        if hasattr(self, 'retranscribe_button'):
            self.retranscribe_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("文字起こし開始準備中...")
        # 逐次追加用に結果初期化
        self.transcription_result = {'text': '', 'segments': []}

        # 文字起こしスレッドを開始
        self.transcription_thread = TranscriptionThread(
            self.current_video_path, options
        )
        self.transcription_thread.progress.connect(self.update_progress)
        self.transcription_thread.status.connect(self.update_status)
        # 逐次セグメント受信
        try:
            self.transcription_thread.segment_ready.connect(self.on_segment_ready)
        except Exception:
            pass
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
        # ログ履歴に追加
        if hasattr(self, "log_text"):
            self.log_text.append(message)
            if self.log_text.isVisible():
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )

    @Slot(dict)
    def on_transcription_finished(self, result):
        # 最終同期（逐次で蓄積済みでも最終結果を信頼して再描画）
        self.transcription_result = result
        self.display_transcription(result)
        self.transcribe_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(True)
        if hasattr(self, 'retranscribe_button'):
            self.retranscribe_button.setEnabled(True)
        if hasattr(self, 'partial_export_button'):
            self.partial_export_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("文字起こし完了")

    @Slot(dict)
    def on_segment_ready(self, seg_dict: dict):
        # 逐次追加: transcription_result が初期化済み前提
        try:
            # レースガード: スレッドが既に差し替えられていないか確認
            thr = getattr(self, 'transcription_thread', None)
            if thr is None or not thr.isRunning():
                return
            if getattr(self, '_cancelled_during_transcribe', False):
                return
            segs = self.transcription_result.get('segments', []) if self.transcription_result else []
            segs.append(seg_dict)
            line = display_text(seg_dict)
            if line:
                current_text = self.transcription_result.get('text','') if self.transcription_result else ''
                self.transcription_result['text'] = (current_text + ('\n' if current_text else '') + line)
            row = self.transcription_table.rowCount()
            self.transcription_table.insertRow(row)
            def fmt_ms(ms:int)->str:
                s = ms/1000.0
                h = int(s//3600); m = int((s%3600)//60); sec = s%60
                return f"{h:02d}:{m:02d}:{sec:06.3f}"[:-1]  # 末尾 1 桁切って mm
            st_ms = int(seg_dict.get('start',0.0)*1000)
            ed_ms = int(seg_dict.get('end',0.0)*1000)
            self.transcription_table.setItem(row, 0, QTableWidgetItem(fmt_ms(st_ms)))
            self.transcription_table.setItem(row, 1, QTableWidgetItem(fmt_ms(ed_ms)))
            self.transcription_table.setItem(row, 2, QTableWidgetItem(f"{seg_dict.get('ja_prob',0.0):.2f}"))
            self.transcription_table.setItem(row, 3, QTableWidgetItem(f"{seg_dict.get('ru_prob',0.0):.2f}"))
            self.transcription_table.setItem(row, 4, QTableWidgetItem(seg_dict.get('text','')))
            self.status_label.setText(f"文字起こし中... ({len(segs)})")
        except Exception:
            pass

    @Slot(str)
    def on_transcription_error(self, error_message):
        logger = logging.getLogger(__name__)
        logger.error(f"Transcription error: {error_message}")
        self.status_label.setText(f"エラー: {error_message}")
        self.transcribe_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(bool(getattr(self, 'transcription_result', None)))
        if hasattr(self, 'retranscribe_button'):
            self.retranscribe_button.setEnabled(bool(getattr(self, 'transcription_result', None)))
        if hasattr(self, 'partial_export_button'):
            self.partial_export_button.setEnabled(bool(getattr(self, 'transcription_result', None)))
        self.progress_bar.setValue(0)

    def cancel_transcription(self):
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.status_label.setText("キャンセル要求送信中...")
            try:
                self.transcription_thread.request_cancel()
                self.cancel_button.setEnabled(False)
            except Exception as e:
                logging.getLogger(__name__).error(f"Cancel failed: {e}")
            # 逐次表示を即停止し画面をクリア
            try:
                self.transcription_table.setRowCount(0)
            except Exception:
                pass
            self.transcription_result = {'text': '', 'segments': []}
            self._cancelled_during_transcribe = True

    def toggle_log_panel(self, checked: bool):
        if checked:
            self.log_text.setVisible(True)
            self.toggle_log_button.setText("▲ ログ非表示")
        else:
            self.log_text.setVisible(False)
            self.toggle_log_button.setText("▼ ログ表示")

    def display_transcription(self, result):
        if not result:
            return
        populate_table(self.transcription_table, result)

    def retranscribe_selected(self):
        """選択行を再文字起こし。
    - 複数行(連続, 30秒以内) → 結合して1行置換
    - 単一行 → その区間のみ再文字起こし
        """
        from PySide6.QtWidgets import QMessageBox, QApplication
        # ボタン押下で一時停止
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlayingState:
                self.media_player.pause()
        except Exception:
            pass
        if not self.transcription_result:
            return
        ranges = self.transcription_table.selectedRanges()
        if not ranges:
            QMessageBox.warning(self, "再文字起こし", "行を選択してください")
            return
        rows = set()
        for r in ranges:
            for i in range(r.topRow(), r.bottomRow()+1):
                rows.add(i)
        rows_sorted = sorted(rows)
        if not rows_sorted:
            return
        segs = as_segment_list(self.transcription_result.get('segments', []))
        try:
            target = [segs[i] for i in rows_sorted]
        except IndexError:
            QMessageBox.critical(self, "エラー", "内部インデックス不整合")
            return
        # GAP 概念廃止: そのまま処理
        # 単一行モード
        if len(rows_sorted) == 1:
            idx = rows_sorted[0]
            seg = target[0]
            start_sec = float(seg.get('start',0.0))
            end_sec = float(seg.get('end', start_sec))
            from transcriber import RangeTranscriptionThread
            model_size = self.model_combo.currentText()
            device = self.device_combo.currentText()
            ja_w = self.ja_weight_slider.value()/100.0
            ru_w = self.ru_weight_slider.value()/100.0
            options = {
                'model': model_size,
                'device': device,
                'ja_weight': ja_w,
                'ru_weight': ru_w,
            }
            self.status_label.setText(f"単一行再文字起こし中 (行 {idx})…")
            self.progress_bar.setValue(0)
            self.range_retranscribing = True
            self.range_thread = RangeTranscriptionThread(self.current_video_path, start_sec, end_sec, options)
            # 既存の範囲用シグナルを流用
            self.range_thread.progress.connect(self.on_range_progress)
            self.range_thread.status.connect(self.on_range_status)
            # 完了時にその行へ反映するラッパー
            def single_finished(seg_result: dict):
                try:
                    if 0 <= idx < len(segs):
                        old = segs[idx]
                        old.update({
                            'text': seg_result.get('text',''),
                            'text_ja': seg_result.get('text_ja',''),
                            'text_ru': seg_result.get('text_ru',''),
                            'ja_prob': seg_result.get('ja_prob',0.0),
                            'ru_prob': seg_result.get('ru_prob',0.0),
                            'chosen_language': seg_result.get('chosen_language', old.get('chosen_language')),
                        })
                finally:
                    self.range_retranscribing = False
                    self._rebuild_text_and_refresh()
                    self.status_label.setText("単一行再文字起こし完了")
            def single_error(err: str):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "再文字起こし失敗", err)
                self.range_retranscribing = False
                self._rebuild_text_and_refresh()
                self.status_label.setText("単一行再文字起こし失敗")
            self.range_thread.range_finished.connect(single_finished)
            self.range_thread.error.connect(single_error)
            self.range_thread.start()
            return
        # 複数行（従来処理）: 連続性 & 30秒以内チェック
        merged_list, insert_index, sec_range = merge_contiguous_segments(
            self.transcription_result.get('segments', []), rows_sorted
        )
        if merged_list is None:
            QMessageBox.critical(self, "エラー", "連続でない行や不正な範囲が含まれています")
            return
        start_sec, end_sec = sec_range
        if (end_sec - start_sec) > 30.0:
            QMessageBox.critical(self, "エラー", "選択範囲が30秒を超えています")
            return
        # セグメント差し替え（プレースホルダ）: 選択複数行を即時 1 行縮約
        self.transcription_result['segments'] = merged_list
        # 結合後の統合行へ [再解析中] プレースホルダ設定
        try:
            if 0 <= insert_index < len(self.transcription_result['segments']):
                ph = self.transcription_result['segments'][insert_index]
                ph['text'] = ph['text_ja'] = ph['text_ru'] = '[再解析中]'
                ph['ja_prob'] = 0.0
                ph['ru_prob'] = 0.0
                ph['chosen_language'] = None
        except Exception:
            pass
        # 行数変化を即時 UI へ反映
        try:
            self._rebuild_text_and_refresh()
        except Exception:
            pass
        # 非同期スレッドで再文字起こし
        from transcriber import RangeTranscriptionThread
        model_size = self.model_combo.currentText()
        device = self.device_combo.currentText()
        ja_w = self.ja_weight_slider.value()/100.0
        ru_w = self.ru_weight_slider.value()/100.0
        options = {
            'model': model_size,
            'device': device,
            'ja_weight': ja_w,
            'ru_weight': ru_w,
        }
        self.status_label.setText("再文字起こし準備中...")
        self.progress_bar.setValue(0)
        if hasattr(self, 'retranscribe_button'):
            self.retranscribe_button.setEnabled(False)
        if hasattr(self, 'partial_export_button'):
            self.partial_export_button.setEnabled(False)
        # フラグON (メニュー等を無効化)
        self.range_retranscribing = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.range_thread = RangeTranscriptionThread(self.current_video_path, start_sec, end_sec, options)
        self.range_thread.progress.connect(self.on_range_progress)
        self.range_thread.status.connect(self.on_range_status)
        # 完了時: placeholder 行 (merge 後の insert_index) を結果で置換。rows_sorted は元範囲参照用。
        cb = lambda seg, rows_sorted=rows_sorted, orig_segs=segs: self.on_range_finished(seg, rows_sorted, orig_segs)
        self.range_thread.range_finished.connect(cb)
        self.range_thread.error.connect(self.on_range_error)
        self.range_thread.start()
        # 後続処理はコールバックで行う
        return
    def on_range_progress(self, value: int):
        self.progress_bar.setValue(value)

    def on_range_status(self, message: str):
        self.status_label.setText(message)
        if hasattr(self, 'log_text'):
            self.log_text.append(message)
            if self.log_text.isVisible():
                self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_range_finished(self, new_seg: dict, rows_sorted: list[int], segs: list[dict]):
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        # フラグ解除
        self.range_retranscribing = False
        new_segments = []
        first = rows_sorted[0]
        for idx, s in enumerate(segs):
            if idx == first:
                new_segments.append({
                    'start': new_seg['start'], 'end': new_seg['end'],
                    'text': new_seg['text'], 'text_ja': new_seg['text_ja'], 'text_ru': new_seg['text_ru'],
                    'chosen_language': new_seg['chosen_language'], 'id': s.get('id', idx),
                    'ja_prob': new_seg['ja_prob'], 'ru_prob': new_seg['ru_prob']
                })
            elif idx in rows_sorted[1:]:
                continue
            else:
                new_segments.append(s)
        self.transcription_result['segments'] = new_segments
        try:
            lines = []
            for s in new_segments:
                lines.append(display_text(s))
            self.transcription_result['text'] = '\n'.join(lines)
        except Exception:
            pass
        self.display_transcription(self.transcription_result)
        self.status_label.setText("再文字起こし完了")
        if hasattr(self, 'retranscribe_button'):
            self.retranscribe_button.setEnabled(True)
        if hasattr(self, 'partial_export_button'):
            self.partial_export_button.setEnabled(True)

    def on_range_error(self, err: str):
        from PySide6.QtWidgets import QApplication, QMessageBox
        QApplication.restoreOverrideCursor()
        # フラグ解除
        self.range_retranscribing = False
        self.status_label.setText("再文字起こし失敗")
        QMessageBox.critical(self, "再文字起こし失敗", err)
        if hasattr(self, 'retranscribe_button'):
            self.retranscribe_button.setEnabled(True)
        if hasattr(self, 'partial_export_button'):
            self.partial_export_button.setEnabled(True)

    def partial_export_selected(self):
        """選択された行範囲の音声をWAV、テキストをTXTとして ./output に保存。ファイル名: 動画ベース_START_END"""
        from PySide6.QtWidgets import QMessageBox
        if not self.transcription_result or not self.current_video_path:
            QMessageBox.warning(self, "部分書き出し", "書き出し対象がありません")
            return
        ranges = self.transcription_table.selectedRanges()
        if not ranges:
            QMessageBox.warning(self, "部分書き出し", "行を選択してください")
            return
        rows = set()
        for r in ranges:
            for i in range(r.topRow(), r.bottomRow()+1):
                rows.add(i)
        rows = sorted(rows)
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
        end_sec = float(target[-1].get('end', start_sec))
        if end_sec <= start_sec:
            QMessageBox.critical(self, "部分書き出し", "時間範囲が不正です")
            return
        from utils.segment_utils import display_text
        lines = [display_text(s) for s in target]
        text_out = '\n'.join(lines)
        base_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        def ts(sec: float):
            h = int(sec // 3600); m = int((sec % 3600)//60); s = int(sec % 60)
            return f"{h:02d}{m:02d}{s:02d}"
        safe_start = ts(start_sec)
        safe_end = ts(end_sec)
        out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(out_dir, exist_ok=True)
        wav_path = os.path.join(out_dir, f"{base_name}_{safe_start}_{safe_end}.wav")
        txt_path = os.path.join(out_dir, f"{base_name}_{safe_start}_{safe_end}.txt")
        import subprocess, tempfile
        try:
            cmd = [
                'ffmpeg','-y',
                '-i', self.current_video_path,
                '-ss', f"{start_sec:.3f}",
                '-to', f"{end_sec:.3f}",
                '-vn',
                '-acodec','pcm_s16le','-ar','16000','-ac','1',
                wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "部分書き出し", "音声抽出に失敗しました (ffmpeg)")
            return
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_out)
        except Exception as e:
            QMessageBox.critical(self, "部分書き出し", f"テキスト保存失敗: {e}")
            return
        duration = end_sec - start_sec
        pretty_start = f"{start_sec:,.3f}s"
        pretty_end = f"{end_sec:,.3f}s"
        seg_count = len(target)
        # ログ/ダイアログでは絶対パスではなく相対パス表示（作業ディレクトリ基準）にする
        try:
            cwd = os.getcwd()
            rel_wav = os.path.relpath(wav_path, cwd)
            rel_txt = os.path.relpath(txt_path, cwd)
        except Exception:
            rel_wav = wav_path
            rel_txt = txt_path
        msg = (
            "部分書き出しが完了しました\n"
            "--------------------------------\n"
            f" 対象行数 : {seg_count}\n"
            f" 時間範囲 : {pretty_start} 〜 {pretty_end} (Δ {duration:.3f}s)\n"
            f" 出力WAV : {rel_wav}\n"
            f" 出力TXT : {rel_txt}\n"
            "--------------------------------"
        )
        self.status_label.setText("部分書き出し完了")
        if hasattr(self, 'log_text'):
            self.log_text.append(msg)
        from PySide6.QtWidgets import QMessageBox as _QMB
        _QMB.information(self, "部分書き出し", msg)

    def delete_selected_segments(self):
        """選択行のテキスト列のみ空にし再描画。"""
        if getattr(self, 'range_retranscribing', False):
            return
        if not getattr(self, 'transcription_result', None):
            return
        segs = self.transcription_result.get('segments', [])
        if not segs:
            return
        rows = self._collect_selected_rows()
        if not rows:
            return
        changed = 0
        for r in rows:
            if 0 <= r < len(segs):
                seg = segs[r]
                if any(seg.get(k) for k in ('text','text_ja','text_ru')):
                    seg['text'] = ''
                    seg['text_ja'] = ''
                    seg['text_ru'] = ''
                    changed += 1
        if changed == 0:
            return
        try:
            from transcriber import rebuild_aggregate_text
            rebuild_aggregate_text(self.transcription_result)
        except Exception:
            pass
        try:
            for r in rows:
                if 0 <= r < self.transcription_table.rowCount():
                    item = self.transcription_table.item(r, 4)
                    if item is None:
                        from PySide6.QtWidgets import QTableWidgetItem
                        item = QTableWidgetItem('')
                        self.transcription_table.setItem(r, 4, item)
                    else:
                        item.setText('')
        except Exception:
            self.display_transcription(self.transcription_result)
        self.status_label.setText(f"{changed}行のテキストを消去しました")
        if hasattr(self, 'delete_button'):
            self.delete_button.setEnabled(False)
        self._update_split_button_state()

    def export_transcription(self):
        """認識結果をファイルに書き出す (json / txt / srt)。デフォルトは json。"""
        if not getattr(self, 'transcription_result', None):
            self.status_label.setText("書き出し対象がありません")
            return
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        # デフォルトファイル名: 動画ファイル名 (拡張子除去) + .json
        if self.current_video_path:
            base = os.path.splitext(os.path.basename(self.current_video_path))[0]
            default_name = f"{base}.json"
        else:
            default_name = "transcription.json"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "書き出しファイルを保存",
            default_name,
            "JSON (*.json);;テキスト (*.txt);;SRT 字幕 (*.srt)"
        )
        if not file_path:
            return
        # フォーマット判定
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
                import json
                payload = build_json_payload(
                    self.transcription_result,
                    {
                        'video_path': self.current_video_path,
                        'model': self.model_combo.currentText() if hasattr(self, 'model_combo') else None,
                        'device': self.device_combo.currentText() if hasattr(self, 'device_combo') else None,
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
            logging.getLogger(__name__).exception("Export failed")
            QMessageBox.critical(self, "書き出しエラー", str(e))
            self.status_label.setText("書き出し失敗")

    # _build_export_text: 外部 exporter.build_export_text へ移行済

    def play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop_video(self):
        self.media_player.stop()

    def seek_relative(self, delta_ms: int):
        """現在位置から delta_ms だけ相対シーク (境界クリップ)。"""
        try:
            cur = self.media_player.position()
            dur = self.media_player.duration() or 0
            new_pos = cur + delta_ms
            if new_pos < 0:
                new_pos = 0
            if dur > 0 and new_pos > dur:
                new_pos = dur
            self.media_player.setPosition(new_pos)
        except Exception:
            pass

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)
        self.current_time_label.setText(format_ms(position))
        # 自動同期: チェックON かつ 再生中 のみ。ポーズ/停止中は固定。
        if getattr(self, 'auto_sync_check', None) and not self.auto_sync_check.isChecked():
            return
        if self.media_player.playbackState() != QMediaPlayer.PlayingState:
            return
        try:
            sel_model = self.transcription_table.selectionModel()
            if sel_model and len(sel_model.selectedRows()) <= 1:
                self.sync_table_selection_with_position(position)
        except Exception:
            pass

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.total_time_label.setText(format_ms(duration))

    def set_position(self, position):
        self.media_player.setPosition(position)

    # 旧 format_time / parse_time_to_ms は utils.timefmt の format_ms / parse_to_ms に統合
    # 後方互換が必要なら以下のようにエイリアス化も可能
    # format_time = staticmethod(format_ms)
    # parse_time_to_ms = staticmethod(parse_to_ms)


def main():
    # ベーシックロギング設定 (transcriber 側で既にハンドラがあれば二重追加は避けられる)
    logger = logging.getLogger()
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(asctime)s %(name)s: %(message)s", "%H:%M:%S"
            )
        )
        logger.addHandler(h)
    logger.setLevel(logging.INFO)

    app = QApplication(sys.argv)
    window = VideoTranscriptionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
