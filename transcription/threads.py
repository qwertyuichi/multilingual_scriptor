"""文字起こし処理用 Qt スレッド群。

提供クラス:
 - `TranscriptionThread`      : 動画全体の文字起こしを非同期実行
 - `RangeTranscriptionThread` : 部分区間再文字起こしを非同期実行
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from transcription.processor import advanced_process_video, transcribe_range
from logging_config import get_logger

logger = get_logger(__name__)


class TranscriptionThread(QThread):
    """GUI から起動されるバックグラウンド文字起こしスレッド。

    Signals
    -------
    progress(int)              : 進捗率 0-100
    status(str)                : ステータスメッセージ
    segment_ready(dict)        : セグメント逐次通知
    finished_transcription(dict) : 完了結果
    error(str)                 : エラーメッセージ
    """

    progress = Signal(int)
    status = Signal(str)
    segment_ready = Signal(dict)
    finished_transcription = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: str, options: dict):
        super().__init__()
        self.video_path = video_path
        self.options = options
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """GUI からのキャンセル要求。"""
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            self.status.emit("文字起こし開始準備中...")
            self.progress.emit(5)
            result = advanced_process_video(
                self.video_path,
                model_size=self.options.get('model', 'large-v3'),
                segmentation_model_size=self.options.get('segmentation_model_size', 'turbo'),
                seg_mode=self.options.get('seg_mode', 'hybrid'),
                device=self.options.get('device'),
                ja_weight=self.options.get('ja_weight', 0.80),
                ru_weight=self.options.get('ru_weight', 1.25),
                min_seg_dur=self.options.get('min_seg_dur', 0.60),
                ambiguous_threshold=self.options.get('ambiguous_threshold', 10.0),
                vad_level=self.options.get('vad_level', 2),
                gap_threshold=self.options.get('gap_threshold', 0.5),
                output_format=self.options.get('output_format', 'txt'),
                srt_max_line=self.options.get('srt_max_line', 50),
                include_silent=self.options.get('include_silent', False),
                debug=self.options.get('debug_segments', False),
                duplicate_merge=self.options.get('duplicate_merge', True),
                duplicate_debug=self.options.get('duplicate_debug', True),
                silence_rms_threshold=self.options.get('silence_rms_threshold'),
                min_voice_ratio=self.options.get('min_voice_ratio'),
                max_silence_repeat=self.options.get('max_silence_repeat'),
                progress_callback=lambda p: self.progress.emit(p),
                status_callback=lambda m: self.status.emit(m),
                cancel_flag=self.is_cancelled,
                segment_callback=lambda d: self.segment_ready.emit(d) if not self.is_cancelled() else None,
            )
            self.progress.emit(100)
            if self.is_cancelled():
                self.status.emit("キャンセルされました")
            else:
                self.status.emit("文字起こし完了")
            self.finished_transcription.emit(result)
        except Exception as e:
            logger.exception("[ERROR] TranscriptionThread exception:")
            self.error.emit(str(e))
            self.status.emit("エラーが発生しました")


class RangeTranscriptionThread(QThread):
    """部分区間再文字起こし専用スレッド。

    Signals
    -------
    progress(int)       : 進捗率 0-100
    status(str)         : ステータスメッセージ
    range_finished(dict): 再文字起こし完了結果
    error(str)          : エラーメッセージ
    """

    progress = Signal(int)
    status = Signal(str)
    range_finished = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: str, start_sec: float, end_sec: float, options: dict):
        super().__init__()
        self.video_path = video_path
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.options = options

    def run(self) -> None:
        try:
            res = transcribe_range(
                self.video_path,
                self.start_sec,
                self.end_sec,
                model_size=self.options.get('model', 'large-v3'),
                device=self.options.get('device'),
                ja_weight=self.options.get('ja_weight', 1.0),
                ru_weight=self.options.get('ru_weight', 1.0),
                ambiguous_threshold=self.options.get('ambiguous_threshold', 10.0),
                progress_callback=lambda p: self.progress.emit(p),
                status_callback=lambda m: self.status.emit(m),
                silence_rms_threshold=self.options.get('silence_rms_threshold'),
                min_voice_ratio=self.options.get('min_voice_ratio'),
            )
            self.progress.emit(100)
            self.status.emit('部分再文字起こし完了')
            self.range_finished.emit(res)
        except Exception as e:
            logger.exception('[ERROR] RangeTranscriptionThread exception:')
            self.error.emit(str(e))
            self.status.emit('部分再文字起こし失敗')
