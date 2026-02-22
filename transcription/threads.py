"""文字起こし処理用 Qt スレッド群。

提供クラス:
 - `TranscriptionThread`      : 動画全体の文字起こしを非同期実行
 - `RangeTranscriptionThread` : 部分区間再文字起こしを非同期実行
"""
from __future__ import annotations

try:
    import ctranslate2
    _CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0
except Exception:
    _CUDA_AVAILABLE = False

from PySide6.QtCore import QThread, Signal

from transcription.processor import advanced_process_video, transcribe_range
from core.logging_config import get_logger

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
    device_fallback_warning = Signal()

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
            # デバイスフォールバックの事前チェック（文字起こし開始前に警告）
            requested_device = self.options.get('device', 'cuda')
            if requested_device == 'cuda' and not _CUDA_AVAILABLE:
                logger.warning('[THREAD] CUDAが要求されましたがCPUにフォールバックします')
                self.device_fallback_warning.emit()
            
            self.status.emit("文字起こし開始準備中...")
            self.progress.emit(5)
            result = advanced_process_video(
                self.video_path,
                model_size=self.options.get('model', 'large-v3'),
                device=self.options.get('device'),
                languages=self.options.get('languages', ['ja']),
                ja_weight=self.options.get('ja_weight', 1.0),
                ru_weight=self.options.get('ru_weight', 1.0),
                beam_size=self.options.get('beam_size', 5),
                no_speech_threshold=self.options.get('no_speech_threshold', 0.6),
                initial_prompt=self.options.get('initial_prompt') or None,
                vad_filter=self.options.get('vad_filter', True),
                vad_threshold=self.options.get('vad_threshold', 0.5),
                vad_min_speech_ms=self.options.get('vad_min_speech_ms', 250),
                vad_min_silence_ms=self.options.get('vad_min_silence_ms', 2000),
                ambiguous_threshold=self.options.get('ambiguous_threshold', 30.0),
                condition_on_previous_text=self.options.get('condition_on_previous_text', True),
                compression_ratio_threshold=self.options.get('compression_ratio_threshold', 2.4),
                log_prob_threshold=self.options.get('log_prob_threshold', -1.0),
                repetition_penalty=self.options.get('repetition_penalty', 1.0),
                speech_pad_ms=self.options.get('speech_pad_ms', 400),
                duplicate_merge=self.options.get('duplicate_merge', True),
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
    device_fallback_warning = Signal()

    def __init__(self, video_path: str, start_sec: float, end_sec: float, options: dict):
        super().__init__()
        self.video_path = video_path
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.options = options

    def run(self) -> None:
        try:
            # デバイスフォールバックの事前チェック（再文字起こし開始前に警告）
            requested_device = self.options.get('device', 'cuda')
            if requested_device == 'cuda' and not _CUDA_AVAILABLE:
                logger.warning('[THREAD] CUDAが要求されましたがCPUにフォールバックします')
                self.device_fallback_warning.emit()
            
            res = transcribe_range(
                self.video_path,
                self.start_sec,
                self.end_sec,
                model_size=self.options.get('model', 'large-v3'),
                device=self.options.get('device'),
                languages=self.options.get('languages', ['ja']),
                ja_weight=self.options.get('ja_weight', 1.0),
                ru_weight=self.options.get('ru_weight', 1.0),
                beam_size=self.options.get('beam_size', 5),
                no_speech_threshold=self.options.get('no_speech_threshold', 0.6),
                condition_on_previous_text=self.options.get('condition_on_previous_text', False),
                progress_callback=lambda p: self.progress.emit(p),
                status_callback=lambda m: self.status.emit(m),
            )
            self.progress.emit(100)
            self.status.emit('部分再文字起こし完了')
            
            self.range_finished.emit(res)
        except Exception as e:
            logger.exception('[ERROR] RangeTranscriptionThread exception:')
            self.error.emit(str(e))
            self.status.emit('部分再文字起こし失敗')
