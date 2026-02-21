"""transcription パッケージ。

旧 `transcriber.py` を機能ごとに分割したもの:
  - audio.py       : 音声処理ユーティリティ群
  - model_cache.py : Whisper モデルの LRU キャッシュ
  - processor.py   : advanced_process_video / transcribe_range
  - threads.py     : TranscriptionThread / RangeTranscriptionThread

このファイルは各モジュールの公開シンボルを re-export し、
`from transcription import TranscriptionThread` のようなシンプルな import を可能にする。
"""
from transcription.audio import clean_hallucination
from transcription.model_cache import _load_cached_model
from transcription.processor import advanced_process_video, transcribe_range
from transcription.threads import TranscriptionThread, RangeTranscriptionThread

__all__ = [
    "clean_hallucination",
    "_load_cached_model",
    "advanced_process_video",
    "transcribe_range",
    "TranscriptionThread",
    "RangeTranscriptionThread",
]
