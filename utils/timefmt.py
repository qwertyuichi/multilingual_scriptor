"""時間フォーマット/パース共通ユーティリティ。

format_ms: ミリ秒 -> "HH:MM:SS.mmm" (先頭0埋め、時間は必要に応じて2桁固定)
parse_to_ms: 文字列 -> ミリ秒 (HH:MM:SS[.mmm] / MM:SS[.mmm] / SS[.mmm])

今後 SRT 用のフォーマット変換 (to_srt_timestamp) などを追加する想定。
"""
from __future__ import annotations
import re

__all__ = ["format_ms", "parse_to_ms", "to_srt_timestamp"]

_TIME_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$")
# 例: 01:02:03.456 / 02:03.456 / 02:03


def format_ms(ms: int | float) -> str:
    if ms is None:
        return "00:00:00.000"
    total_ms = int(round(ms))
    if total_ms < 0:
        total_ms = 0
    h = total_ms // 3600000
    rem = total_ms % 3600000
    m = rem // 60000
    rem2 = rem % 60000
    s = rem2 // 1000
    milli = rem2 % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def parse_to_ms(text: str) -> int:
    if not text:
        return 0
    text = text.strip()
    # 純粋な秒 or 秒.ミリ秒
    if re.match(r"^\d+(?:\.\d+)?$", text):
        sec = float(text)
        return int(round(sec * 1000))
    m = _TIME_RE.match(text)
    if not m:
        return 0
    h_grp, m_grp, s_grp, ms_grp = m.groups()
    hours = int(h_grp) if h_grp else 0
    minutes = int(m_grp)
    seconds = int(s_grp)
    millis = int(ms_grp.ljust(3, '0')) if ms_grp else 0
    total = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    return total


def to_srt_timestamp(ms: int | float) -> str:
    if ms is None:
        ms = 0
    total_ms = int(round(ms))
    if total_ms < 0:
        total_ms = 0
    h = total_ms // 3600000
    rem = total_ms % 3600000
    m = rem // 60000
    rem2 = rem % 60000
    s = rem2 // 1000
    milli = rem2 % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"
