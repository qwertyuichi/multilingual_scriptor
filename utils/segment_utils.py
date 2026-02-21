"""セグメント関連の共通ユーティリティ。

display_text(segment):
    - GAP 行は空文字を返す (特別表示しない簡潔仕様)
    - chosen_language 優先
    - それ以外は ja_prob >= ru_prob で fallback

normalize_segment_id(segment, fallback_id):
    - segment['id'] を int 化。失敗したら fallback_id を返す。
"""
from __future__ import annotations
from typing import Any, Dict

__all__ = ["display_text", "normalize_segment_id"]


def display_text(seg: Dict[str, Any]) -> str:
    if not seg:
        return ""
    if seg.get("gap"):
        return ""  # GAP は空文字表示
    # 無音セグメントは「[無音]」と表示
    if seg.get("chosen_language") == "silence":
        return "[無音]"
    # プレースホルダ (再解析中) はそのまま表示
    for key in ('text', 'text_ja', 'text_ru'):
        v = seg.get(key)
        if isinstance(v, str) and v.startswith('[再解析中]'):
            return v
    chosen = seg.get("chosen_language")
    ja = seg.get("text_ja", "") or ""
    ru = seg.get("text_ru", "") or ""
    jp = seg.get("ja_prob", 0.0) or 0.0
    rp = seg.get("ru_prob", 0.0) or 0.0
    if chosen == 'ja' and ja:
        return ja
    if chosen == 'ru' and ru:
        return ru
    # fallback
    if jp >= rp:
        return ja or ru
    return ru or ja


def normalize_segment_id(seg: Dict[str, Any], fallback_id: int) -> int:
    raw_id = seg.get('id', fallback_id)
    if isinstance(raw_id, int):
        return raw_id
    try:
        return int(raw_id)
    except Exception:
        return fallback_id
