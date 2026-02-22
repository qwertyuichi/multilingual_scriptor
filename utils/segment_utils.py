"""セグメント関連の共通ユーティリティ。

display_text(segment):
    - GAP 行は「[無音]」を返す (BUG-7 修正: silence と統一)
    - chosen_language 優先
    - chosen_language='other' の場合は Phase1 テキスト (text フィールド) を使用
    - それ以外は lang1_prob >= lang2_prob で fallback

normalize_segment_id(segment, fallback_id):
    - segment['id'] を int 化。失敗したら fallback_id を返す。
"""
from __future__ import annotations
from typing import Any, Dict

__all__ = ["display_text", "normalize_segment_id"]


def display_text(seg: Dict[str, Any]) -> str:
    if not seg:
        return ""
    # GAP と silence は両方とも「[無音]」表示 (BUG-7 修正)
    if seg.get("gap"):
        return "[無音]"
    if seg.get("chosen_language") == "silence":
        return "[無音]"
    # プレースホルダ (再解析中) はそのまま表示
    for key in ('text', 'text_lang1', 'text_lang2'):
        v = seg.get(key)
        if isinstance(v, str) and v.startswith('[再解析中]'):
            return v
    chosen = seg.get("chosen_language")
    lang1_code = seg.get("lang1_code", "ja")
    lang2_code = seg.get("lang2_code", "ru")
    t1 = seg.get("text_lang1", "") or ""
    t2 = seg.get("text_lang2", "") or ""
    p1 = seg.get("lang1_prob", 0.0) or 0.0
    p2 = seg.get("lang2_prob", 0.0) or 0.0
    if chosen and chosen not in {lang1_code, lang2_code, 'other', 'silence'}:
        prefix = f"[{str(chosen).upper()}?]"
        base = seg.get('text', '') or ''
        if base.startswith(prefix):
            return base
        return f"{prefix} {base}".strip()
    # 'other' = 選択外言語確定 → Phase1 テキスト (text フィールド) を保持
    if chosen == 'other':
        return seg.get('text', '') or ''
    if chosen == lang1_code and t1:
        return t1
    if chosen == lang2_code and t2:
        return t2
    # fallback
    if p1 >= p2:
        return t1 or t2
    return t2 or t1


def normalize_segment_id(seg: Dict[str, Any], fallback_id: int) -> int:
    raw_id = seg.get('id', fallback_id)
    if isinstance(raw_id, int):
        return raw_id
    try:
        return int(raw_id)
    except Exception:
        return fallback_id
