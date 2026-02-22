"""再文字起こし / 分割支援の上位レベルサービス。

提供関数:
 - `dynamic_time_split(segments, row_index, current_time_sec)` -> (新しいセグメントリスト, 前側インデックス) もしくは (None, None)
     一つの非 GAP セグメント内部で現在再生時刻が範囲内にある場合、時間比から文字位置を推定して 2 分割。
 - `merge_contiguous_segments(segments, indices)` -> (新しいリスト, 挿入位置, (開始秒, 終了秒)) もしくは (None, None, None)
     連続した複数セグメントを 1 つのプレースホルダへ統合し、テキストを空にして再文字起こし用にする。

前提:
 - すべてのセグメント表現は `list[dict]`。
 - Qt 依存なし (純粋ロジック)。
"""
from __future__ import annotations
from typing import List, Tuple, Optional
from models.segment import as_segment_list, Segment
from services.segment_ops import split_segment_at_position


def dynamic_time_split(segments: List[dict], row_index: int, current_time_sec: float) -> Tuple[Optional[List[dict]], Optional[int]]:
    segs = as_segment_list(segments)
    if row_index < 0 or row_index >= len(segs):
        return None, None
    seg = segs[row_index]
    start = float(seg.get('start', 0.0)); end = float(seg.get('end', start))
    if not (start < current_time_sec < end):
        return None, None
    # どのテキストを基準に長さ計測するか: chosen_language 優先、なければ確率優勢言語
    text_lang1 = seg.get('text_lang1', '')
    text_lang2 = seg.get('text_lang2', '')
    lang1_prob = seg.get('lang1_prob', 0.0); lang2_prob = seg.get('lang2_prob', 0.0)
    lang1_code = seg.get('lang1_code', 'ja')
    lang2_code = seg.get('lang2_code', 'ru')
    chosen = seg.get('chosen_language')
    if chosen == lang1_code and text_lang1:
        base_text = text_lang1; which = 'lang1'
    elif chosen == lang2_code and text_lang2:
        base_text = text_lang2; which = 'lang2'
    else:
        if lang1_prob >= lang2_prob:
            base_text = text_lang1; which = 'lang1'
        else:
            base_text = text_lang2; which = 'lang2'
    # Fallback: if specific language fields are empty but a generic 'text' exists,
    # use it as the basis for position calculation. This covers cases where a
    # forced re-recognition updated 'text' but not 'text_lang1'/'text_lang2'.
    if not base_text:
        fallback_text = seg.get('text', '')
        if fallback_text:
            base_text = fallback_text
            which = 'text'
        else:
            return None, None
    # 文字位置計算
    ratio = (current_time_sec - start) / max(1e-9, (end - start))
    pos = int(len(base_text) * ratio)
    if pos <= 0 or pos >= len(base_text):
        return None, None
    new_list = split_segment_at_position(segments, row_index, which, pos)
    if new_list is segments:
        return None, None
    return new_list, row_index


def merge_contiguous_segments(segments: List[dict], indices: List[int]) -> Tuple[Optional[List[dict]], Optional[int], Optional[Tuple[float, float]]]:
    """連続インデックス群を 1 つのプレースホルダセグメントへ統合。

    統合結果:
      - 最初の start と最後の end を保持
      - テキスト関連フィールドは空文字に初期化 (再文字起こしで再充填)

    返り値:
      - 成功: (新しいセグメントリスト, 挿入位置(先頭インデックス), (開始秒, 終了秒))
      - 失敗: (None, None, None)
    """
    if not indices:
        return None, None, None
    idx_sorted = sorted(indices)
    # contiguity check
    for a, b in zip(idx_sorted, idx_sorted[1:]):
        if b != a + 1:
            return None, None, None
    segs = as_segment_list(segments)
    try:
        selected = [segs[i] for i in idx_sorted]
    except Exception:
        return None, None, None
    start = float(selected[0].get('start', 0.0))
    end = float(selected[-1].get('end', start))
    if end <= start:
        return None, None, None
    # Build merged placeholder segment (id: keep first's id)
    first = selected[0]
    # Concatenate text from selected segments
    merged_text = ' '.join([s.get('text', '').strip() for s in selected if s.get('text')])
    placeholder = Segment(
        start=start, end=end,
        text=merged_text, text_lang1='', text_lang2='',
        chosen_language=None,
        id=first.get('id', idx_sorted[0]),
        lang1_prob=0.0, lang2_prob=0.0,
        lang1_code=first.get('lang1_code', 'ja'),
        lang2_code=first.get('lang2_code', 'ru'),
    )
    out: List[Segment] = []
    for i, s in enumerate(segs):
        if i == idx_sorted[0]:
            out.append(placeholder)
        if i in idx_sorted[1:]:
            continue
        if i not in idx_sorted:
            out.append(s)
    return [s.to_dict() for s in out], idx_sorted[0], (start, end)
