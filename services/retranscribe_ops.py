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
    text_ja = seg.get('text_ja', '')
    text_ru = seg.get('text_ru', '')
    ja_prob = seg.get('ja_prob', 0.0); ru_prob = seg.get('ru_prob', 0.0)
    chosen = seg.get('chosen_language')
    if chosen == 'ja' and text_ja:
        base_text = text_ja; which = 'ja'
    elif chosen == 'ru' and text_ru:
        base_text = text_ru; which = 'ru'
    else:
        if ja_prob >= ru_prob:
            base_text = text_ja; which = 'ja'
        else:
            base_text = text_ru; which = 'ru'
    if not base_text:
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
    placeholder = Segment(
        start=start, end=end,
        text='', text_ja='', text_ru='',
        chosen_language=first.get('chosen_language'),
        id=first.get('id', idx_sorted[0]),
        ja_prob=0.0, ru_prob=0.0
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
