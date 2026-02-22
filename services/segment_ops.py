"""セグメントの純粋操作群 (Qt 非依存)。

提供関数:
 - `split_segment_at_position(segments, index, which, pos)`
     単一セグメントを指定言語テキストの文字位置で二分。
 - `adjust_boundary(segments, index_front, new_mid)`
     2 つの連続セグメント境界を新しい時刻 `new_mid` へ移動。

入出力はすべて辞書リスト (`list[dict]`)。内部では安全性と属性アクセス性のため `Segment` に正規化。
変更なしだった場合は元の `segments` をそのまま返すことで「失敗/無変更」をシグナル。
"""
from __future__ import annotations
from typing import List
from models.segment import as_segment_list, Segment
from utils.segment_utils import normalize_segment_id

def split_segment_at_position(segments: list[dict], index: int, which: str, pos: int) -> list[dict]:
    """指定インデックスの非 GAP セグメントを文字位置で二分し、新しいリストを返す。

    Parameters
    ----------
    segments : list[dict]
        セグメント辞書リスト。
    index : int
        分割対象行。
    which : str
        'lang1' または 'lang2'。その言語テキストの長さ基準で時間を按分。
    pos : int
        文字インデックス (0 < pos < len(text))。

    Returns
    -------
    list[dict]
        分割結果。条件不成立 (範囲外/GAP/空文字) の場合は入力 `segments` を返す。
    """
    segs = as_segment_list(segments)
    if index < 0 or index >= len(segs):
        return segments
    seg = segs[index]
    text_lang1 = seg.get('text_lang1', '')
    text_lang2 = seg.get('text_lang2', '')
    base = text_lang1 if which == 'lang1' else text_lang2
    if not base or pos <= 0 or pos >= len(base):
        return segments
    start = float(seg.get('start', 0.0)); end = float(seg.get('end', start))
    if end <= start:
        return segments
    dur = end - start
    # BUG-4 修正: 分割比率を算出し各言語テキストに独立に適用
    ratio = pos / len(base)
    mid_time = start + (dur * ratio)
    lang1_prob = seg.get('lang1_prob', 0.0); lang2_prob = seg.get('lang2_prob', 0.0)
    id_candidates = [normalize_segment_id(s, i) for i, s in enumerate(segs)]
    new_id_base = (max(id_candidates) + 1) if id_candidates else 0

    def split_by_ratio(full: str) -> tuple[str, str]:
        if not full:
            return '', ''
        cut = max(1, min(len(full) - 1, int(len(full) * ratio)))
        return full[:cut], full[cut:]

    lang1_a, lang1_b = split_by_ratio(text_lang1)
    lang2_a, lang2_b = split_by_ratio(text_lang2)
    # main text follows higher-prob language
    chosen = seg.get('chosen_language')
    lang1_code = seg.get('lang1_code', 'ja')
    lang2_code = seg.get('lang2_code', 'ru')
    def choose_text(t1: str, t2: str) -> str:
        if chosen == lang1_code:
            return t1
        if chosen == lang2_code:
            return t2
        return t1 if lang1_prob >= lang2_prob else t2

    first = Segment(
        start=start, end=mid_time,
        text=choose_text(lang1_a, lang2_a),
        text_lang1=lang1_a, text_lang2=lang2_a,
        chosen_language=chosen,
        id=seg.get('id', index),
        lang1_prob=lang1_prob, lang2_prob=lang2_prob,
        lang1_code=lang1_code, lang2_code=lang2_code,
    )
    second = Segment(
        start=mid_time, end=end,
        text=choose_text(lang1_b, lang2_b),
        text_lang1=lang1_b, text_lang2=lang2_b,
        chosen_language=chosen,
        id=new_id_base,
        lang1_prob=lang1_prob, lang2_prob=lang2_prob,
        lang1_code=lang1_code, lang2_code=lang2_code,
    )
    out = []
    for i, s in enumerate(segs):
        if i == index:
            out.append(first); out.append(second)
        else:
            out.append(s)
    return [s.to_dict() for s in out]

def adjust_boundary(segments: list[dict], index_front: int, new_mid: float, min_dur: float = 0.2) -> list[dict]:
    """連続する 2 セグメント (index_front とその次) の境界時刻を調整。

    成功条件:
      - index_front / index_front+1 が範囲内
      - どちらも GAP でない
      - start < new_mid < end を満たす
      - それぞれの最小長さが `min_dur` 以上確保できる

    条件を満たさない場合は元の `segments` を返す。
    """
    segs = as_segment_list(segments)
    r1 = index_front; r2 = index_front + 1
    if r1 < 0 or r2 >= len(segs):
        return segments
    s1 = segs[r1]; s2 = segs[r2]
    start = float(s1.get('start', 0.0))
    end = float(s2.get('end', start))
    if not (start < new_mid < end):
        return segments
    if (new_mid - start) < min_dur or (end - new_mid) < min_dur:
        return segments
    s1.end = new_mid
    s2.start = new_mid
    return [s.to_dict() for s in segs]
