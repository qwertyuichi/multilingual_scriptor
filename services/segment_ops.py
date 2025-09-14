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
        'ja' または 'ru'。その言語テキストの長さ基準で時間を按分。
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
    text_ja = seg.get('text_ja', '')
    text_ru = seg.get('text_ru', '')
    base = text_ja if which == 'ja' else text_ru
    if not base or pos <= 0 or pos >= len(base):
        return segments
    start = float(seg.get('start', 0.0)); end = float(seg.get('end', start))
    if end <= start:
        return segments
    dur = end - start
    mid_time = start + (dur * (pos / len(base)))
    ja_prob = seg.get('ja_prob', 0.0); ru_prob = seg.get('ru_prob', 0.0)
    id_candidates = [normalize_segment_id(s, i) for i, s in enumerate(segs)]
    new_id_base = (max(id_candidates) + 1) if id_candidates else 0
    def split_text(full: str):
        if not full:
            return '', ''
        return full[:pos], full[pos:]
    ja1, ja2 = split_text(text_ja)
    ru1, ru2 = split_text(text_ru)
    first = Segment(
        start=start, end=mid_time,
        text=ja1 if ja_prob >= ru_prob else ru1,
        text_ja=ja1, text_ru=ru1,
        chosen_language=seg.get('chosen_language'),
        id=seg.get('id', index), ja_prob=ja_prob, ru_prob=ru_prob
    )
    second = Segment(
        start=mid_time, end=end,
        text=ja2 if ja_prob >= ru_prob else ru2,
        text_ja=ja2, text_ru=ru2,
        chosen_language=seg.get('chosen_language'),
        id=new_id_base, ja_prob=ja_prob, ru_prob=ru_prob
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
