"""Segment データクラス / ユーティリティ。

目的: 既存の dict ベースセグメントとの互換ラッパ。

満たす要件:
 - 属性: start, end, text, text_ja, text_ru, chosen_language, id, ja_prob, ru_prob, gap
 - dict 互換アクセス: `seg['start']` / `seg.get('start')`
 - `to_dict()` : 純粋な辞書へ変換 (JSON シリアライズ用)
 - `from_dict()` : 既存辞書から生成
 - `update(mapping)` : まとめて属性更新 (従来の `old.update({...})` 互換)

将来拡張 (例): confidence, notes などのメタ情報追加時の集約ポイント。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable

__all__ = ["Segment", "as_segment_list"]

@dataclass
class Segment:
    start: float
    end: float
    text: str = ""
    text_ja: str = ""
    text_ru: str = ""
    chosen_language: str | None = None
    id: int | str | None = None
    ja_prob: float = 0.0
    ru_prob: float = 0.0
    gap: bool = False

    # --- dict 互換 API ---
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # JSON 書き出し等でそのまま扱えるように
    def __iter__(self):  # allows dict(seg)
        for k, v in asdict(self).items():
            yield k, v

    def get(self, key: str, default: Any = None) -> Any:  # seg.get("start") 互換
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def update(self, mapping: Dict[str, Any]) -> None:
        for k, v in mapping.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        init_kwargs = {k: data.get(k) for k in fields}
        return cls(**init_kwargs)  # type: ignore[arg-type]


def as_segment_list(items: Iterable[Dict[str, Any] | Segment]) -> list[Segment]:
    out: list[Segment] = []
    for it in items:
        if isinstance(it, Segment):
            out.append(it)
        else:
            out.append(Segment.from_dict(it))
    return out
