"""Segment データクラス / ユーティリティ。

目的: dict ベースセグメントとの互換ラッパ。

フィールド:
 - start, end      : タイムスタンプ (秒)
 - text            : 表示用テキスト (chosen_language 優先)
 - text_lang1      : 第1言語テキスト
 - text_lang2      : 第2言語テキスト
 - chosen_language : 選択言語コード ('ja', 'ru', 'silence', 'other', ...) または None
 - lang1_code      : 転写時の第1言語コード (デフォルト 'ja')
 - lang2_code      : 転写時の第2言語コード (デフォルト 'ru'、単言語モードは '')
 - lang1_prob      : 第1言語確率 (%)
 - lang2_prob      : 第2言語確率 (%)
 - gap             : True なら無音ギャップ扱い (silence と同義)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields as dc_fields
from typing import Any, Dict, Iterable

__all__ = ["Segment", "as_segment_list"]

@dataclass
class Segment:
    start: float
    end: float
    text: str = ""
    text_lang1: str = ""
    text_lang2: str = ""
    chosen_language: str | None = None
    id: int | str | None = None
    lang1_prob: float = 0.0
    lang2_prob: float = 0.0
    lang1_code: str = "ja"
    lang2_code: str = "ru"
    gap: bool = False

    # --- dict 互換 API ---
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __iter__(self):
        for k, v in asdict(self).items():
            yield k, v

    def get(self, key: str, default: Any = None) -> Any:
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
        known = {f.name for f in dc_fields(cls)}
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
        # start/end は必須: None なら 0.0 に補正 (BUG-29)
        kwargs["start"] = float(kwargs.get("start") or 0.0)
        kwargs["end"]   = float(kwargs.get("end")   or 0.0)
        return cls(**kwargs)


def as_segment_list(items: Iterable[Dict[str, Any] | Segment]) -> list[Segment]:
    out: list[Segment] = []
    for it in items:
        if isinstance(it, Segment):
            out.append(it)
        else:
            out.append(Segment.from_dict(it))
    return out
