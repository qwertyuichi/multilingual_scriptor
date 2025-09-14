"""テーブル表示 / 集約テキスト再構築用ヘルパ。

`VideoTranscriptionApp` から分離した UI 更新ロジック群。

提供関数:
 - `rebuild_aggregate_text(result: dict)` : セグメント表示テキストを再構築し `result['text']` を副作用更新。
 - `populate_table(table: QTableWidget, result: dict)` : テーブルへセグメントを行として反映。
"""
from __future__ import annotations
from typing import Dict, Any
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
from PySide6.QtCore import Qt
from models.segment import as_segment_list
from utils.timefmt import format_ms
from utils.segment_utils import display_text

def rebuild_aggregate_text(result: Dict[str, Any]) -> None:
    segs = as_segment_list(result.get('segments', []))
    lines: list[str] = []
    for s in segs:
        dt = display_text(s)
        # dict へ戻す際も text 更新しておく (GAP なら空文字)
        s['text'] = dt
        if dt:
            lines.append(dt)
    # 反映
    result['segments'] = [s.to_dict() for s in segs]
    result['text'] = '\n'.join(lines)


def populate_table(table: QTableWidget, result: Dict[str, Any]) -> None:
    table.setRowCount(0)
    segments = as_segment_list(result.get('segments', []))
    for seg in segments:
        row = table.rowCount()
        table.insertRow(row)
        start_sec = seg.get('start', 0.0)
        end_sec = seg.get('end', 0.0)
        start_str = format_ms(int(start_sec * 1000))
        end_str = format_ms(int(end_sec * 1000))
        # GAP でも通常行として空文字表示 (色付与不要で簡潔化)
        ja_prob = seg.get('ja_prob', 0.0)
        ru_prob = seg.get('ru_prob', 0.0)
        disp_txt = display_text(seg)
        start_item = QTableWidgetItem(start_str)
        end_item = QTableWidgetItem(end_str)
        ja_item = QTableWidgetItem(f"{ja_prob:.2f}")
        ru_item = QTableWidgetItem(f"{ru_prob:.2f}")
        text_item = QTableWidgetItem(disp_txt)
        for it in (start_item, end_item, ja_item, ru_item):
            it.setTextAlignment(Qt.AlignCenter)
        from PySide6.QtGui import QColor
        if ja_prob >= ru_prob:
            ja_item.setForeground(QColor(200, 0, 0))
            ru_item.setForeground(QColor(0, 0, 180))
        else:
            ru_item.setForeground(QColor(200, 0, 0))
            ja_item.setForeground(QColor(0, 0, 180))
        table.setItem(row, 0, start_item)
        table.setItem(row, 1, end_item)
        table.setItem(row, 2, ja_item)
        table.setItem(row, 3, ru_item)
        table.setItem(row, 4, text_item)
