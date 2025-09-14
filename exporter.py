"""書き出し(エクスポート)関連ユーティリティ。

責務:
 - TXT / SRT 形式のテキスト文字列を構築
 - JSON (単一オブジェクト) 保存 / 連携用の辞書ペイロードを生成

UI (`main.py`) から利用される公開関数:
 - `build_export_text(result: dict, fmt: str) -> str`
 - `build_json_payload(result: dict, meta: dict) -> dict`

前提:
 - `result['segments']` はセグメント辞書のリスト。
 - Segment は呼び出し側では dict で扱われる想定で、必要に応じて `as_segment_list` でラップする。
"""
from __future__ import annotations
from typing import Dict, Any, List
from utils.timefmt import format_ms, to_srt_timestamp
from models.segment import as_segment_list

def build_json_payload(result: dict, meta: dict) -> dict:
    """書き出し用 JSON ペイロードを構築して返す。

    Parameters
    ----------
    result : dict
        文字起こし結果全体。`result['segments']` を参照。
    meta : dict
        モデル名・デバイス名・動画パスなど付帯情報。

    Returns
    -------
    dict
        シリアライズしやすい平坦な辞書。
    """
    payload = {
        'video_path': meta.get('video_path'),
        'model': meta.get('model'),
        'device': meta.get('device'),
        'segments': result.get('segments', []),
    }
    return payload

def build_export_text(result: dict, fmt: str) -> str:
    """指定フォーマット(`txt` / `srt`)の書き出し文字列を生成。

    Notes:
        - `txt` 形式: 各行に [開始 -> 終了] + 言語ごとの確率 + JA/RU テキスト。
        - `srt` 形式: SRT 互換。優勢(確率の高い)言語テキストを採用。
        - JSON 出力は `build_json_payload` を利用し呼び出し側で `json.dump` してください。
    """
    fmt = fmt.lower()
    segments = as_segment_list(result.get('segments', []))
    if fmt == 'txt':
        lines: List[str] = []
        def fmt_ts(sec: float) -> str:
            return format_ms(int(sec * 1000))
        for seg in segments:
            if seg.get('gap'):
                continue  # GAP は出力しない
            st = float(seg.get('start', 0.0)); ed = float(seg.get('end', 0.0))
            ja_prob = seg.get('ja_prob', 0.0); ru_prob = seg.get('ru_prob', 0.0)
            ja_text = seg.get('text_ja', '') or ''
            ru_text = seg.get('text_ru', '') or ''
            if not (ja_text or ru_text):
                continue
            lines.append(
                f"[{fmt_ts(st)} -> {fmt_ts(ed)}] [JA:{ja_prob:05.2f}%] [RU:{ru_prob:05.2f}%] "
                f"JA={ja_text} | RU={ru_text}"
            )
        return '\n'.join(lines)
    if fmt == 'srt':
        blocks: List[str] = []
        n = 1
        for seg in segments:
            if seg.get('gap'):
                continue  # GAP は出力しない
            st = float(seg.get('start', 0.0)); ed = float(seg.get('end', 0.0))
            ja_prob = seg.get('ja_prob', 0.0); ru_prob = seg.get('ru_prob', 0.0)
            text_ja = seg.get('text_ja', '') or ''
            text_ru = seg.get('text_ru', '') or ''
            chosen = seg.get('chosen_language')
            if chosen == 'ja' and text_ja:
                txt = text_ja
            elif chosen == 'ru' and text_ru:
                txt = text_ru
            else:
                txt = text_ja if ja_prob >= ru_prob else text_ru
            if not txt:
                continue
            blocks.append(
                f"{n}\n{to_srt_timestamp(int(st*1000))} --> {to_srt_timestamp(int(ed*1000))}\n{txt}\n"
            )
            n += 1
        return '\n'.join(blocks)
    raise ValueError(f"Unsupported format: {fmt}")
