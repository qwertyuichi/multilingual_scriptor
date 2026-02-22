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
        - `txt` 形式: 各行に [開始 -> 終了] + 言語ごとの確率 + LANG1/LANG2 テキスト。
          LANG2 テキストが空の場合は出力しない (BUG-12 修正)。
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
            lang1_prob = seg.get('lang1_prob', 0.0); lang2_prob = seg.get('lang2_prob', 0.0)
            lang1_code = (seg.get('lang1_code') or 'lang1').upper()
            lang2_code = (seg.get('lang2_code') or '').upper()
            t1 = seg.get('text_lang1', '') or ''
            t2 = seg.get('text_lang2', '') or ''
            if not (t1 or t2):
                continue
            # BUG-12 修正: lang2 テキストが空なら lang2 部分を省略
            if t2 and lang2_code:
                line = (
                    f"[{fmt_ts(st)} -> {fmt_ts(ed)}] "
                    f"[{lang1_code}:{lang1_prob:05.2f}%] [{lang2_code}:{lang2_prob:05.2f}%] "
                    f"{lang1_code}={t1} | {lang2_code}={t2}"
                )
            else:
                line = (
                    f"[{fmt_ts(st)} -> {fmt_ts(ed)}] "
                    f"[{lang1_code}:{lang1_prob:05.2f}%] "
                    f"{lang1_code}={t1}"
                )
            lines.append(line)
        return '\n'.join(lines)
    if fmt == 'srt':
        blocks: List[str] = []
        n = 1
        for seg in segments:
            if seg.get('gap'):
                continue  # GAP は出力しない
            st = float(seg.get('start', 0.0)); ed = float(seg.get('end', 0.0))
            lang1_prob = seg.get('lang1_prob', 0.0); lang2_prob = seg.get('lang2_prob', 0.0)
            lang1_code = seg.get('lang1_code', 'ja')
            lang2_code = seg.get('lang2_code')
            t1 = seg.get('text_lang1', '') or ''
            t2 = seg.get('text_lang2', '') or ''
            chosen = seg.get('chosen_language')
            if chosen == lang1_code and t1:
                txt = t1
            elif lang2_code and chosen == lang2_code and t2:
                txt = t2
            else:
                txt = t1 if lang1_prob >= lang2_prob else (t2 or t1)
            if not txt:
                continue
            blocks.append(
                f"{n}\n{to_srt_timestamp(int(st*1000))} --> {to_srt_timestamp(int(ed*1000))}\n{txt}\n"
            )
            n += 1
        return '\n'.join(blocks)
    raise ValueError(f"Unsupported format: {fmt}")
