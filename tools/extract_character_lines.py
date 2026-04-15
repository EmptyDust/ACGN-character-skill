"""
Extract character utterances from merged script exports.

Primary input is the merged script with explicit speaker markers (e.g. ``＃冬子``).
Optional CN text can be aligned by file name and utterance order to produce a
paired output for downstream character distillation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FILE_NAME_RE = re.compile(r"^FILE NAME:\s*(.+?)\s*$")
SPEAKER_MARK_RE = re.compile(r"^[＃#]\s*(.+?)\s*$")


@dataclass
class Utterance:
    file_name: str
    order: int
    speaker: str
    text: str


def _split_file_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_file: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        match = FILE_NAME_RE.match(line.strip())
        if match:
            if current_file is not None:
                sections.append((current_file, current_lines))
            current_file = match.group(1).strip()
            current_lines = []
            continue
        if current_file is not None:
            current_lines.append(line)

    if current_file is not None:
        sections.append((current_file, current_lines))

    return sections


def _normalize_dialogue_text(text: str) -> str:
    return text.strip()


def _next_nonempty_line(lines: list[str], start_idx: int) -> tuple[str | None, int | None]:
    for idx in range(start_idx, len(lines)):
        candidate = lines[idx].strip()
        if candidate:
            return candidate, idx
    return None, None


def extract_merged_utterances(text: str, target_speakers: set[str]) -> list[Utterance]:
    utterances: list[Utterance] = []

    for file_name, lines in _split_file_sections(text):
        order = 0
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            speaker_match = SPEAKER_MARK_RE.match(line)
            if not speaker_match:
                idx += 1
                continue

            speaker = speaker_match.group(1).strip()
            dialogue, next_idx = _next_nonempty_line(lines, idx + 1)
            idx += 1
            if speaker not in target_speakers or not dialogue or next_idx is None:
                continue

            if SPEAKER_MARK_RE.match(dialogue):
                continue

            order += 1
            utterances.append(
                Utterance(
                    file_name=file_name,
                    order=order,
                    speaker="朽木冬子",
                    text=_normalize_dialogue_text(dialogue),
                )
            )
            idx = next_idx + 1

    return utterances


def extract_cn_utterances(text: str, target_speakers: set[str]) -> list[Utterance]:
    utterances: list[Utterance] = []

    for file_name, lines in _split_file_sections(text):
        order = 0
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line not in target_speakers:
                idx += 1
                continue

            dialogue, next_idx = _next_nonempty_line(lines, idx + 1)
            idx += 1
            if not dialogue or next_idx is None:
                continue
            if dialogue in target_speakers:
                continue

            order += 1
            utterances.append(
                Utterance(
                    file_name=file_name,
                    order=order,
                    speaker="朽木冬子",
                    text=_normalize_dialogue_text(dialogue),
                )
            )
            idx = next_idx + 1

    return utterances


def pair_utterances_by_file_and_order(
    merged: Iterable[Utterance], cn: Iterable[Utterance]
) -> list[dict[str, str | int | None]]:
    cn_map = {(u.file_name, u.order): u for u in cn}
    paired: list[dict[str, str | int | None]] = []

    for utterance in merged:
        match = cn_map.get((utterance.file_name, utterance.order))
        paired.append(
            {
                "file_name": utterance.file_name,
                "order": utterance.order,
                "speaker": utterance.speaker,
                "text_ja": utterance.text,
                "text_zh": match.text if match else None,
            }
        )

    return paired


def write_outputs(records: list[dict[str, str | int | None]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_prefix.with_suffix(".jsonl")
    txt_path = output_prefix.with_suffix(".txt")
    md_path = output_prefix.with_suffix(".md")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        for record in records:
            file_stem = Path(str(record["file_name"])).stem
            text = record["text_zh"] or record["text_ja"] or ""
            f.write(f"[{file_stem}] 朽木冬子: {text}\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 朽木冬子台词摘录\n\n")
        for record in records:
            f.write(f"## {record['file_name']} #{record['order']}\n\n")
            f.write(f"- speaker: {record['speaker']}\n")
            f.write(f"- ja: {record['text_ja']}\n")
            if record["text_zh"]:
                f.write(f"- zh: {record['text_zh']}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Kuchiki Touko utterances from script dumps")
    parser.add_argument("merged_script", type=Path, help="Merged script file with explicit speaker markers")
    parser.add_argument("--cn-script", type=Path, default=None, help="Optional Chinese script for aligned output")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("characters/kuchiki-touko/knowledge/touko_lines"),
        help="Output prefix path without extension",
    )
    parser.add_argument(
        "--speaker-alias",
        action="append",
        default=["冬子", "朽木冬子"],
        help="Speaker alias to match. Can be provided multiple times.",
    )
    args = parser.parse_args()

    merged_text = args.merged_script.read_text(encoding="utf-8")
    target_speakers = set(args.speaker_alias)
    merged_utterances = extract_merged_utterances(merged_text, target_speakers)

    cn_utterances: list[Utterance] = []
    if args.cn_script:
        cn_text = args.cn_script.read_text(encoding="utf-8")
        cn_utterances = extract_cn_utterances(cn_text, target_speakers)

    records = pair_utterances_by_file_and_order(merged_utterances, cn_utterances)
    write_outputs(records, args.output_prefix)

    print(f"Extracted {len(records)} utterances")
    print(f"JSONL: {args.output_prefix.with_suffix('.jsonl')}")
    print(f"TXT:   {args.output_prefix.with_suffix('.txt')}")
    print(f"MD:    {args.output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
