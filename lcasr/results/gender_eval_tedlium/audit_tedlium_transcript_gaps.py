#!/usr/bin/env python3
"""Audit TEDLIUM STM files for large gaps between consecutive transcribed utterances.

This is meant to help curate talks for gender-transfer experiments where we want
continuous transcribed audio and need to avoid recordings with long untranscribed
stretches inside them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Segment:
    start: float
    end: float


DEFAULT_TEDLIUM_ROOT = Path("/store/store4/data/TEDLIUM_release1/legacy")
DEFAULT_SPLITS = ("train", "dev", "test")


@dataclass
class GapEvent:
    after_segment_index: int
    previous_end: float
    next_start: float
    gap_seconds: float


@dataclass
class TalkAudit:
    split: str
    talk_id: str
    stm_path: str
    utterance_count: int
    ignored_segment_count: int
    first_start: float | None
    last_end: float | None
    transcript_span_seconds: float
    transcript_coverage_seconds: float
    max_gap_seconds: float
    large_gap_count: int
    large_gaps: list[GapEvent]
    has_large_gap: bool


def load_segments(stm_path: Path) -> tuple[list[Segment], int]:
    segments: list[Segment] = []
    ignored_segment_count = 0
    for line in stm_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        start = float(parts[3])
        end = float(parts[4])
        text = " ".join(parts[6:])
        if text == "ignore_time_segment_in_scoring":
            ignored_segment_count += 1
            continue
        segments.append(Segment(start=start, end=end))
    segments.sort(key=lambda seg: (seg.start, seg.end))
    return segments, ignored_segment_count


def iter_large_gaps_from_segments(segments: list[Segment], threshold_seconds: float) -> Iterable[GapEvent]:
    for idx, (prev_seg, next_seg) in enumerate(zip(segments, segments[1:])):
        gap_seconds = next_seg.start - prev_seg.end
        if gap_seconds > threshold_seconds:
            yield GapEvent(
                after_segment_index=idx,
                previous_end=prev_seg.end,
                next_start=next_seg.start,
                gap_seconds=gap_seconds,
            )


def parse_stm_file(stm_path: Path, threshold_seconds: float, max_events: int) -> TalkAudit:
    segments, ignored_segment_count = load_segments(stm_path)
    utterance_count = len(segments)
    first_start = segments[0].start if segments else None
    last_end = segments[-1].end if segments else None
    transcript_coverage_seconds = sum(max(0.0, seg.end - seg.start) for seg in segments)
    transcript_span_seconds = 0.0
    if first_start is not None and last_end is not None:
        transcript_span_seconds = max(0.0, last_end - first_start)

    all_large_gaps = list(iter_large_gaps_from_segments(segments, threshold_seconds))
    max_gap_seconds = max((event.gap_seconds for event in all_large_gaps), default=0.0)
    if max_gap_seconds == 0.0:
        for prev_seg, next_seg in zip(segments, segments[1:]):
            gap_seconds = next_seg.start - prev_seg.end
            if gap_seconds > 0:
                max_gap_seconds = max(max_gap_seconds, gap_seconds)

    large_gaps = all_large_gaps
    if max_events >= 0:
        large_gaps = sorted(all_large_gaps, key=lambda ev: ev.gap_seconds, reverse=True)[:max_events]

    return TalkAudit(
        split=stm_path.parent.parent.name,
        talk_id=stm_path.stem,
        stm_path=str(stm_path),
        utterance_count=utterance_count,
        ignored_segment_count=ignored_segment_count,
        first_start=first_start,
        last_end=last_end,
        transcript_span_seconds=transcript_span_seconds,
        transcript_coverage_seconds=transcript_coverage_seconds,
        max_gap_seconds=max_gap_seconds,
        large_gap_count=len(all_large_gaps),
        large_gaps=large_gaps,
        has_large_gap=(max_gap_seconds > threshold_seconds),
    )


def collect_audit(root: Path, splits: list[str], threshold_seconds: float, max_events: int) -> list[TalkAudit]:
    rows: list[TalkAudit] = []
    for split in splits:
        stm_dir = root / split / "stm"
        if not stm_dir.exists():
            raise FileNotFoundError(f"STM directory not found: {stm_dir}")
        for stm_path in sorted(stm_dir.glob("*.stm")):
            rows.append(parse_stm_file(stm_path, threshold_seconds, max_events))
    return rows


def build_summary(rows: list[TalkAudit], threshold_seconds: float) -> dict:
    by_split = {}
    for split in sorted({row.split for row in rows}):
        split_rows = [row for row in rows if row.split == split]
        clean_rows = [row for row in split_rows if not row.has_large_gap]
        by_split[split] = {
            "total_talks": len(split_rows),
            "clean_talks": len(clean_rows),
            "threshold_seconds": threshold_seconds,
            "clean_talk_ids": [row.talk_id for row in clean_rows],
            "worst_talks": [
                {
                    "talk_id": row.talk_id,
                    "max_gap_seconds": row.max_gap_seconds,
                    "large_gap_count": row.large_gap_count,
                }
                for row in sorted(split_rows, key=lambda row: row.max_gap_seconds, reverse=True)[:10]
            ],
        }
    return {
        "threshold_seconds": threshold_seconds,
        "total_talks": len(rows),
        "total_clean_talks": sum(1 for row in rows if not row.has_large_gap),
        "splits": by_split,
    }


def markdown_report(summary: dict, rows: list[TalkAudit]) -> str:
    lines = [
        f"# TEDLIUM transcript gap audit (threshold: {summary['threshold_seconds']:.2f}s)",
        "",
        f"Clean talks: {summary['total_clean_talks']} / {summary['total_talks']}",
        "",
    ]
    for split, split_summary in summary["splits"].items():
        lines.extend(
            [
                f"## {split}",
                f"- clean: {split_summary['clean_talks']} / {split_summary['total_talks']}",
                f"- clean talks: {', '.join(split_summary['clean_talk_ids']) if split_summary['clean_talk_ids'] else 'none'}",
                "- worst max gaps:",
            ]
        )
        for worst in split_summary["worst_talks"][:5]:
            lines.append(
                f"  - {worst['talk_id']}: max_gap={worst['max_gap_seconds']:.2f}s, large_gap_count={worst['large_gap_count']}"
            )
        lines.append("")

    flagged = [row for row in rows if row.has_large_gap]
    flagged.sort(key=lambda row: row.max_gap_seconds, reverse=True)
    lines.append("## Top flagged talks")
    for row in flagged[:15]:
        top_events = ", ".join(f"{event.gap_seconds:.2f}s" for event in row.large_gaps[:3]) or "n/a"
        lines.append(
            f"- {row.split}/{row.talk_id}: max_gap={row.max_gap_seconds:.2f}s, large_gap_count={row.large_gap_count}, top_gaps={top_events}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_TEDLIUM_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--threshold-seconds", type=float, default=2.0)
    parser.add_argument("--max-events", type=int, default=10, help="How many large-gap events to store per talk in the JSON output.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    rows = collect_audit(args.root, args.splits, args.threshold_seconds, args.max_events)
    summary = build_summary(rows, args.threshold_seconds)
    payload = {
        "summary": summary,
        "talks": [
            {
                **asdict(row),
                "large_gaps": [asdict(event) for event in row.large_gaps],
            }
            for row in rows
        ],
    }

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.md_out is not None:
        args.md_out.write_text(markdown_report(summary, rows))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
