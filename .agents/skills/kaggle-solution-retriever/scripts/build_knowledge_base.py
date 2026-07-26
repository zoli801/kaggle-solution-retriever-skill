#!/usr/bin/env python3
"""Build the complete task-specific Kaggle knowledge base and compact context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from kaggle_core import Catalog, CatalogError, resolve_db_path
from knowledge_base import (
    build_knowledge_base,
    final_report,
    render_context_summary,
    render_knowledge_base,
)


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt is not None:
        value = prompt
    elif prompt_file == "-":
        value = sys.stdin.read()
    elif prompt_file:
        value = Path(prompt_file).expanduser().read_text(encoding="utf-8")
    else:
        raise CatalogError("Provide --prompt or --prompt-file")
    value = value.strip()
    if not value:
        raise CatalogError("Prompt is empty")
    if len(value) > 100_000:
        raise CatalogError("Prompt exceeds the 100,000-character safety limit")
    return value


def _read_profile(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if not path:
        return None
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid task-profile JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogError("Task-profile JSON must contain an object")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", help="UTF-8 path, or - for stdin")
    parser.add_argument(
        "--task-profile",
        type=Path,
        help="Optional Codex-authored JSON profile overriding heuristic fields",
    )
    parser.add_argument("--db", help="SQLite catalog path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kaggle_relevant_solutions_knowledge_base.md"),
    )
    parser.add_argument(
        "--context-output",
        type=Path,
        default=Path("kaggle_relevant_solutions_context.md"),
    )
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--min-competitions", type=int, default=3)
    parser.add_argument("--max-competitions", type=int, default=10)
    parser.add_argument("--initial-threshold", type=float, default=0.62)
    parser.add_argument("--minimum-threshold", type=float, default=0.44)
    parser.add_argument("--max-context-chars", type=int, default=14_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 3 <= args.min_competitions <= args.max_competitions <= 10:
        print(
            "error: competition limits must satisfy 3 <= minimum <= maximum <= 10",
            file=sys.stderr,
        )
        return 2
    if not 0 <= args.minimum_threshold <= args.initial_threshold <= 1:
        print("error: invalid relevance threshold range", file=sys.stderr)
        return 2
    if args.max_context_chars < 2_000:
        print("error: --max-context-chars must be at least 2000", file=sys.stderr)
        return 2
    try:
        prompt = _read_prompt(args.prompt, args.prompt_file)
        profile = _read_profile(args.task_profile)
        output = args.output.expanduser().resolve()
        context_output = args.context_output.expanduser().resolve()
        with Catalog(resolve_db_path(args.db)) as catalog:
            catalog.init()
            result = build_knowledge_base(
                catalog,
                prompt,
                profile_override=profile,
                min_competitions=args.min_competitions,
                max_competitions=args.max_competitions,
                initial_threshold=args.initial_threshold,
                minimum_threshold=args.minimum_threshold,
            )
        _atomic_write(output, render_knowledge_base(result))
        _atomic_write(
            context_output,
            render_context_summary(result, output, max_chars=args.max_context_chars),
        )
        report = final_report(result, output)
        report["context_summary_path"] = str(context_output)
        if args.report_output:
            _atomic_write(
                args.report_output,
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            report["report_path"] = str(args.report_output.expanduser().resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CatalogError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
