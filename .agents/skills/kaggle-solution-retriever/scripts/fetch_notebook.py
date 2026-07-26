#!/usr/bin/env python3
"""Fetch one selected Kaggle notebook and emit a bounded static excerpt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

from kaggle_core import (
    Catalog,
    CatalogError,
    DEFAULT_NOTEBOOK_CACHE,
    NOTEBOOK_REF_RE,
    extract_notebook_excerpt,
    resolve_db_path,
    stable_id,
)


SUPPORTED_SUFFIXES = {".ipynb", ".py", ".r", ".jl", ".sql", ".txt", ".md"}


def _ref_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme == "https"
        and parsed.netloc.casefold() in {"kaggle.com", "www.kaggle.com"}
        and len(parts) == 3
        and parts[0] == "code"
    ):
        ref = f"{parts[1]}/{parts[2]}"
        return ref if NOTEBOOK_REF_RE.fullmatch(ref) else ""
    return ""


def _trusted_local_source(
    raw_path: str, trusted_root: Optional[Path]
) -> Optional[Path]:
    if not raw_path or trusted_root is None:
        return None
    root = trusted_root.expanduser().resolve()
    unresolved = Path(raw_path).expanduser()
    if unresolved.is_symlink():
        raise CatalogError(f"Refusing catalog local_path symlink: {unresolved}")
    source = unresolved.resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise CatalogError(
            f"Catalog local_path escapes --local-code-root: {source}"
        ) from exc
    if not source.is_file():
        raise CatalogError(f"Trusted local source is not a regular file: {source}")
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CatalogError(f"Unsupported trusted local source type: {source.suffix}")
    return source


def _notebook_refs(row: Mapping[str, Any]) -> list[str]:
    values = [row["notebook_url"]]
    try:
        decoded = json.loads(row["notebook_urls"] or "[]")
    except json.JSONDecodeError:
        decoded = []
    if isinstance(decoded, list):
        values.extend(str(value) for value in decoded)
    refs: list[str] = []
    for value in values:
        ref = _ref_from_url(value)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _download(ref: str, target: Path) -> Path:
    if not NOTEBOOK_REF_RE.fullmatch(ref):
        raise CatalogError(f"Unsafe or invalid Kaggle notebook ref: {ref!r}")
    target.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "kernels", "pull", ref, "-p", str(target), "-m"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as exc:
        raise CatalogError("Kaggle CLI is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise CatalogError(f"Kaggle CLI timed out while pulling {ref}") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip().splitlines()
        detail = message[-1] if message else f"exit {completed.returncode}"
        raise CatalogError(
            f"Kaggle CLI failed: {detail}. Configure Kaggle CLI authentication and retry."
        )
    candidates = sorted(
        (
            path
            for path in target.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: (path.suffix.lower() != ".ipynb", path.name),
    )
    if not candidates:
        raise CatalogError(f"Kaggle pull produced no supported source file in {target}")
    return candidates[0]


def _read_query(args: argparse.Namespace, default: str) -> str:
    if args.query is not None:
        return args.query
    if args.query_file == "-":
        return sys.stdin.read()
    if args.query_file:
        return Path(args.query_file).read_text(encoding="utf-8")
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-id", required=True)
    parser.add_argument("--db", help="SQLite catalog path")
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--query")
    query.add_argument("--query-file", help="UTF-8 file path, or - for stdin")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_NOTEBOOK_CACHE)
    parser.add_argument(
        "--local-code-root",
        type=Path,
        help="Explicit trusted root required before catalog local_path is read",
    )
    parser.add_argument("--max-chars", type=int, default=8_000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_chars < 500:
        print("error: --max-chars must be at least 500", file=sys.stderr)
        return 2
    try:
        with Catalog(resolve_db_path(args.db)) as catalog:
            catalog.init()
            row = catalog.get_solution(args.solution_id)
            if not row:
                raise CatalogError(f"Unknown solution id: {args.solution_id!r}")
            if not row["public"]:
                raise CatalogError("Refusing to fetch a non-public catalog artifact")
            source_path = _trusted_local_source(
                row["local_path"],
                args.local_code_root,
            )
            if source_path is None:
                refs = _notebook_refs(row)
                ref = row["notebook_ref"] or (refs[0] if refs else "")
                if not ref:
                    local_hint = (
                        " Pass --local-code-root to authorize its catalog local_path."
                        if row["local_path"]
                        else ""
                    )
                    raise CatalogError(
                        "Selected artifact has no safe Kaggle notebook ref."
                        + local_hint
                    )
                source_path = _download(
                    ref,
                    args.cache_dir.expanduser().resolve()
                    / stable_id("notebook-cache", args.solution_id),
                )
            query = _read_query(
                args,
                " ".join(
                    [
                        row["competition_title"],
                        row["title"],
                        row["summary"],
                        row["tags"],
                    ]
                ),
            )
            excerpt = extract_notebook_excerpt(source_path, query, args.max_chars)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(excerpt, encoding="utf-8")
        else:
            sys.stdout.write(excerpt)
        return 0
    except (CatalogError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
