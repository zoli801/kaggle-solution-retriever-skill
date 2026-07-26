#!/usr/bin/env python3
"""Prepare reviewed catalog JSONL from one competition's leaderboard exports.

The input is one JSON object with this shape:

{
  "competition": {
    "id": "competition-slug",
    "slug": "competition-slug",
    "title": "Competition title",
    "status": "completed",
    "url": "https://www.kaggle.com/competitions/competition-slug"
  },
  "public_leaderboard": [
    {
      "rank": 1,
      "team_id": "123",
      "team_name": "Example team",
      "member_handles": ["alice", "bob"],
      "score": "0.91",
      "writeup_url":
        "https://www.kaggle.com/competitions/competition-slug/writeups/example"
    }
  ],
  "private_leaderboard": [],
  "team_analyses": [
    {
      "team_id": "123",
      "core_idea": "Source-stated method",
      "extracted_facts": ["The writeup states ..."],
      "analyst_inferences": ["Analysis: ..."]
    }
  ],
  "leaderboard_url":
    "https://www.kaggle.com/competitions/competition-slug/leaderboard",
  "ranks_verified_at": "2026-07-26T10:00:00Z",
  "public_scan_exhausted": false,
  "private_scan_exhausted": true
}

Each board export must contain a complete rank prefix beginning at rank 1 and is
sorted by its own rank before selection. A team is eligible only when the same
identity is present on both boards, both ranks are positive, and a leaderboard
row contains an official same-competition Kaggle /writeups/ URL. The first five
eligible teams on each board are selected independently. If a prefix produces
fewer than five, the corresponding *_scan_exhausted flag must attest that no
more leaderboard rows are available. Overlap is emitted once as one
leaderboard_team plus one canonical solution record. Each opposite-board prefix
must also extend through every provisional selection's cross-rank; missing
cross-evidence is an error and never causes substitution by a lower rank.

Votes, upvotes, likes, and similar popularity fields are never read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import unquote, urlparse, urlunparse


SELECTION_SIZE = 5
KAGGLE_HOSTS = {"kaggle.com", "www.kaggle.com"}

COMPETITION_FIELDS = {
    "id",
    "slug",
    "title",
    "description",
    "end_date",
    "status",
    "tags",
    "url",
    "source_updated_at",
    "task_types",
    "target_types",
    "modalities",
    "metrics",
    "dataset_structure",
    "validation_structure",
    "leakage_risks",
    "transferable_methods",
    "feature_methods",
    "domains",
    "compute_profiles",
    "constraints",
    "profile_source_url",
    "profile_verified",
    "profile_updated_at",
}

SOLUTION_TEXT_FIELDS = {
    "summary",
    "core_idea",
    "validation_strategy",
    "preprocessing",
    "feature_engineering",
    "models",
    "training_procedure",
    "ensembling",
    "post_processing",
    "leakage_prevention",
    "failed_approaches",
    "compute_requirements",
    "robustness_notes",
    "transferable_ideas",
    "application_risks",
}

SOLUTION_LIST_FIELDS = {
    "techniques",
    "validation_tags",
    "model_tags",
    "feature_tags",
    "ensemble_tags",
    "extracted_facts",
    "analyst_inferences",
}

SOLUTION_URL_LIST_FIELDS = {
    "code_urls",
    "notebook_urls",
    "repository_urls",
    "external_urls",
}

VOTE_FIELD_RE = re.compile(
    r"(?:^|_)(?:vote|votes|upvote|upvotes|like|likes|popularity)(?:$|_)",
    re.IGNORECASE,
)


class ManifestError(ValueError):
    """Raised when a research export cannot be converted safely."""


@dataclass(frozen=True)
class BoardRow:
    """Normalized row from one leaderboard."""

    board: str
    source_index: int
    own_rank: int
    team_id: str
    team_name: str
    member_handles: Tuple[str, ...]
    fallback_signature: Tuple[str, Tuple[str, ...]]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class Candidate:
    """One verified cross-board team with an official writeup route."""

    team_id: str
    team_name: str
    member_handles: Tuple[str, ...]
    public_row: BoardRow
    private_row: BoardRow
    public_rank: int
    private_rank: int
    public_score: str
    private_score: str
    writeup_url: str


def _text(value: Any) -> str:
    return str(value or "").strip()


def _url_text(value: Any, field: str) -> str:
    raw = str(value or "")
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in raw
    ):
        raise ManifestError(f"{field} contains whitespace or control characters")
    return raw


def _strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = _text(value).casefold()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"", "0", "false", "no"}:
        return False
    raise ManifestError(f"{field} must be a boolean")


def _iso_timestamp(value: Any, field: str) -> str:
    timestamp = _text(value)
    if not timestamp:
        raise ManifestError(f"{field} is required")
    try:
        parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"{field} must include a timezone")
    return timestamp


def _positive_rank(value: Any) -> Optional[int]:
    if value is None or _text(value) == "":
        return None
    try:
        rank = int(float(_text(value).replace(",", "")))
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _string_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    result: List[str] = []
    seen = set()
    for item in value:
        text = _text(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _member_handles(row: Mapping[str, Any]) -> Tuple[str, ...]:
    raw_members = row.get("member_handles")
    if raw_members is None:
        raw_members = row.get("members")
    if raw_members is None:
        return ()
    if not isinstance(raw_members, (list, tuple)):
        raw_members = [raw_members]
    handles: List[str] = []
    for member in raw_members:
        if isinstance(member, Mapping):
            handle = _text(
                member.get("handle")
                or member.get("username")
                or member.get("user_name")
            )
        else:
            handle = _text(member)
        if handle:
            handles.append(handle)
    return tuple(sorted(set(handles)))


def _fallback_signature(row: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return (_text(row.get("team_name")), _member_handles(row))


def _fallback_team_id(
    competition_id: str,
    team_name: str,
    member_handles: Iterable[str],
) -> str:
    exact_identity = json.dumps(
        [competition_id, team_name, sorted(member_handles)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(exact_identity.encode("utf-8")).hexdigest()[:24]
    return f"team-{digest}"


def _canonical_http_url(value: Any, field: str) -> str:
    url = _url_text(value, field)
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ManifestError(f"{field} must be an http(s) URL: {url!r}")
    return url


def _official_writeup_url(value: Any, competition_slug: str) -> str:
    url = _url_text(value, "writeup_url")
    if not url:
        return ""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or host not in KAGGLE_HOSTS
        or len(parts) != 4
        or parts[0] != "competitions"
        or parts[1] != competition_slug
        or parts[2] != "writeups"
        or not parts[3]
    ):
        return ""
    canonical_path = "/" + "/".join(parts)
    return urlunparse(("https", "www.kaggle.com", canonical_path, "", "", ""))


def _official_competition_url(value: Any, competition_slug: str) -> str:
    url = _url_text(value, "competition.url")
    if not url:
        return ""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or host not in KAGGLE_HOSTS
        or parts != ["competitions", competition_slug]
    ):
        raise ManifestError(
            "competition URL must be the official same-competition HTTPS "
            f"Kaggle route: {url!r}"
        )
    return urlunparse(
        (
            "https",
            "www.kaggle.com",
            f"/competitions/{competition_slug}",
            "",
            "",
            "",
        )
    )


def _official_leaderboard_url(value: Any, competition_slug: str) -> str:
    url = _url_text(value, "leaderboard source URL")
    if not url:
        return ""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or host not in KAGGLE_HOSTS
        or parts != ["competitions", competition_slug, "leaderboard"]
    ):
        raise ManifestError(
            "leaderboard source must be the same competition's official "
            f"Kaggle leaderboard: {url!r}"
        )
    return urlunparse(
        (
            "https",
            "www.kaggle.com",
            f"/competitions/{competition_slug}/leaderboard",
            "",
            "",
            "",
        )
    )


def _normalize_board_rows(
    rows: Any,
    board: str,
    skipped: Counter,
) -> List[BoardRow]:
    if not isinstance(rows, list):
        raise ManifestError(f"{board}_leaderboard must be an array")
    rank_field = f"{board}_rank"
    normalized: List[BoardRow] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            skipped[f"{board}: row is not an object"] += 1
            continue
        rank = _positive_rank(raw.get(rank_field, raw.get("rank")))
        if rank is None:
            skipped[f"{board}: missing or invalid own rank"] += 1
            continue
        team_id = _text(raw.get("team_id"))
        team_name, handles = _fallback_signature(raw)
        if not team_id and (not team_name or not handles):
            skipped[
                f"{board}: identity needs team_id or exact team_name plus member handles"
            ] += 1
            continue
        normalized.append(
            BoardRow(
                board=board,
                source_index=index,
                own_rank=rank,
                team_id=team_id,
                team_name=team_name,
                member_handles=handles,
                fallback_signature=(team_name, handles),
                raw=raw,
            )
        )
    normalized = sorted(normalized, key=lambda row: (row.own_rank, row.source_index))
    ranks = [row.own_rank for row in normalized]
    if ranks:
        expected = list(range(1, ranks[-1] + 1))
        if ranks != expected:
            raise ManifestError(
                f"{board}_leaderboard must contain exactly one row for every "
                f"rank from 1 through {ranks[-1]}; received {ranks!r}"
            )
    return normalized


def _row_match_candidates(
    row: BoardRow,
    other_rows: Sequence[BoardRow],
) -> List[BoardRow]:
    exact_id = (
        [other for other in other_rows if other.team_id == row.team_id]
        if row.team_id
        else []
    )
    if exact_id:
        return exact_id

    team_name, handles = row.fallback_signature
    if not team_name or not handles:
        return []
    fallback_matches = [
        other
        for other in other_rows
        if other.fallback_signature == row.fallback_signature
        and not (row.team_id and other.team_id and row.team_id != other.team_id)
    ]
    return fallback_matches


def _cross_rank_is_consistent(
    public_row: BoardRow,
    private_row: BoardRow,
) -> bool:
    public_cross = public_row.raw.get("private_rank")
    private_cross = private_row.raw.get("public_rank")
    if public_cross not in (None, ""):
        parsed = _positive_rank(public_cross)
        if parsed is None or parsed != private_row.own_rank:
            return False
    if private_cross not in (None, ""):
        parsed = _positive_rank(private_cross)
        if parsed is None or parsed != public_row.own_rank:
            return False
    return True


def _candidate_for_row(
    row: BoardRow,
    other_rows: Sequence[BoardRow],
    competition_id: str,
    competition_slug: str,
) -> Tuple[Optional[Candidate], str]:
    matches = _row_match_candidates(row, other_rows)
    if not matches:
        return None, "no cross-board identity"
    if len(matches) != 1:
        return None, "ambiguous cross-board identity"
    other = matches[0]
    public_row = row if row.board == "public" else other
    private_row = row if row.board == "private" else other
    if not _cross_rank_is_consistent(public_row, private_row):
        return None, "conflicting cross-board rank"

    canonical_urls = {
        candidate
        for candidate in (
            _official_writeup_url(public_row.raw.get("writeup_url"), competition_slug),
            _official_writeup_url(private_row.raw.get("writeup_url"), competition_slug),
        )
        if candidate
    }
    if not canonical_urls:
        return None, "no official same-competition writeup URL"
    if len(canonical_urls) > 1:
        return None, "conflicting official writeup URLs"

    supplied_ids = {item for item in (public_row.team_id, private_row.team_id) if item}
    if len(supplied_ids) > 1:
        return None, "conflicting team IDs"
    team_name = public_row.team_name or private_row.team_name
    member_handles = public_row.member_handles or private_row.member_handles
    team_id = (
        next(iter(supplied_ids))
        if supplied_ids
        else _fallback_team_id(competition_id, team_name, member_handles)
    )

    public_score = _text(
        public_row.raw.get("public_score", public_row.raw.get("score"))
    )
    private_score = _text(
        private_row.raw.get("private_score", private_row.raw.get("score"))
    )
    return (
        Candidate(
            team_id=team_id,
            team_name=team_name,
            member_handles=member_handles,
            public_row=public_row,
            private_row=private_row,
            public_rank=public_row.own_rank,
            private_rank=private_row.own_rank,
            public_score=public_score,
            private_score=private_score,
            writeup_url=next(iter(canonical_urls)),
        ),
        "",
    )


def _select_board(
    rows: Sequence[BoardRow],
    other_rows: Sequence[BoardRow],
    competition_id: str,
    competition_slug: str,
    skipped: Counter,
) -> Tuple[List[Candidate], int]:
    selected: List[Candidate] = []
    selected_ids = set()
    scanned = 0
    for row in rows:
        if len(selected) >= SELECTION_SIZE:
            break
        scanned += 1
        own_writeup = _official_writeup_url(
            row.raw.get("writeup_url"),
            competition_slug,
        )
        if not own_writeup:
            skipped[f"{row.board}: no official same-competition writeup URL"] += 1
            continue
        candidate, reason = _candidate_for_row(
            row,
            other_rows,
            competition_id,
            competition_slug,
        )
        if candidate is None:
            opposite = "Private" if row.board == "public" else "Public"
            raise ManifestError(
                f"{row.board.title()} rank {row.own_rank} has a qualifying "
                f"official writeup but its {opposite} cross-rank evidence is "
                f"incomplete ({reason}). Extend the complete {opposite} rank "
                "prefix through this team; do not substitute a lower-ranked "
                "writeup."
            )
        if candidate.team_id in selected_ids:
            skipped[f"{row.board}: duplicate team row"] += 1
            continue
        selected_ids.add(candidate.team_id)
        selected.append(candidate)
    return selected, scanned


def _analysis_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = payload.get("team_analyses", payload.get("analyses", []))
    if raw in (None, ""):
        return []
    if isinstance(raw, Mapping):
        rows: List[Mapping[str, Any]] = []
        for identity, analysis in raw.items():
            if not isinstance(analysis, Mapping):
                raise ManifestError("Each team analysis must be an object")
            copied = dict(analysis)
            copied.setdefault("team_id", identity)
            rows.append(copied)
        return rows
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise ManifestError("team_analyses must be an array or object map")
    return list(raw)


def _analysis_for(
    candidate: Candidate,
    analyses: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_id = [
        item for item in analyses if _text(item.get("team_id")) == candidate.team_id
    ]
    if len(by_id) > 1:
        raise ManifestError(f"Multiple analyses use team_id {candidate.team_id!r}")
    if by_id:
        return by_id[0]

    fallback = [
        item
        for item in analyses
        if _fallback_signature(item) == (candidate.team_name, candidate.member_handles)
    ]
    if len(fallback) > 1:
        raise ManifestError(f"Multiple analyses match team {candidate.team_name!r}")
    return fallback[0] if fallback else {}


def _first_value(
    rows: Sequence[Mapping[str, Any]],
    *fields: str,
) -> Any:
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value not in (None, "", []):
                return value
    return ""


def _validate_url_list(value: Any, field: str) -> List[str]:
    result: List[str] = []
    for raw in _string_list(value):
        url = _canonical_http_url(raw, field)
        if url not in result:
            result.append(url)
    return result


def _leaderboard_source(
    payload: Mapping[str, Any],
    candidate: Candidate,
    board: str,
    competition_slug: str,
) -> str:
    row = candidate.public_row if board == "public" else candidate.private_row
    value = (
        row.raw.get(f"{board}_rank_source_url")
        or row.raw.get("rank_source_url")
        or payload.get(f"{board}_rank_source_url")
        or payload.get("leaderboard_url")
    )
    if not value:
        raise ManifestError(
            f"{board}_rank_source_url or leaderboard_url is required; "
            "the converter never fabricates rank evidence"
        )
    return _official_leaderboard_url(value, competition_slug)


def _competition_record(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw = payload.get("competition")
    if not isinstance(raw, Mapping):
        raise ManifestError("competition must be an object")
    slug = _text(raw.get("slug"))
    if not slug:
        raise ManifestError("competition.slug is required")
    title = _text(raw.get("title"))
    if not title:
        raise ManifestError("competition.title is required")
    if _text(raw.get("status")).lower() != "completed":
        raise ManifestError("Only status='completed' competitions are accepted")
    competition_id = _text(raw.get("id")) or slug
    record = {field: raw[field] for field in COMPETITION_FIELDS if field in raw}
    record.update(
        {
            "record_type": "competition",
            "id": competition_id,
            "slug": slug,
            "title": title,
            "status": "completed",
            "url": _official_competition_url(
                raw.get("url") or f"https://www.kaggle.com/competitions/{slug}",
                slug,
            ),
        }
    )
    return record


def _leaderboard_record(
    payload: Mapping[str, Any],
    competition: Mapping[str, Any],
    candidate: Candidate,
) -> Dict[str, Any]:
    slug = _text(competition["slug"])
    return {
        "record_type": "leaderboard_team",
        "competition_id": competition["id"],
        "team_id": candidate.team_id,
        "team_name": candidate.team_name,
        "public_rank": candidate.public_rank,
        "private_rank": candidate.private_rank,
        "public_score": candidate.public_score,
        "private_score": candidate.private_score,
        "public_rank_verified": True,
        "private_rank_verified": True,
        "public_rank_source_url": _leaderboard_source(
            payload, candidate, "public", slug
        ),
        "private_rank_source_url": _leaderboard_source(
            payload, candidate, "private", slug
        ),
        "ranks_verified_at": _iso_timestamp(
            payload.get("ranks_verified_at"),
            "ranks_verified_at",
        ),
    }


def _solution_record(
    payload: Mapping[str, Any],
    competition: Mapping[str, Any],
    candidate: Candidate,
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    source_rows = [
        analysis,
        candidate.public_row.raw,
        candidate.private_row.raw,
        payload,
    ]
    title = (
        _text(_first_value(source_rows, "title", "writeup_title"))
        or f"{candidate.team_name or candidate.team_id} Solution Writeup"
    )
    author = _text(_first_value(source_rows, "author"))
    publication_author = _text(_first_value(source_rows, "publication_author"))
    record: Dict[str, Any] = {
        "record_type": "solution",
        "id": (
            f"{competition['id']}-solution-"
            + hashlib.sha256(
                f"{competition['id']}\x1f{candidate.team_id}\x1f"
                f"{candidate.writeup_url}".encode("utf-8")
            ).hexdigest()[:20]
        ),
        "competition_id": competition["id"],
        "team_id": candidate.team_id,
        "team_name": candidate.team_name,
        "title": title,
        "writeup_title": _text(_first_value(source_rows, "writeup_title", "title"))
        or title,
        "author": author,
        "publication_author": publication_author,
        "private_rank": candidate.private_rank,
        "public_rank": candidate.public_rank,
        "private_rank_verified": True,
        "public_rank_verified": True,
        "private_rank_source_url": _leaderboard_source(
            payload, candidate, "private", _text(competition["slug"])
        ),
        "public_rank_source_url": _leaderboard_source(
            payload, candidate, "public", _text(competition["slug"])
        ),
        "private_score": candidate.private_score,
        "public_score": candidate.public_score,
        "ranks_verified_at": _iso_timestamp(
            payload.get("ranks_verified_at"),
            "ranks_verified_at",
        ),
        "public": True,
        "source_kind": "writeup",
        "writeup_url": candidate.writeup_url,
        "solution_verified": True,
        "writeup_verified": True,
        "writeup_verified_at": _iso_timestamp(
            _first_value(
                source_rows,
                "writeup_verified_at",
                "source_accessed_at",
                "ranks_verified_at",
            ),
            "writeup_verified_at",
        ),
        "source_accessed_at": _iso_timestamp(
            _first_value(
                source_rows,
                "source_accessed_at",
                "writeup_verified_at",
                "ranks_verified_at",
            ),
            "source_accessed_at",
        ),
        "verification_note": (
            "The supplied Public and Private leaderboard exports contain the "
            "same team identity and verified cross-ranks; a leaderboard row "
            "contains an official same-competition Kaggle /writeups/ URL."
        ),
        "tags": competition.get("tags", []),
        "license": _text(_first_value(source_rows, "license"))
        or "unknown/source-specific",
        "provenance_url": candidate.writeup_url,
        "confidence": _text(_first_value(source_rows, "confidence")) or "unknown",
    }

    for field in sorted(SOLUTION_TEXT_FIELDS):
        record[field] = _text(_first_value(source_rows, field))
    for field in sorted(SOLUTION_LIST_FIELDS):
        record[field] = _string_list(_first_value(source_rows, field))
    for field in sorted(SOLUTION_URL_LIST_FIELDS):
        record[field] = _validate_url_list(
            _first_value(source_rows, field),
            f"solution.{field}",
        )
    return record


def _selection_report_rows(
    candidates: Sequence[Candidate],
) -> List[Dict[str, Any]]:
    return [
        {
            "selected_position": index,
            "team_id": candidate.team_id,
            "team_name": candidate.team_name,
            "public_rank": candidate.public_rank,
            "private_rank": candidate.private_rank,
            "public_score": candidate.public_score,
            "private_score": candidate.private_score,
            "writeup_url": candidate.writeup_url,
        }
        for index, candidate in enumerate(candidates, 1)
    ]


def _count_vote_fields(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(bool(VOTE_FIELD_RE.search(_text(key)))) + _count_vote_fields(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_count_vote_fields(item) for item in value)
    return 0


def prepare_manifest(
    payload: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return catalog-compatible records and an auditable selection report."""
    if not isinstance(payload, Mapping):
        raise ManifestError("Input JSON must be one object")

    competition = _competition_record(payload)
    _iso_timestamp(payload.get("ranks_verified_at"), "ranks_verified_at")
    competition_id = _text(competition["id"])
    competition_slug = _text(competition["slug"])
    source_values = {
        board: (
            payload.get(f"{board}_rank_source_url") or payload.get("leaderboard_url")
        )
        for board in ("public", "private")
    }
    for board, source_value in source_values.items():
        if not source_value:
            raise ManifestError(
                f"{board}_rank_source_url or leaderboard_url is required"
            )
        _official_leaderboard_url(source_value, competition_slug)
    public_exhausted = _strict_bool(
        payload.get("public_scan_exhausted"),
        "public_scan_exhausted",
    )
    private_exhausted = _strict_bool(
        payload.get("private_scan_exhausted"),
        "private_scan_exhausted",
    )
    skipped: Counter = Counter()
    public_rows = _normalize_board_rows(
        payload.get("public_leaderboard"),
        "public",
        skipped,
    )
    private_rows = _normalize_board_rows(
        payload.get("private_leaderboard"),
        "private",
        skipped,
    )

    public_selected, public_scanned = _select_board(
        public_rows,
        private_rows,
        competition_id,
        competition_slug,
        skipped,
    )
    private_selected, private_scanned = _select_board(
        private_rows,
        public_rows,
        competition_id,
        competition_slug,
        skipped,
    )
    if len(public_selected) < SELECTION_SIZE and not public_exhausted:
        raise ManifestError(
            "Public scan found fewer than five qualifying writeups; continue "
            "the complete rank-prefix scan or set public_scan_exhausted=true "
            "only after the final leaderboard row was reviewed"
        )
    if len(private_selected) < SELECTION_SIZE and not private_exhausted:
        raise ManifestError(
            "Private scan found fewer than five qualifying writeups; continue "
            "the complete rank-prefix scan or set private_scan_exhausted=true "
            "only after the final leaderboard row was reviewed"
        )

    canonical: Dict[str, Candidate] = {}
    selection_boards: MutableMapping[str, List[str]] = {}
    for board, candidates in (
        ("Public", public_selected),
        ("Private", private_selected),
    ):
        for candidate in candidates:
            canonical.setdefault(candidate.team_id, candidate)
            selection_boards.setdefault(candidate.team_id, []).append(board)

    analyses = _analysis_rows(payload)
    ordered_candidates = sorted(
        canonical.values(),
        key=lambda item: (
            item.private_rank,
            item.public_rank,
            item.team_id,
        ),
    )
    records: List[Dict[str, Any]] = [competition]
    for candidate in ordered_candidates:
        records.append(_leaderboard_record(payload, competition, candidate))
        records.append(
            _solution_record(
                payload,
                competition,
                candidate,
                _analysis_for(candidate, analyses),
            )
        )

    report: Dict[str, Any] = {
        "ok": True,
        "competition_id": competition_id,
        "selection_policy": {
            "ranking_basis": "leaderboard rank only; votes are ignored",
            "per_board_target": SELECTION_SIZE,
            "official_writeup_rule": (
                "HTTPS Kaggle /competitions/<same-slug>/writeups/<id>"
            ),
            "cross_board_identity_required": True,
        },
        "input_counts": {
            "public_rows": len(payload.get("public_leaderboard") or []),
            "private_rows": len(payload.get("private_leaderboard") or []),
            "analyses": len(analyses),
        },
        "normalized_counts": {
            "public_rows": len(public_rows),
            "private_rows": len(private_rows),
        },
        "scanned_counts": {
            "public_rows": public_scanned,
            "private_rows": private_scanned,
        },
        "scan_proof": {
            "public_complete_prefix_through_rank": (
                public_rows[public_scanned - 1].own_rank if public_scanned else 0
            ),
            "private_complete_prefix_through_rank": (
                private_rows[private_scanned - 1].own_rank if private_scanned else 0
            ),
            "public_scan_exhausted": public_exhausted,
            "private_scan_exhausted": private_exhausted,
            "public_rank_source_url": _official_leaderboard_url(
                source_values["public"], competition_slug
            ),
            "private_rank_source_url": _official_leaderboard_url(
                source_values["private"], competition_slug
            ),
        },
        "selected_counts": {
            "public": len(public_selected),
            "private": len(private_selected),
            "canonical_solutions": len(ordered_candidates),
        },
        "shortfalls": {
            "public": max(0, SELECTION_SIZE - len(public_selected)),
            "private": max(0, SELECTION_SIZE - len(private_selected)),
        },
        "public_selection": _selection_report_rows(public_selected),
        "private_selection": _selection_report_rows(private_selected),
        "canonical_selection": [
            {
                "team_id": candidate.team_id,
                "team_name": candidate.team_name,
                "selected_via": selection_boards[candidate.team_id],
                "public_rank": candidate.public_rank,
                "private_rank": candidate.private_rank,
                "writeup_url": candidate.writeup_url,
            }
            for candidate in ordered_candidates
        ],
        "skipped": dict(sorted(skipped.items())),
        "ignored_vote_fields": _count_vote_fields(payload),
        "output_record_counts": {
            "competition": 1,
            "leaderboard_team": len(ordered_candidates),
            "solution": len(ordered_candidates),
            "total": len(records),
        },
    }
    return records, report


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="JSON file containing one completed competition export",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination reviewed JSONL file for catalog.py ingest-jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional destination for the same JSON audit report printed to stdout",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if input_path == output_path:
            raise ManifestError("--input and --output must be different files")
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records, report = prepare_manifest(payload)
        _write_jsonl_atomic(output_path, records)
        report["output_path"] = str(output_path)
        rendered_report = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        if args.report:
            report_path = args.report.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(rendered_report + "\n", encoding="utf-8")
        print(rendered_report)
        return 0
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
