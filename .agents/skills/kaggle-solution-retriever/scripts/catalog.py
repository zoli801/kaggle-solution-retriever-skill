#!/usr/bin/env python3
"""Initialize, inspect, and populate the Kaggle solution catalog."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from kaggle_core import (
    Catalog,
    CatalogError,
    csv_header,
    csv_rows,
    find_column,
    locate_csv,
    normalize_tags,
    parse_rank,
    resolve_db_path,
    stable_id,
    utc_now,
)


def _date_is_completed(value: str) -> bool:
    if not value:
        return False
    candidate = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed < dt.datetime.now(dt.timezone.utc)
    except ValueError:
        try:
            parsed_date = dt.date.fromisoformat(candidate[:10])
            return parsed_date < dt.datetime.now(dt.timezone.utc).date()
        except ValueError:
            return False


def _load_competition_tags(meta_dir: Path) -> Dict[str, List[str]]:
    tags_path = locate_csv(meta_dir, ["Tags.csv"], required=False)
    links_path = locate_csv(
        meta_dir,
        ["CompetitionTags.csv", "CompetitionTagLinks.csv"],
        required=False,
    )
    if not tags_path or not links_path:
        return {}

    tag_header = csv_header(tags_path)
    tag_id_col = find_column(tag_header, ["Id", "TagId"], "tag id")
    tag_name_col = find_column(
        tag_header, ["Name", "Slug", "FullPath", "Title"], "tag name"
    )
    tag_names = {
        row[tag_id_col]: row[tag_name_col]
        for row in csv_rows(tags_path)
        if row.get(tag_id_col) and row.get(tag_name_col)
    }

    link_header = csv_header(links_path)
    comp_col = find_column(
        link_header,
        ["CompetitionId", "SourceCompetitionId"],
        "competition tag competition id",
    )
    link_tag_col = find_column(link_header, ["TagId"], "competition tag id")
    result: Dict[str, List[str]] = defaultdict(list)
    for row in csv_rows(links_path):
        name = tag_names.get(row.get(link_tag_col, ""))
        competition_id = row.get(comp_col, "")
        if name and competition_id:
            result[competition_id].append(name)
    return {key: normalize_tags(values) for key, values in result.items()}


def _find_code_path(code_dir: Optional[Path], version_id: str) -> str:
    if not code_dir:
        return ""
    for base in (code_dir, code_dir / "versions"):
        for name in (
            version_id,
            f"{version_id}.ipynb",
            f"{version_id}.py",
            f"{version_id}.r",
            f"{version_id}.jl",
            f"{version_id}.sql",
        ):
            candidate = base / name
            if candidate.is_file():
                return str(candidate.resolve())
    return ""


def _notebook_identity(
    username: str,
    version_row: Mapping[str, str],
    slug_col: Optional[str],
    url_col: Optional[str],
) -> Tuple[str, str]:
    raw_slug = (version_row.get(slug_col, "") if slug_col else "").strip()
    raw_url = (version_row.get(url_col, "") if url_col else "").strip()
    notebook_ref = ""
    notebook_url = ""

    if raw_url.startswith(("http://", "https://")):
        notebook_url = raw_url
        parts = [part for part in urlparse(raw_url).path.split("/") if part]
        if "code" in parts:
            index = parts.index("code")
            if len(parts) > index + 2:
                notebook_ref = f"{parts[index + 1]}/{parts[index + 2]}"
    elif "/" in raw_url:
        notebook_ref = raw_url.strip("/")
    if not notebook_ref and username and raw_slug:
        notebook_ref = f"{username}/{raw_slug.strip('/')}"
    if notebook_ref and not notebook_url:
        notebook_url = f"https://www.kaggle.com/code/{notebook_ref}"
    return notebook_ref, notebook_url


def import_competitions(catalog: Catalog, meta_dir: Path) -> Dict[str, Any]:
    """Seed the lightweight completed-competition catalog from Meta Kaggle."""
    meta_dir = meta_dir.resolve()
    if not meta_dir.is_dir():
        raise CatalogError(f"Meta Kaggle directory does not exist: {meta_dir}")
    competitions_path = locate_csv(meta_dir, ["Competitions.csv"])
    competition_tags = _load_competition_tags(meta_dir)
    header = csv_header(competitions_path)
    id_col = find_column(header, ["Id", "CompetitionId"], "competition id")
    slug_col = find_column(header, ["Slug"], "competition slug")
    title_col = find_column(header, ["Title", "Name"], "competition title")
    deadline_col = find_column(
        header,
        ["DeadlineDate", "ModelSubmissionDeadlineDate", "FinalDeadlineDate"],
        "competition deadline",
    )
    subtitle_col = find_column(
        header, ["Subtitle"], "competition subtitle", required=False
    )
    description_col = find_column(
        header, ["Description"], "competition description", required=False
    )
    metric_col = find_column(
        header,
        ["EvaluationAlgorithmAbbreviation", "EvaluationAlgorithmName"],
        "competition metric",
        required=False,
    )
    counts = {"competitions": 0, "completed_competitions": 0}
    with catalog.conn:
        for row in csv_rows(competitions_path):
            competition_id = row.get(id_col, "").strip()
            slug = row.get(slug_col, "").strip()
            title = row.get(title_col, "").strip()
            deadline = row.get(deadline_col, "").strip()
            if not competition_id or not slug or not title:
                continue
            completed = _date_is_completed(deadline)
            description_parts = [
                row.get(column, "").strip()
                for column in (subtitle_col, description_col, metric_col)
                if column and row.get(column, "").strip()
            ]
            metric = row.get(metric_col, "").strip() if metric_col else ""
            catalog.upsert_competition(
                {
                    "id": competition_id,
                    "slug": slug,
                    "title": title,
                    "description": " ".join(description_parts),
                    "end_date": deadline,
                    "status": "completed" if completed else "active",
                    "tags": competition_tags.get(competition_id, []),
                    "url": f"https://www.kaggle.com/competitions/{slug}",
                    "metrics": [metric] if metric else [],
                    "profile_source_url": (
                        "https://www.kaggle.com/datasets/kaggle/meta-kaggle"
                    ),
                    "profile_verified": False,
                }
            )
            counts["competitions"] += 1
            counts["completed_competitions"] += int(completed)
    counts["source"] = "Meta Kaggle Competitions.csv"
    return counts


def import_meta(
    catalog: Catalog,
    meta_dir: Path,
    code_dir: Optional[Path],
    private_top_rank: int,
    public_top_rank: int,
) -> Dict[str, Any]:
    """Stream Meta Kaggle joins without loading its large fact tables wholesale."""
    meta_dir = meta_dir.resolve()
    if not meta_dir.is_dir():
        raise CatalogError(f"Meta Kaggle directory does not exist: {meta_dir}")
    if code_dir:
        code_dir = code_dir.resolve()
        if not code_dir.is_dir():
            raise CatalogError(f"Meta Kaggle Code directory does not exist: {code_dir}")

    competitions_path = locate_csv(meta_dir, ["Competitions.csv"])
    teams_path = locate_csv(meta_dir, ["Teams.csv"])
    memberships_path = locate_csv(meta_dir, ["TeamMemberships.csv"])
    users_path = locate_csv(meta_dir, ["Users.csv"])
    kernels_path = locate_csv(meta_dir, ["Kernels.csv", "Scripts.csv"])
    versions_path = locate_csv(meta_dir, ["KernelVersions.csv", "ScriptVersions.csv"])
    sources_path = locate_csv(
        meta_dir,
        ["KernelVersionCompetitionSources.csv", "ScriptVersionCompetitionSources.csv"],
    )
    competition_tags = _load_competition_tags(meta_dir)

    comp_header = csv_header(competitions_path)
    comp_id_col = find_column(comp_header, ["Id", "CompetitionId"], "competition id")
    comp_slug_col = find_column(comp_header, ["Slug"], "competition slug")
    comp_title_col = find_column(comp_header, ["Title", "Name"], "competition title")
    comp_desc_col = find_column(
        comp_header,
        ["Subtitle", "Description", "EvaluationAlgorithmAbbreviation"],
        "competition description",
        required=False,
    )
    comp_deadline_col = find_column(
        comp_header,
        ["DeadlineDate", "ModelSubmissionDeadlineDate", "FinalDeadlineDate"],
        "competition deadline",
    )

    completed_competitions: Set[str] = set()
    competition_slugs: Dict[str, str] = {}
    competition_titles: Dict[str, str] = {}
    with catalog.conn:
        for row in csv_rows(competitions_path):
            competition_id = row.get(comp_id_col, "").strip()
            slug = row.get(comp_slug_col, "").strip()
            title = row.get(comp_title_col, "").strip()
            deadline = row.get(comp_deadline_col, "").strip()
            if not competition_id or not slug or not title:
                continue
            completed = _date_is_completed(deadline)
            if completed:
                completed_competitions.add(competition_id)
            competition_slugs[competition_id] = slug
            competition_titles[competition_id] = title
            catalog.upsert_competition(
                {
                    "id": competition_id,
                    "slug": slug,
                    "title": title,
                    "description": row.get(comp_desc_col, "") if comp_desc_col else "",
                    "end_date": deadline,
                    "status": "completed" if completed else "active",
                    "tags": competition_tags.get(competition_id, []),
                    "url": f"https://www.kaggle.com/competitions/{slug}",
                }
            )

    team_header = csv_header(teams_path)
    team_id_col = find_column(team_header, ["Id", "TeamId"], "team id")
    team_comp_col = find_column(team_header, ["CompetitionId"], "team competition id")
    private_rank_col = find_column(
        team_header,
        ["PrivateLeaderboardRank", "FinalLeaderboardRank", "PrivateRank", "FinalRank"],
        "private leaderboard rank",
    )
    public_rank_col = find_column(
        team_header,
        ["PublicLeaderboardRank", "PublicRank"],
        "public leaderboard rank",
        required=False,
    )
    team_name_col = find_column(
        team_header, ["TeamName", "Name"], "team name", required=False
    )
    eligible_teams: Dict[str, Dict[str, Any]] = {}
    for row in csv_rows(teams_path):
        competition_id = row.get(team_comp_col, "").strip()
        if competition_id not in completed_competitions:
            continue
        try:
            private_rank = parse_rank(row.get(private_rank_col))
            public_rank = (
                parse_rank(row.get(public_rank_col)) if public_rank_col else None
            )
        except CatalogError:
            continue
        private_top = private_rank is not None and private_rank <= private_top_rank
        public_faller = (
            public_rank is not None
            and public_rank <= public_top_rank
            and private_rank is not None
            and private_rank > private_top_rank
        )
        if not (private_top or public_faller):
            continue
        team_id = row.get(team_id_col, "").strip()
        if team_id:
            eligible_teams[team_id] = {
                "team_id": team_id,
                "competition_id": competition_id,
                "private_rank": private_rank,
                "public_rank": public_rank,
                "leaderboard_group": (
                    "private-top" if private_top else "public-top-private-faller"
                ),
                "team_name": row.get(team_name_col, "").strip()
                if team_name_col
                else "",
            }

    membership_header = csv_header(memberships_path)
    membership_team_col = find_column(
        membership_header, ["TeamId"], "membership team id"
    )
    membership_user_col = find_column(
        membership_header, ["UserId"], "membership user id"
    )
    user_teams: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in csv_rows(memberships_path):
        team = eligible_teams.get(row.get(membership_team_col, "").strip())
        user_id = row.get(membership_user_col, "").strip()
        if team and user_id:
            user_teams[user_id].append(team)

    user_header = csv_header(users_path)
    user_id_col = find_column(user_header, ["Id", "UserId"], "user id")
    username_col = find_column(
        user_header, ["UserName", "Username", "Slug"], "Kaggle username"
    )
    display_name_col = find_column(
        user_header,
        ["DisplayName", "Display Name", "Name"],
        "user display name",
        required=False,
    )
    usernames: Dict[str, str] = {}
    display_names: Dict[str, str] = {}
    target_users = set(user_teams)
    for row in csv_rows(users_path):
        user_id = row.get(user_id_col, "").strip()
        if user_id in target_users:
            usernames[user_id] = row.get(username_col, "").strip()
            if display_name_col and row.get(display_name_col, "").strip():
                display_names[user_id] = row.get(display_name_col, "").strip()

    stored_members = 0
    with catalog.conn:
        for user_id, teams in user_teams.items():
            username = usernames.get(user_id, "")
            if not username:
                continue
            for team in teams:
                competition_id = team["competition_id"]
                slug = competition_slugs[competition_id]
                leaderboard_url = (
                    f"https://www.kaggle.com/competitions/{slug}/leaderboard"
                )
                catalog.upsert_leaderboard_member(
                    {
                        "competition_id": competition_id,
                        "team_id": team["team_id"],
                        "user_id": user_id,
                        "username": username,
                        "aliases": [display_names[user_id]]
                        if user_id in display_names
                        else [],
                        "team_name": team["team_name"],
                        "private_rank": team["private_rank"],
                        "public_rank": team["public_rank"],
                        "private_rank_verified": True,
                        "public_rank_verified": team["public_rank"] is not None,
                        "private_rank_source_url": leaderboard_url,
                        "public_rank_source_url": (
                            leaderboard_url if team["public_rank"] is not None else ""
                        ),
                        "ranks_verified_at": utc_now(),
                    }
                )
                stored_members += 1

    kernel_header = csv_header(kernels_path)
    kernel_id_col = find_column(
        kernel_header, ["Id", "KernelId", "ScriptId"], "kernel id"
    )
    kernel_author_col = find_column(
        kernel_header,
        ["AuthorUserId", "OwnerUserId", "UserId"],
        "kernel author user id",
    )
    current_version_col = find_column(
        kernel_header,
        ["CurrentKernelVersionId", "CurrentScriptVersionId", "CurrentVersionId"],
        "current kernel version id",
    )
    versions_by_id: Dict[str, Dict[str, Any]] = {}
    for row in csv_rows(kernels_path):
        author_id = row.get(kernel_author_col, "").strip()
        if author_id not in target_users:
            continue
        version_id = row.get(current_version_col, "").strip()
        if version_id:
            versions_by_id[version_id] = {
                "kernel_id": row.get(kernel_id_col, "").strip(),
                "author_id": author_id,
            }

    source_header = csv_header(sources_path)
    source_version_col = find_column(
        source_header,
        ["KernelVersionId", "ScriptVersionId", "VersionId"],
        "competition source version id",
    )
    source_comp_col = find_column(
        source_header,
        ["CompetitionId", "SourceCompetitionId"],
        "competition source competition id",
    )
    linked_versions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in csv_rows(sources_path):
        version_id = row.get(source_version_col, "").strip()
        kernel = versions_by_id.get(version_id)
        if not kernel:
            continue
        competition_id = row.get(source_comp_col, "").strip()
        for team in user_teams[kernel["author_id"]]:
            if team["competition_id"] == competition_id:
                linked_versions[version_id].append(team)

    version_header = csv_header(versions_path)
    version_id_col = find_column(
        version_header,
        ["Id", "KernelVersionId", "ScriptVersionId"],
        "kernel version id",
    )
    version_title_col = find_column(
        version_header, ["Title", "Name"], "kernel version title"
    )
    version_slug_col = find_column(
        version_header,
        ["Slug", "ScriptSlug", "KernelSlug"],
        "kernel slug",
        required=False,
    )
    version_url_col = find_column(
        version_header,
        ["ScriptUrl", "KernelUrl", "Url", "Ref"],
        "kernel URL/ref",
        required=False,
    )
    imported = 0
    with catalog.conn:
        for row in csv_rows(versions_path):
            version_id = row.get(version_id_col, "").strip()
            teams = linked_versions.get(version_id)
            if not teams:
                continue
            kernel = versions_by_id[version_id]
            author_id = kernel["author_id"]
            username = usernames.get(author_id, "")
            title = (
                row.get(version_title_col, "").strip() or f"Kernel version {version_id}"
            )
            notebook_ref, notebook_url = _notebook_identity(
                username, row, version_slug_col, version_url_col
            )
            local_path = _find_code_path(code_dir, version_id)
            for team in teams:
                competition_id = team["competition_id"]
                slug = competition_slugs[competition_id]
                private_rank = team["private_rank"]
                public_rank = team["public_rank"]
                leaderboard_url = (
                    f"https://www.kaggle.com/competitions/{slug}/leaderboard"
                )
                if team["leaderboard_group"] == "private-top":
                    rank_summary = f"private rank {private_rank}"
                    if public_rank is not None:
                        rank_summary += f", public rank {public_rank}"
                else:
                    drop = private_rank - public_rank
                    rank_summary = (
                        f"public rank {public_rank}, private rank {private_rank}, "
                        f"drop {drop} places"
                    )
                solution_id = "meta-" + stable_id(competition_id, version_id)
                catalog.upsert_solution(
                    {
                        "id": solution_id,
                        "competition_id": competition_id,
                        "title": title,
                        "author": username,
                        "team_name": team["team_name"],
                        "private_rank": private_rank,
                        "public_rank": public_rank,
                        "private_rank_verified": True,
                        "public_rank_verified": public_rank is not None,
                        "solution_verified": False,
                        "private_rank_source_url": leaderboard_url,
                        "public_rank_source_url": leaderboard_url
                        if public_rank is not None
                        else "",
                        "ranks_verified_at": utc_now(),
                        "verification_note": (
                            f"Meta Kaggle links the author to a team with {rank_summary} and "
                            "the current public kernel to this competition. The artifact-to-"
                            "solution claim is not independently verified."
                        ),
                        "public": True,
                        "source_kind": "meta-kaggle-code" if local_path else "notebook",
                        "notebook_ref": notebook_ref,
                        "notebook_url": notebook_url,
                        "local_path": local_path,
                        "summary": (
                            f"Public competition notebook by a team member with {rank_summary}. "
                            "Treat as a discovery candidate, not automatically as the final solution."
                        ),
                        "tags": competition_tags.get(competition_id, []),
                        "license": (
                            "Apache-2.0 (Meta Kaggle Code dataset)"
                            if local_path
                            else "unknown/source-specific"
                        ),
                        "provenance_url": notebook_url
                        or "https://www.kaggle.com/datasets/kaggle/meta-kaggle",
                    }
                )
                imported += 1

    return {
        "completed_competitions": len(completed_competitions),
        "eligible_teams": len(eligible_teams),
        "private_top_teams": sum(
            team["leaderboard_group"] == "private-top"
            for team in eligible_teams.values()
        ),
        "public_top_private_fallers": sum(
            team["leaderboard_group"] == "public-top-private-faller"
            for team in eligible_teams.values()
        ),
        "eligible_team_users": len(target_users),
        "stored_leaderboard_members": stored_members,
        "candidate_current_versions": len(versions_by_id),
        "competition_linked_versions": len(linked_versions),
        "imported_artifacts": imported,
        "verification": "public/private leaderboard ranks verified; solution artifact not verified",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite catalog path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create or migrate the catalog")
    subparsers.add_parser("status", help="Print catalog counts and capabilities")
    subparsers.add_parser("rebuild-search", help="Rebuild the SQLite FTS index")

    ingest = subparsers.add_parser("ingest-jsonl", help="Import reviewed JSONL records")
    ingest.add_argument("--input", required=True, type=Path)

    competition_import = subparsers.add_parser(
        "import-competitions",
        help="Seed competition metadata from Meta Kaggle without importing notebooks",
    )
    competition_import.add_argument("--meta-dir", required=True, type=Path)

    meta = subparsers.add_parser(
        "import-meta", help="Import candidates from Meta Kaggle CSVs"
    )
    meta.add_argument("--meta-dir", required=True, type=Path)
    meta.add_argument("--code-dir", type=Path)
    meta.add_argument("--private-top-rank", type=int, default=10)
    meta.add_argument("--public-top-rank", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        getattr(args, "private_top_rank", 1) < 1
        or getattr(args, "public_top_rank", 1) < 1
    ):
        print("error: leaderboard cutoffs must be positive", file=sys.stderr)
        return 2
    try:
        with Catalog(resolve_db_path(args.db)) as catalog:
            catalog.init()
            if args.command == "init":
                result: Any = catalog.status()
            elif args.command == "status":
                result = catalog.status()
            elif args.command == "rebuild-search":
                result = {"indexed_solutions": catalog.rebuild_search()}
            elif args.command == "ingest-jsonl":
                result = catalog.ingest_jsonl(args.input.resolve())
                result["status"] = catalog.status()
            elif args.command == "import-competitions":
                result = import_competitions(catalog, args.meta_dir)
                result["status"] = catalog.status()
            elif args.command == "import-meta":
                result = import_meta(
                    catalog,
                    args.meta_dir,
                    args.code_dir,
                    args.private_top_rank,
                    args.public_top_rank,
                )
                result["status"] = catalog.status()
            else:
                raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CatalogError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
