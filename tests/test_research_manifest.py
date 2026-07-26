from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / ".agents" / "skills" / "kaggle-solution-retriever" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_core import Catalog  # noqa: E402
from prepare_research_manifest import (  # noqa: E402
    ManifestError,
    main,
    prepare_manifest,
)


COMPETITION_SLUG = "synthetic-ranking"


def writeup(team: str, *, competition: str = COMPETITION_SLUG) -> str:
    return f"https://www.kaggle.com/competitions/{competition}/writeups/{team}"


def competition_payload() -> dict:
    return {
        "competition": {
            "id": COMPETITION_SLUG,
            "slug": COMPETITION_SLUG,
            "title": "Synthetic Ranking",
            "description": "A completed synthetic fixture.",
            "end_date": "2025-01-01T00:00:00Z",
            "status": "completed",
            "tags": ["tabular", "classification"],
            "url": (f"https://www.kaggle.com/competitions/{COMPETITION_SLUG}"),
            "task_types": ["binary classification"],
            "metrics": ["log loss"],
        },
        "public_leaderboard": [],
        "private_leaderboard": [],
        "leaderboard_url": (
            f"https://www.kaggle.com/competitions/{COMPETITION_SLUG}/leaderboard"
        ),
        "ranks_verified_at": "2026-07-26T10:00:00Z",
        "public_scan_exhausted": True,
        "private_scan_exhausted": True,
    }


def board_row(
    team_number: int,
    public_rank: int,
    private_rank: int,
    *,
    url: Optional[str] = None,
    votes: int = 0,
) -> tuple[dict, dict]:
    team_id = f"team-{team_number}"
    common = {
        "team_id": team_id,
        "team_name": f"Team {team_number}",
        "member_handles": [f"user-{team_number}"],
        "writeup_url": writeup(team_id) if url is None else url,
        "votes": votes,
    }
    public = {
        **common,
        "public_rank": public_rank,
        "private_rank": private_rank,
        "public_score": f"0.{900 - public_rank:03d}",
    }
    private = {
        **common,
        "private_rank": private_rank,
        "public_rank": public_rank,
        "private_score": f"0.{900 - private_rank:03d}",
    }
    return public, private


class ManifestSelectionTests(unittest.TestCase):
    def test_scans_by_each_rank_skips_bad_writeups_and_ignores_votes(self):
        payload = competition_payload()
        pairs = [
            board_row(1, 1, 8, votes=0),
            board_row(2, 2, 7, url="", votes=100_000),
            board_row(3, 3, 6, votes=1),
            board_row(
                4,
                4,
                5,
                url=writeup("team-4", competition="other-competition"),
                votes=99_999,
            ),
            board_row(5, 5, 4, votes=2),
            board_row(6, 6, 3, votes=3),
            board_row(7, 7, 2, votes=4),
            board_row(8, 8, 1, votes=5),
        ]
        payload["public_leaderboard"] = [pair[0] for pair in reversed(pairs)]
        payload["private_leaderboard"] = [pair[1] for pair in pairs]

        records, report = prepare_manifest(payload)

        self.assertEqual(
            [row["public_rank"] for row in report["public_selection"]],
            [1, 3, 5, 6, 7],
        )
        self.assertEqual(
            [row["private_rank"] for row in report["private_selection"]],
            [1, 2, 3, 4, 6],
        )
        self.assertEqual(report["selected_counts"]["public"], 5)
        self.assertEqual(report["selected_counts"]["private"], 5)
        self.assertEqual(report["selected_counts"]["canonical_solutions"], 6)
        self.assertGreater(report["ignored_vote_fields"], 0)
        self.assertEqual(
            len([record for record in records if record["record_type"] == "solution"]),
            6,
        )
        self.assertTrue(all("votes" not in record for record in records))

    def test_fallback_identity_uses_exact_name_and_sorted_handles(self):
        payload = competition_payload()
        public = {
            "rank": 1,
            "team_name": "Exact Team",
            "member_handles": ["zeta", "alpha"],
            "writeup_url": writeup("exact-team"),
        }
        private = {
            "rank": 1,
            "team_name": "Exact Team",
            "members": [
                {"username": "alpha"},
                {"handle": "zeta"},
            ],
            "writeup_url": writeup("exact-team"),
        }
        payload["public_leaderboard"] = [public]
        payload["private_leaderboard"] = [private]

        first_records, first_report = prepare_manifest(payload)
        second_records, second_report = prepare_manifest(payload)

        team_id = first_report["canonical_selection"][0]["team_id"]
        self.assertTrue(team_id.startswith("team-"))
        self.assertEqual(
            team_id,
            second_report["canonical_selection"][0]["team_id"],
        )
        solution = next(
            record for record in first_records if record["record_type"] == "solution"
        )
        self.assertEqual(solution["team_id"], team_id)
        self.assertEqual(first_records, second_records)

    def test_team_id_is_preferred_when_names_change_between_boards(self):
        payload = competition_payload()
        payload["public_leaderboard"] = [
            {
                "rank": 1,
                "team_id": "real-kaggle-id",
                "team_name": "Old display name",
                "writeup_url": writeup("real-team"),
            }
        ]
        payload["private_leaderboard"] = [
            {
                "rank": 1,
                "team_id": "real-kaggle-id",
                "team_name": "New display name",
                "writeup_url": writeup("real-team"),
            }
        ]

        _, report = prepare_manifest(payload)

        self.assertEqual(
            report["canonical_selection"][0]["team_id"],
            "real-kaggle-id",
        )

    def test_missing_or_conflicting_cross_rank_blocks_lower_rank_substitution(self):
        payload = competition_payload()
        payload["public_leaderboard"] = [
            {
                "public_rank": 1,
                "private_rank": 9,
                "team_id": "conflict",
                "team_name": "Conflict",
                "writeup_url": writeup("conflict"),
            },
            {
                "public_rank": 2,
                "team_id": "public-only",
                "team_name": "Public only",
                "writeup_url": writeup("public-only"),
            },
        ]
        payload["private_leaderboard"] = [
            {
                "private_rank": 1,
                "public_rank": 1,
                "team_id": "conflict",
                "team_name": "Conflict",
                "writeup_url": writeup("conflict"),
            }
        ]

        with self.assertRaisesRegex(ManifestError, "do not substitute"):
            prepare_manifest(payload)

    def test_cross_rank_outside_opposite_prefix_does_not_change_top_five(self):
        payload = competition_payload()
        pairs = [
            board_row(1, 1, 6),
            board_row(2, 2, 1),
            board_row(3, 3, 2),
            board_row(4, 4, 3),
            board_row(5, 5, 4),
            board_row(6, 6, 5),
        ]
        payload["public_leaderboard"] = [pair[0] for pair in pairs]
        payload["private_leaderboard"] = [
            pair[1] for pair in pairs if pair[1]["private_rank"] <= 5
        ]

        _, report = prepare_manifest(payload)

        self.assertEqual(
            [item["public_rank"] for item in report["public_selection"]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [item["private_rank"] for item in report["private_selection"]],
            [1, 2, 3, 4, 5],
        )
        public_first = report["public_selection"][0]
        self.assertEqual(public_first["team_id"], "team-1")
        self.assertEqual(public_first["private_rank"], 6)
        self.assertEqual(
            report["scan_proof"]["private_complete_prefix_through_rank"],
            5,
        )

    def test_missing_cross_rank_blocks_manifest_without_backfilling(self):
        payload = competition_payload()
        pairs = [
            board_row(1, 1, 6),
            board_row(2, 2, 1),
            board_row(3, 3, 2),
            board_row(4, 4, 3),
            board_row(5, 5, 4),
            board_row(6, 6, 5),
        ]
        pairs[0][0].pop("private_rank")
        payload["public_leaderboard"] = [pair[0] for pair in pairs]
        payload["private_leaderboard"] = [
            pair[1] for pair in pairs if pair[1]["private_rank"] <= 5
        ]

        with self.assertRaisesRegex(
            ManifestError,
            "Public rank 1.*do not substitute",
        ):
            prepare_manifest(payload)

    def test_optional_analysis_is_copied_without_promoting_inference_to_fact(self):
        payload = competition_payload()
        public, private = board_row(1, 1, 1)
        payload["public_leaderboard"] = [public]
        payload["private_leaderboard"] = [private]
        payload["team_analyses"] = [
            {
                "team_id": "team-1",
                "core_idea": "A source-stated ensemble.",
                "extracted_facts": ["The writeup states five folds."],
                "analyst_inferences": ["The rank decline may indicate drift."],
                "repository_urls": ["https://github.com/example/solution"],
                "confidence": "high",
            }
        ]

        records, _ = prepare_manifest(payload)
        solution = next(
            record for record in records if record["record_type"] == "solution"
        )

        self.assertEqual(solution["core_idea"], "A source-stated ensemble.")
        self.assertEqual(
            solution["extracted_facts"],
            ["The writeup states five folds."],
        )
        self.assertEqual(
            solution["analyst_inferences"],
            ["The rank decline may indicate drift."],
        )
        self.assertEqual(
            solution["repository_urls"],
            ["https://github.com/example/solution"],
        )

    def test_output_records_are_accepted_by_catalog_ingest(self):
        payload = competition_payload()
        pairs = [board_row(index, index, 6 - index) for index in range(1, 6)]
        payload["public_leaderboard"] = [pair[0] for pair in pairs]
        payload["private_leaderboard"] = [pair[1] for pair in pairs]
        records, _ = prepare_manifest(payload)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "reviewed.jsonl"
            manifest_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                ),
                encoding="utf-8",
            )
            with Catalog(root / "catalog.sqlite3") as catalog:
                catalog.init()
                counts = catalog.ingest_jsonl(manifest_path)

        self.assertEqual(counts["competition"], 1)
        self.assertEqual(counts["leaderboard_team"], 5)
        self.assertEqual(counts["solution"], 5)

    def test_cli_writes_jsonl_and_report(self):
        payload = competition_payload()
        public, private = board_row(1, 1, 1)
        payload["public_leaderboard"] = [public]
        payload["private_leaderboard"] = [private]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            output_path = root / "output.jsonl"
            report_path = root / "report.json"
            input_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(
                report["selected_counts"]["canonical_solutions"],
                1,
            )

    def test_rejects_non_completed_competition(self):
        payload = competition_payload()
        payload["competition"]["status"] = "active"

        with self.assertRaisesRegex(ManifestError, "completed"):
            prepare_manifest(payload)

    def test_rejects_extra_writeup_path_segments_and_invalid_review_time(self):
        payload = competition_payload()
        public, private = board_row(
            1,
            1,
            1,
            url=writeup("team-1") + "/not-the-canonical-route",
        )
        payload["public_leaderboard"] = [public]
        payload["private_leaderboard"] = [private]

        records, report = prepare_manifest(payload)

        self.assertEqual(len(records), 1)
        self.assertEqual(report["selected_counts"]["canonical_solutions"], 0)

        payload["ranks_verified_at"] = "not-a-timestamp"
        with self.assertRaisesRegex(ManifestError, "ISO-8601"):
            prepare_manifest(payload)

    def test_requires_complete_rank_prefix_and_explicit_official_source(self):
        payload = competition_payload()
        pairs = [board_row(index, index + 19, index + 19) for index in range(1, 6)]
        payload["public_leaderboard"] = [pair[0] for pair in pairs]
        payload["private_leaderboard"] = [pair[1] for pair in pairs]

        with self.assertRaisesRegex(ManifestError, "every rank from 1"):
            prepare_manifest(payload)

        payload = competition_payload()
        public, private = board_row(1, 1, 1)
        payload["public_leaderboard"] = [public]
        payload["private_leaderboard"] = [private]
        payload["leaderboard_url"] = ""

        with self.assertRaisesRegex(ManifestError, "leaderboard_url is required"):
            prepare_manifest(payload)


if __name__ == "__main__":
    unittest.main()
