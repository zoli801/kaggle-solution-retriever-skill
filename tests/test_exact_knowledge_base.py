from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / ".agents" / "skills" / "kaggle-solution-retriever" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_core import Catalog, CatalogError, SCHEMA_VERSION  # noqa: E402
from knowledge_base import (  # noqa: E402
    RELEVANCE_WEIGHTS,
    analyze_task,
    build_knowledge_base,
    final_report,
    render_context_summary,
    render_knowledge_base,
    score_competition,
    select_relevant_competitions,
    select_writeups,
)


RICH_PROMPT = """
Build a binary classification model for a large tabular retail dataset with
millions of rows. The target is 0/1 and the evaluation metric is F1. Rows are
grouped by user, with repeated entities and timestamped events plus severe class imbalance.
Use GroupKFold so a user never crosses train and validation, and prevent entity
leakage and future temporal leakage. Start with CatBoost/GBDT and aggregation
features. We have limited compute with a single GPU, and external data is forbidden.
""".strip()

RELEVANT_PROFILE = {
    "task_types": ["binary classification"],
    "target_types": ["binary"],
    "modalities": ["tabular"],
    "metrics": ["F1"],
    "dataset_structure": [
        "grouped entities",
        "users",
        "repeated entities",
        "temporal order",
        "class imbalance",
        "large dataset",
    ],
    "validation_structure": ["GroupKFold"],
    "leakage_risks": ["entity leakage", "temporal leakage"],
    "transferable_methods": ["GBDT"],
    "feature_methods": ["aggregation features"],
    "domains": ["retail"],
    "compute_profiles": ["single GPU", "limited compute"],
    "constraints": ["external data restricted"],
}


class TemporaryCatalog:
    def __enter__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.catalog = Catalog(self.root / "catalog.sqlite3")
        self.catalog.init()
        return self

    def __exit__(self, *_):
        self.catalog.close()
        self.tempdir.cleanup()


def add_competition(
    catalog: Catalog,
    competition_id: str,
    *,
    status: str = "completed",
    profile=None,
    title: str = "",
) -> None:
    profile = dict(RELEVANT_PROFILE if profile is None else profile)
    catalog.upsert_competition(
        {
            "record_type": "competition",
            "id": competition_id,
            "slug": competition_id,
            "title": title or competition_id.replace("-", " ").title(),
            "description": "Synthetic competition metadata for exact-policy tests.",
            "end_date": "2024-01-01T00:00:00Z",
            "status": status,
            "tags": profile.get("modalities", []) + profile.get("task_types", []),
            "url": f"https://www.kaggle.com/competitions/{competition_id}",
            **profile,
            "profile_source_url": (
                f"https://www.kaggle.com/competitions/{competition_id}/overview"
            ),
            "profile_verified": True,
            "profile_updated_at": "2026-01-01T00:00:00Z",
        }
    )


def add_leaderboard_team(
    catalog: Catalog,
    competition_id: str,
    team_id: str,
    public_rank: int,
    private_rank: int,
    *,
    team_name: str = "",
    public_verified: bool = True,
    private_verified: bool = True,
) -> None:
    catalog.upsert_leaderboard_team(
        {
            "record_type": "leaderboard_team",
            "competition_id": competition_id,
            "team_id": team_id,
            "team_name": team_name or team_id,
            "public_rank": public_rank,
            "private_rank": private_rank,
            "public_score": f"0.{9000 - public_rank:04d}",
            "private_score": f"0.{9000 - private_rank:04d}",
            "public_rank_verified": public_verified,
            "private_rank_verified": private_verified,
            "public_rank_source_url": (
                f"https://www.kaggle.com/competitions/{competition_id}"
                "/leaderboard?view=public"
            ),
            "private_rank_source_url": (
                f"https://www.kaggle.com/competitions/{competition_id}/leaderboard"
            ),
            "ranks_verified_at": "2026-01-01T00:00:00Z",
        }
    )


def add_writeup(
    catalog: Catalog,
    competition_id: str,
    team_id: str,
    public_rank: int,
    private_rank: int,
    *,
    team_name: str = "",
    artifact_suffix: str = "primary",
    official_url: bool = True,
    public_verified: bool = True,
    private_verified: bool = True,
    writeup_verified: bool = True,
    long_core_idea: str = "",
) -> str:
    team_name = team_name or team_id
    writeup_url = (
        f"https://www.kaggle.com/competitions/{competition_id}/writeups/"
        f"{team_id}-{artifact_suffix}"
        if official_url
        else (
            f"https://www.kaggle.com/competitions/{competition_id}/discussion/"
            f"{team_id}-{artifact_suffix}"
        )
    )
    solution_id = f"{competition_id}-{team_id}-{artifact_suffix}"
    catalog.upsert_solution(
        {
            "record_type": "solution",
            "id": solution_id,
            "competition_id": competition_id,
            "team_id": team_id,
            "team_name": team_name,
            "title": f"{team_name} solution",
            "writeup_title": f"{team_name} official solution writeup",
            "author": f"author-{team_id}",
            "publication_author": f"author-{team_id}",
            "public_rank": public_rank,
            "private_rank": private_rank,
            "public_score": f"0.{9000 - public_rank:04d}",
            "private_score": f"0.{9000 - private_rank:04d}",
            "public_rank_verified": public_verified,
            "private_rank_verified": private_verified,
            "public_rank_source_url": (
                f"https://www.kaggle.com/competitions/{competition_id}"
                "/leaderboard?view=public"
            ),
            "private_rank_source_url": (
                f"https://www.kaggle.com/competitions/{competition_id}/leaderboard"
            ),
            "ranks_verified_at": "2026-01-01T00:00:00Z",
            "solution_verified": True,
            "writeup_verified": writeup_verified,
            "writeup_verified_at": "2026-01-01T00:00:00Z",
            "public": True,
            "source_kind": "writeup",
            "writeup_url": writeup_url,
            "summary": "Leakage-safe grouped validation with GBDT ensembling.",
            "core_idea": (
                long_core_idea
                or "Blend diverse GBDT models trained with leakage-safe grouped folds."
            ),
            "validation_strategy": "GroupKFold by user with out-of-fold predictions.",
            "preprocessing": "Fold-local missing-value handling.",
            "feature_engineering": "Leakage-safe user aggregation features.",
            "models": "CatBoost and LightGBM.",
            "training_procedure": "Seeded fold training with early stopping.",
            "ensembling": "Out-of-fold weighted averaging.",
            "post_processing": "Threshold selected only from out-of-fold F1.",
            "leakage_prevention": "All aggregations are fit inside each fold.",
            "failed_approaches": "Random row split inflated local validation.",
            "compute_requirements": "One GPU.",
            "robustness_notes": "Grouped validation tracked the private leaderboard.",
            "transferable_ideas": "Group-aware validation and OOF blending.",
            "application_risks": "Entity definitions must match the target dataset.",
            "techniques": ["GBDT", "OOF blending"],
            "validation_tags": ["GroupKFold"],
            "model_tags": ["CatBoost", "LightGBM"],
            "feature_tags": ["aggregation features"],
            "ensemble_tags": ["weighted averaging"],
            "extracted_facts": ["The writeup states that GroupKFold was used."],
            "analyst_inferences": [
                "The small rank change suggests validation transferred well."
            ],
            "confidence": "high",
            "code_urls": [
                f"https://www.kaggle.com/code/synthetic/{competition_id}-{team_id}"
            ],
            "source_accessed_at": "2026-01-01T00:00:00Z",
            "license": "source-specific",
            "provenance_url": writeup_url,
        }
    )
    return solution_id


def competition_row(catalog: Catalog, competition_id: str):
    return dict(
        catalog.conn.execute(
            "SELECT * FROM competitions WHERE id=?", (competition_id,)
        ).fetchone()
    )


def populate_seven_overlapping_writeups(
    catalog: Catalog,
    competition_id: str,
    *,
    long_core_idea: str = "",
) -> None:
    # Public top five documented teams: A, B, C, D, E -> ranks 1, 3, 7, 11, 14.
    # Private top five documented teams: F, C, D, G, E -> ranks 1, 2, 7, 11, 14.
    ranks = {
        "A": (1, 30),
        "B": (3, 25),
        "C": (7, 2),
        "D": (11, 7),
        "E": (14, 14),
        "F": (20, 1),
        "G": (21, 11),
    }
    for label, (public_rank, private_rank) in ranks.items():
        add_writeup(
            catalog,
            competition_id,
            f"{competition_id}-team-{label.lower()}",
            public_rank,
            private_rank,
            team_name=f"Team {label}",
            long_core_idea=long_core_idea if label == "G" else "",
        )

    # These ranked teams have no public Solution Writeup and must be skipped.
    for rank in (2, 4, 5, 6, 8, 9, 10, 12, 13):
        add_leaderboard_team(
            catalog,
            competition_id,
            f"{competition_id}-undocumented-public-{rank}",
            rank,
            40 + rank,
        )
    for rank in (3, 4, 5, 6, 8, 9, 10, 12, 13):
        add_leaderboard_team(
            catalog,
            competition_id,
            f"{competition_id}-undocumented-private-{rank}",
            40 + rank,
            rank,
        )


class ExactKnowledgeBaseTests(unittest.TestCase):
    def test_schema_v3_and_rich_task_profile(self):
        profile = analyze_task(RICH_PROMPT)

        self.assertEqual(SCHEMA_VERSION, 3)
        self.assertIn("binary classification", profile["task_types"])
        self.assertIn("binary", profile["target_types"])
        self.assertIn("tabular", profile["modalities"])
        self.assertIn("F1", profile["metrics"])
        for structure in (
            "grouped entities",
            "users",
            "repeated entities",
            "temporal order",
            "class imbalance",
            "large dataset",
        ):
            self.assertIn(structure, profile["dataset_structure"])
        self.assertIn("GroupKFold", profile["validation_structure"])
        self.assertIn("entity leakage", profile["leakage_risks"])
        self.assertIn("temporal leakage", profile["leakage_risks"])
        self.assertIn("single GPU", profile["compute_profiles"])
        self.assertIn("limited compute", profile["compute_profiles"])
        self.assertIn("external data restricted", profile["constraints"])
        self.assertTrue(
            any(item.startswith("metric: F1") for item in profile["search_criteria"])
        )

        with TemporaryCatalog() as fixture:
            tables = {
                row[0]
                for row in fixture.catalog.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertTrue(
                {
                    "leaderboard_teams",
                    "competition_profiles",
                    "solution_details",
                }.issubset(tables)
            )

    def test_structured_relevance_prefers_metric_validation_and_structure(self):
        task_profile = analyze_task(RICH_PROMPT)
        exact = {
            "id": "exact",
            "title": "Grouped F1 Tabular",
            "description": "",
            "tags": [],
            **RELEVANT_PROFILE,
        }
        broad = {
            "id": "broad",
            "title": "Generic Binary Classification",
            "description": "",
            "tags": [],
            "task_types": ["binary classification"],
            "target_types": ["binary"],
            "modalities": ["tabular"],
            "metrics": ["ROC AUC"],
            "dataset_structure": ["independent rows"],
            "validation_structure": ["StratifiedKFold"],
            "leakage_risks": [],
            "transferable_methods": ["GBDT"],
            "feature_methods": [],
            "domains": ["finance"],
            "compute_profiles": ["CPU-only"],
            "constraints_json": [],
        }

        exact_score = score_competition(task_profile, exact)
        broad_score = score_competition(task_profile, broad)

        self.assertAlmostEqual(sum(RELEVANCE_WEIGHTS.values()), 1.0)
        self.assertGreater(exact_score["score"], broad_score["score"] + 0.20)
        self.assertGreater(
            exact_score["components"]["metric"],
            broad_score["components"]["metric"],
        )
        self.assertGreater(
            exact_score["components"]["validation"],
            broad_score["components"]["validation"],
        )
        self.assertTrue(exact_score["hard_relevant"])

    def test_completed_only_threshold_relaxation_and_exact_3_to_10_limits(self):
        task_profile = analyze_task(RICH_PROMPT)
        partial_profile = {
            **RELEVANT_PROFILE,
            "feature_methods": ["target encoding"],
            "domains": ["finance"],
            "compute_profiles": ["CPU-only"],
        }
        irrelevant_profile = {
            "task_types": ["segmentation"],
            "target_types": ["mask"],
            "modalities": ["image"],
            "metrics": ["Dice"],
            "dataset_structure": ["independent images"],
            "validation_structure": ["KFold"],
            "leakage_risks": [],
            "transferable_methods": ["vision transformer"],
            "feature_methods": ["image augmentation"],
            "domains": ["remote sensing"],
            "compute_profiles": ["multi-GPU"],
            "constraints": [],
        }
        with TemporaryCatalog() as fixture:
            with fixture.catalog.conn:
                for index in range(12):
                    competition_id = f"relevant-{index:02d}"
                    add_competition(
                        fixture.catalog,
                        competition_id,
                        profile=partial_profile,
                    )
                    add_writeup(
                        fixture.catalog,
                        competition_id,
                        f"team-{index:02d}",
                        1,
                        1,
                    )
                add_competition(
                    fixture.catalog,
                    "active-exact",
                    status="active",
                    profile=RELEVANT_PROFILE,
                )
                add_writeup(fixture.catalog, "active-exact", "active-team", 1, 1)
                add_competition(
                    fixture.catalog,
                    "irrelevant-completed",
                    profile=irrelevant_profile,
                )
                add_writeup(
                    fixture.catalog,
                    "irrelevant-completed",
                    "irrelevant-team",
                    1,
                    1,
                )

            selected, metadata = select_relevant_competitions(
                fixture.catalog,
                task_profile,
                min_competitions=3,
                max_competitions=10,
                initial_threshold=0.99,
                minimum_threshold=0.40,
                threshold_step=0.02,
            )

            self.assertEqual(len(selected), 10)
            self.assertTrue(metadata["minimum_met"])
            self.assertLess(metadata["threshold_used"], 0.99)
            self.assertTrue(all(item["status"] == "completed" for item in selected))
            self.assertTrue(
                all(item["id"].startswith("relevant-") for item in selected)
            )
            self.assertNotIn("active-exact", {item["id"] for item in selected})
            self.assertNotIn("irrelevant-completed", {item["id"] for item in selected})

            with self.assertRaises(CatalogError):
                select_relevant_competitions(
                    fixture.catalog,
                    task_profile,
                    min_competitions=2,
                    max_competitions=10,
                )
            with self.assertRaises(CatalogError):
                select_relevant_competitions(
                    fixture.catalog,
                    task_profile,
                    min_competitions=3,
                    max_competitions=11,
                )

    def test_independent_top_five_beyond_rank_ten_and_canonical_overlap(self):
        with TemporaryCatalog() as fixture:
            with fixture.catalog.conn:
                add_competition(fixture.catalog, "selection-test")
                populate_seven_overlapping_writeups(fixture.catalog, "selection-test")
                add_writeup(
                    fixture.catalog,
                    "selection-test",
                    "unverified-cross-rank",
                    2,
                    50,
                    team_name="Unverified Cross Rank",
                    public_verified=False,
                )
                add_writeup(
                    fixture.catalog,
                    "selection-test",
                    "discussion-only",
                    2,
                    3,
                    team_name="Discussion Only",
                    official_url=False,
                    writeup_verified=False,
                )

            selected = select_writeups(
                fixture.catalog,
                competition_row(fixture.catalog, "selection-test"),
            )

            self.assertEqual(
                [item["public_rank"] for item in selected["public_selection"]],
                [1, 3, 7, 11, 14],
            )
            self.assertEqual(
                [item["private_rank"] for item in selected["private_selection"]],
                [1, 2, 7, 11, 14],
            )
            self.assertEqual(
                [
                    item["public_selected_position"]
                    for item in selected["public_selection"]
                ],
                [1, 2, 3, 4, 5],
            )
            self.assertEqual(
                [
                    item["private_selected_position"]
                    for item in selected["private_selection"]
                ],
                [1, 2, 3, 4, 5],
            )
            self.assertGreater(selected["public_scan_through_rank"], 10)
            self.assertGreater(selected["private_scan_through_rank"], 10)

            canonical = selected["canonical_solutions"]
            self.assertEqual(len(canonical), 7)
            membership_counts = {
                label: sum(item["selected_through"] == label for item in canonical)
                for label in ("Public", "Private", "Both")
            }
            self.assertEqual(
                membership_counts,
                {"Public": 2, "Private": 2, "Both": 3},
            )
            all_team_ids = {item["team_id"] for item in canonical}
            self.assertNotIn("unverified-cross-rank", all_team_ids)
            self.assertNotIn("discussion-only", all_team_ids)
            self.assertTrue(
                all(
                    urlparse(item["writeup_url"]).path.startswith(
                        "/competitions/selection-test/writeups/"
                    )
                    for item in canonical
                )
            )
            self.assertTrue(
                any(
                    "mandatory Public/Private cross-rank" in message
                    for message in selected["missing_information"]
                )
            )
            self.assertTrue(
                any(
                    "non-official or malformed" in message
                    for message in selected["missing_information"]
                )
            )

            by_name = {item["team_name"]: item for item in canonical}
            self.assertEqual(by_name["Team A"]["rank_change"]["direction"], "declined")
            self.assertEqual(by_name["Team A"]["rank_change"]["absolute_delta"], 29)
            self.assertEqual(by_name["Team F"]["rank_change"]["direction"], "improved")
            self.assertEqual(by_name["Team F"]["rank_change"]["absolute_delta"], 19)
            self.assertEqual(by_name["Team E"]["rank_change"]["direction"], "stable")
            self.assertEqual(by_name["Team E"]["rank_change"]["absolute_delta"], 0)

    def test_verified_sources_metadata_refresh_and_rank_conflicts_are_strict(self):
        with TemporaryCatalog() as fixture:
            with fixture.catalog.conn:
                add_competition(fixture.catalog, "strict-evidence")
                add_leaderboard_team(
                    fixture.catalog,
                    "strict-evidence",
                    "stable-team",
                    4,
                    27,
                )

            fixture.catalog.upsert_competition(
                {
                    "id": "strict-evidence",
                    "slug": "strict-evidence",
                    "title": "Strict Evidence",
                    "status": "completed",
                    "url": ("https://www.kaggle.com/competitions/strict-evidence"),
                }
            )
            preserved = competition_row(fixture.catalog, "strict-evidence")
            self.assertEqual(
                preserved["description"],
                "Synthetic competition metadata for exact-policy tests.",
            )
            self.assertNotEqual(preserved["tags"], "[]")
            self.assertEqual(preserved["end_date"], "2024-01-01T00:00:00Z")

            with self.assertRaisesRegex(CatalogError, "official HTTPS Kaggle"):
                fixture.catalog.upsert_leaderboard_team(
                    {
                        "competition_id": "strict-evidence",
                        "team_id": "bad-source",
                        "team_name": "Bad Source",
                        "public_rank": 1,
                        "private_rank": 1,
                        "public_rank_verified": True,
                        "private_rank_verified": True,
                        "public_rank_source_url": "https://example.com/leaderboard",
                        "private_rank_source_url": "https://example.com/leaderboard",
                    }
                )

            with self.assertRaisesRegex(CatalogError, "Conflicting verified ranks"):
                add_leaderboard_team(
                    fixture.catalog,
                    "strict-evidence",
                    "stable-team",
                    5,
                    27,
                )
            stable = fixture.catalog.conn.execute(
                """
                SELECT public_rank, private_rank
                FROM leaderboard_teams
                WHERE competition_id='strict-evidence' AND team_id='stable-team'
                """
            ).fetchone()
            self.assertEqual((stable["public_rank"], stable["private_rank"]), (4, 27))

    def test_required_markdown_source_dedupe_full_file_and_truthful_report(self):
        sentinel = "END_OF_COMPLETE_KNOWLEDGE_BASE_SENTINEL"
        # Keep the sentinel inside the renderer's documented per-field limit while
        # making the full document much larger than the bounded context summary.
        long_core_idea = ("transferable-evidence " * 600) + sentinel
        with TemporaryCatalog() as fixture:
            with fixture.catalog.conn:
                for competition_id in (
                    "knowledge-one",
                    "knowledge-two",
                    "knowledge-three",
                ):
                    add_competition(fixture.catalog, competition_id)
                    populate_seven_overlapping_writeups(
                        fixture.catalog,
                        competition_id,
                        long_core_idea=(
                            long_core_idea
                            if competition_id == "knowledge-three"
                            else ""
                        ),
                    )

            result = build_knowledge_base(
                fixture.catalog,
                RICH_PROMPT,
                profile_override=RELEVANT_PROFILE,
                min_competitions=3,
                max_competitions=3,
                initial_threshold=0.0,
                minimum_threshold=0.0,
            )
            markdown = render_knowledge_base(result)

            required_headings = (
                "# Kaggle Knowledge Base for the Current Task",
                "## Current Task Profile",
                "## Competition Selection Summary",
                "### Public Leaderboard Solution Selection",
                "### Private Leaderboard Solution Selection",
                "### Canonical Solution Analyses",
                "### Competition-Level Lessons",
                "## Cross-Competition Synthesis",
                "### Repeated High-Value Techniques",
                "### Validation Recommendations",
                "### Model Recommendations",
                "### Feature-Engineering Recommendations",
                "### Ensembling Recommendations",
                "### Public-to-Private Robustness",
                "### Proposed Experimental Plan",
                "## Source Index",
                "## Verification Summary",
            )
            for heading in required_headings:
                self.assertIn(heading, markdown)
            self.assertIn(
                "| Selected Position | Team | Public Rank | Private Rank | "
                "Public Score | Private Score | Solution Writeup |",
                markdown,
            )
            self.assertIn(
                "| Selected Position | Team | Private Rank | Public Rank | "
                "Private Score | Public Score | Solution Writeup |",
                markdown,
            )
            self.assertGreater(len(markdown), 14_000)
            self.assertIn(sentinel, markdown)
            self.assertNotIn("Context summary truncated", markdown)

            source_section = markdown.split("## Source Index", 1)[1].split(
                "## Missing or Inaccessible Information", 1
            )[0]
            source_urls = re.findall(r"\]\(<(https://[^>]+)>\)", source_section)
            self.assertTrue(source_urls)
            self.assertEqual(len(source_urls), len(set(source_urls)))
            self.assertTrue(
                all(
                    urlparse(url).netloc in {"kaggle.com", "www.kaggle.com"}
                    for url in source_urls
                )
            )

            knowledge_base_path = fixture.root / "knowledge-base.md"
            knowledge_base_path.write_text(markdown, encoding="utf-8")
            context = render_context_summary(
                result,
                knowledge_base_path,
                max_chars=800,
            )
            self.assertLessEqual(len(context), 800)
            self.assertIn("Context summary truncated", context)
            self.assertIn(sentinel, knowledge_base_path.read_text(encoding="utf-8"))

            report = final_report(result, knowledge_base_path)
            self.assertEqual(
                report["knowledge_base_path"],
                str(knowledge_base_path.resolve()),
            )
            self.assertTrue(report["all_leaderboard_ranks_and_links_verified"])
            self.assertEqual(len(report["selected_competitions"]), 3)
            for competition in report["selected_competitions"]:
                self.assertEqual(competition["public_writeups"], 5)
                self.assertEqual(competition["private_writeups"], 5)
                self.assertEqual(competition["unique_canonical_solutions"], 7)

    def test_final_report_never_claims_verification_without_selected_evidence(self):
        with TemporaryCatalog() as fixture:
            result = build_knowledge_base(
                fixture.catalog,
                RICH_PROMPT,
                profile_override=RELEVANT_PROFILE,
                min_competitions=3,
                max_competitions=10,
                initial_threshold=0.0,
                minimum_threshold=0.0,
            )
            report = final_report(result, fixture.root / "empty-knowledge-base.md")

            self.assertFalse(result["all_ranks_and_links_verified"])
            self.assertFalse(report["all_leaderboard_ranks_and_links_verified"])
            self.assertEqual(report["selected_competitions"], [])
            self.assertTrue(report["missing_or_inaccessible_information"])
            self.assertFalse(report["competition_selection"]["minimum_met"])


if __name__ == "__main__":
    unittest.main()
