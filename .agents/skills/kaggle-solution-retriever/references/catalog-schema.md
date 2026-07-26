# Reviewed catalog schema

The catalog is SQLite, populated from newline-delimited JSON. Run:

```bash
python3 scripts/catalog.py ingest-jsonl --input /absolute/path/to/reviewed.jsonl
```

Records may appear in any order; competition records are ingested before dependent records.

## Competition

```json
{
  "record_type": "competition",
  "id": "example-competition",
  "slug": "example-competition",
  "title": "Example Competition",
  "description": "Grouped binary classification of repeated users.",
  "end_date": "2025-01-01T00:00:00Z",
  "status": "completed",
  "tags": ["tabular", "classification"],
  "url": "https://www.kaggle.com/competitions/example-competition",
  "task_types": ["binary classification"],
  "target_types": ["binary", "probability"],
  "modalities": ["tabular"],
  "metrics": ["F1"],
  "dataset_structure": ["grouped users", "repeated entities", "class imbalance"],
  "validation_structure": ["GroupKFold"],
  "leakage_risks": ["entity leakage"],
  "transferable_methods": ["GBDT"],
  "feature_methods": ["aggregation features"],
  "domains": ["retail"],
  "compute_profiles": ["single GPU"],
  "constraints": ["runtime limit"],
  "profile_source_url": "https://www.kaggle.com/competitions/example-competition/overview",
  "profile_verified": true,
  "profile_updated_at": "2026-07-26T10:00:00Z"
}
```

Only `status: completed` is eligible.

## Leaderboard team

Use one canonical row per competition/team. It contains both leaderboard positions.

```json
{
  "record_type": "leaderboard_team",
  "competition_id": "example-competition",
  "team_id": "example-competition-team-abc",
  "team_name": "Team ABC",
  "public_rank": 4,
  "private_rank": 27,
  "public_score": "0.81234",
  "private_score": "0.73125",
  "public_rank_verified": true,
  "private_rank_verified": true,
  "public_rank_source_url": "https://www.kaggle.com/competitions/example-competition/leaderboard",
  "private_rank_source_url": "https://www.kaggle.com/competitions/example-competition/leaderboard",
  "ranks_verified_at": "2026-07-26T10:10:00Z"
}
```

If a real Kaggle team ID is unavailable, use a deterministic ID based on the competition plus exact team identity. Reuse exactly the same ID in the solution record. Omitting `team_id` is allowed when `team_name` is exact; the importer derives a stable fallback, but an explicit ID is safer.

## Canonical solution/writeup

Create one primary solution record per team. Multiple supporting URLs belong in arrays.

```json
{
  "record_type": "solution",
  "id": "example-competition-team-abc-solution",
  "competition_id": "example-competition",
  "team_id": "example-competition-team-abc",
  "team_name": "Team ABC",
  "title": "Team ABC Solution",
  "writeup_title": "4th Place Solution",
  "author": "member_handle",
  "publication_author": "Team ABC",
  "private_rank": 27,
  "public_rank": 4,
  "private_rank_verified": true,
  "public_rank_verified": true,
  "private_rank_source_url": "https://www.kaggle.com/competitions/example-competition/leaderboard",
  "public_rank_source_url": "https://www.kaggle.com/competitions/example-competition/leaderboard",
  "private_score": "0.73125",
  "public_score": "0.81234",
  "ranks_verified_at": "2026-07-26T10:10:00Z",
  "public": true,
  "source_kind": "writeup",
  "writeup_url": "https://www.kaggle.com/competitions/example-competition/writeups/team-abc-solution",
  "solution_verified": true,
  "writeup_verified": true,
  "writeup_verified_at": "2026-07-26T10:12:00Z",
  "source_accessed_at": "2026-07-26T10:15:00Z",
  "verification_note": "Official Solution link on the team's leaderboard row.",
  "code_urls": ["https://github.com/example/team-abc-solution"],
  "notebook_urls": ["https://www.kaggle.com/code/example/example-notebook"],
  "repository_urls": ["https://github.com/example/team-abc-solution"],
  "external_urls": [],
  "core_idea": "Source-stated core method.",
  "validation_strategy": "Source-stated grouped five-fold validation.",
  "preprocessing": "Source-stated preprocessing.",
  "feature_engineering": "Source-stated features.",
  "models": "Source-stated models.",
  "training_procedure": "Source-stated training details.",
  "ensembling": "Source-stated blend.",
  "post_processing": "Source-stated post-processing.",
  "leakage_prevention": "Source-stated leakage controls.",
  "failed_approaches": "Source-stated failed experiments.",
  "compute_requirements": "Source-stated compute.",
  "robustness_notes": "Only robustness claims explicitly made by the authors.",
  "transferable_ideas": "Source-supported transferable ideas.",
  "application_risks": "Analyst risk note, explicitly labeled as analysis.",
  "techniques": ["CatBoost", "OOF blending"],
  "validation_tags": ["GroupKFold"],
  "model_tags": ["CatBoost"],
  "feature_tags": ["group aggregation"],
  "ensemble_tags": ["OOF weighted average"],
  "extracted_facts": [
    "The writeup states that five grouped folds were used."
  ],
  "analyst_inferences": [
    "The large Public-to-Private decline suggests that local validation may not have matched the hidden split."
  ],
  "confidence": "high",
  "license": "unknown/source-specific",
  "provenance_url": "https://www.kaggle.com/competitions/example-competition/writeups/team-abc-solution"
}
```

Rules:

- A strict writeup URL must use the same competition's `/writeups/` route.
- Both cross-ranks and both verification flags are mandatory for selection.
- Both rank-source URLs are mandatory.
- `solution_verified`/`writeup_verified` means the official leaderboard row linked this writeup, not merely that a team member posted it.
- Technical text fields are reserved for source-stated facts, except `application_risks`, which is explicitly analyst-authored and rendered as an inference. Put other conclusions in `analyst_inferences`.
- Use an empty string/list for missing source information. Never synthesize a fact to fill a field.
- URLs are preserved as provenance. External content is never executed automatically.

## Optional leaderboard member

`leaderboard_member` records map user handles to a team for discovery. They do not determine selection and do not consume solution slots.
