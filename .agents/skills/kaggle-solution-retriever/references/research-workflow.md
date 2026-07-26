# Verified Kaggle research workflow

Use this workflow only when the local catalog lacks enough evidence for competitions selected for the current task. Do not browse every writeup on Kaggle during every prompt.

## Authoritative admission evidence

The official leaderboard page is the authority:

```text
https://www.kaggle.com/competitions/<competition-slug>/leaderboard
```

A valid writeup must be linked from the row's `Solution` column and resolve to:

```text
https://www.kaggle.com/competitions/<competition-slug>/writeups/<writeup-slug>
```

Notebook votes, search-result rank, discussion votes, medals, and a title containing “solution” do not establish eligibility.

## Targeted collection

For each structurally selected competition:

1. Verify on the official page that the competition is completed.
2. Inspect the **Public** leaderboard from rank 1 downward.
3. Export a complete Public rank prefix beginning at rank 1. Record every scanned team's team name, member handles, Public rank, Public score, and official Solution link when present. Never export only the qualifying rows.
4. Continue until five provisional teams with official Solution links are found. Do not stop at rank 10.
5. Inspect the **Private** leaderboard independently from rank 1 downward and export its complete rank prefix until five qualifying teams are found.
6. Extend the Private prefix through the Private rank of every provisional Public selection, and extend the Public prefix through the Public rank of every provisional Private selection. Keep both exports complete from rank 1. This closure step prevents an incomplete opposite-board export from silently replacing a higher-ranked writeup with a lower-ranked one.
7. Join the two tables by stable team identity. Prefer Kaggle team ID when available; otherwise use the exact team name plus sorted member handles. Never join by rank.
8. Record both ranks and both scores for every qualifying team. If either cross-rank cannot be verified, stop and extend or recheck the evidence; do not substitute a lower-ranked writeup.
9. Deduplicate the union of the Public and Private selections before opening writeups.
10. Read only those unique official writeups. Extract source-stated facts into the structured fields in `catalog-schema.md`.
11. Follow links from an admitted writeup to code/notebooks/repositories only when needed. These links are supporting artifacts; they do not replace the official writeup.
12. Record `source_accessed_at`, `writeup_verified_at`, `ranks_verified_at`, and the official leaderboard URL. The converter will not invent a missing source URL.
13. Ingest the reviewed JSONL and rerun `build_knowledge_base.py`.

When Browser access is available, use focused DOM extraction from the two leaderboard tabs rather than copying the entire page into context. When it is unavailable, import a reviewed manifest prepared from the same official pages. Do not replace missing official evidence with web-search snippets.

Store the compact browser/reviewer export as one JSON object with:

- `competition`;
- `public_leaderboard`;
- `private_leaderboard`;
- optional `team_analyses`;
- `ranks_verified_at`;
- the official `leaderboard_url`;
- `public_scan_exhausted` and `private_scan_exhausted`. Leave them false when five writeups were found. Set a flag true only when the complete corresponding leaderboard was reviewed and still yielded fewer than five.

Then validate and convert it:

```bash
python3 scripts/prepare_research_manifest.py \
  --input /absolute/path/to/leaderboard_research.json \
  --output /absolute/path/to/reviewed_records.jsonl \
  --report /absolute/path/to/research_audit.json

python3 scripts/catalog.py ingest-jsonl \
  --input /absolute/path/to/reviewed_records.jsonl
```

The preparation script independently sorts both boards, requires each export to be a complete rank prefix starting at rank 1, ignores every vote/popularity field, validates official same-competition leaderboard and writeup routes, joins cross-ranks by stable team identity, selects five per board, and emits overlap only once.

## Analysis fields

For each canonical team, capture only what the source supports:

- core idea;
- validation strategy;
- preprocessing;
- feature engineering;
- models;
- training procedure;
- ensembling;
- post-processing;
- leakage prevention;
- failed approaches;
- compute requirements;
- robustness notes explicitly discussed by the authors;
- transferable ideas;
- application risks;
- structured technique/validation/model/feature/ensemble tags;
- explicit facts;
- analyst inferences in a separate list;
- confidence.

Missing content must remain empty or be labeled missing. Do not convert an inference into a source fact.

## Verification checklist

Before setting verification fields to true:

- [ ] competition status is completed;
- [ ] Solution URL came from the official leaderboard row;
- [ ] Solution URL uses the competition's `/writeups/` route;
- [ ] exact team identity is consistent across Public and Private tabs;
- [ ] Public rank and score came from the Public tab;
- [ ] Private rank and score came from the Private tab;
- [ ] cross-ranks are stored in one `leaderboard_team` record;
- [ ] source URLs and timestamps are present;
- [ ] facts and analyst inferences are separate;
- [ ] no notebook-vote or discussion-vote signal was used.

If any mandatory rank/link item fails, keep the record for audit if useful but leave its verification flag false. The strict builder will exclude it and report the missing evidence.
