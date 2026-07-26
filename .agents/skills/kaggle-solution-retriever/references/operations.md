# Operations

## Storage

The skill is install-location independent. Defaults:

```text
~/.cache/kaggle-solution-retriever/catalog.sqlite3
~/.cache/kaggle-solution-retriever/notebooks/
```

Overrides:

```bash
export KAGGLE_SOLUTION_CACHE_DIR=/absolute/cache/directory
export KAGGLE_SOLUTION_DB=/absolute/catalog.sqlite3
```

The database and fetched notebooks are local cache artifacts and must not be committed.

## Initialize and inspect

```bash
python3 scripts/catalog.py init
python3 scripts/catalog.py status
```

Important status fields:

- `completed_competitions`;
- `leaderboard_teams`;
- `verified_cross_rank_writeups`;
- `schema_version`;
- `last_ingested_at`.

An empty fresh installation is valid, but it cannot produce real recommendations until reviewed metadata is ingested.

## Import competition metadata

Meta Kaggle can seed completed competition metadata:

```bash
python3 scripts/catalog.py import-competitions \
  --meta-dir /absolute/path/to/meta-kaggle
```

The optional legacy `import-meta` command can also import team-authored notebook candidates. Those notebooks are discovery candidates only. They are never admitted as Solution Writeups and notebook votes are ignored.

For exact prompt-time behavior, use `research-workflow.md` to create reviewed `competition`, `leaderboard_team`, and `solution` records for the selected competitions. The reviewed manifest is the authority for official Solution links and technical analysis.

## Ingest reviewed evidence

```bash
python3 scripts/catalog.py ingest-jsonl \
  --input /absolute/path/to/reviewed_records.jsonl
```

The importer validates ranks, URLs, booleans, and foreign keys. It keeps competition profiles and rich solution analyses in schema-v3 side tables.

## Build a knowledge base

```bash
python3 scripts/build_knowledge_base.py \
  --prompt-file /absolute/path/to/task.txt \
  --task-profile /absolute/path/to/task_profile.json \
  --output /absolute/path/to/kaggle_relevant_solutions_knowledge_base.md \
  --context-output /absolute/path/to/kaggle_relevant_solutions_context.md \
  --report-output /absolute/path/to/kaggle_retrieval_report.json
```

The complete Markdown is never truncated to the Codex context budget. Only the separate context summary is bounded.

The report is successful but explicitly incomplete when:

- fewer than three competitions survive the safe relevance floor;
- fewer than five qualifying Public or Private writeups exist in the reviewed catalog;
- mandatory cross-ranks or source URLs are missing;
- official writeup links cannot be verified.

This is intentional. The system must not fill gaps with irrelevant competitions or inferred ranks.

## Refresh policy

- Refresh completed-competition metadata periodically, not on every prompt.
- Research only competitions selected for the current task and cache the reviewed result.
- Recheck links/ranks when Kaggle data changes or a verification timestamp is stale.
- Do not overwrite a reviewed writeup with an unverified notebook candidate.
- Never run downloaded notebooks during refresh.

## On-demand code inspection

After selection, `fetch_notebook.py` may statically extract relevant cells for a selected notebook:

```bash
python3 scripts/fetch_notebook.py \
  --solution-id SOLUTION_ID \
  --query-file /absolute/path/to/task.txt \
  --output /absolute/path/to/notebook_excerpt.md
```

Supporting Kaggle URLs stored in `notebook_urls` are eligible for static fetching. A catalog `local_path` is ignored unless the user explicitly authorizes its parent tree with `--local-code-root /trusted/code/root`. Cache paths are derived from a hash rather than an untrusted solution ID.

The extractor excludes notebook outputs, redacts obvious secret assignments, applies a hard character budget, and never executes code.

## Failure handling

- Kaggle unavailable: keep cached evidence and mark freshness/access gaps.
- Missing credentials for Meta Kaggle download: report the prerequisite; do not invent data.
- Fewer than five official writeups: return the exact verified count and scan cutoff.
- Conflicting ranks for one team: do not infer the correct value; re-verify the official Public and Private tabs.
- Non-`/writeups/` post: reject it for strict admission even if its title says “solution.”
