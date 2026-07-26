# Kaggle Solution Retriever Behavior Specification

This document is the normative behavior contract for the
`kaggle-solution-retriever` Codex skill.

## Objective

When a user explicitly invokes the skill for a task, Codex must analyze the
current task, identify its technical characteristics, find the most relevant
completed Kaggle competitions, extract the strongest publicly available
solution writeups, and create a focused Markdown knowledge base for solving the
current task.

The skill must not load a large generic collection of Kaggle material into the
Codex context. It must retrieve only information that is highly relevant to the
current problem.

## 1. Analyze the current task

Parse the user's complete prompt and create an internal structured task profile
before searching for competitions.

Extract as many relevant characteristics as possible:

- machine-learning domain;
- task type;
- data modality;
- target type;
- evaluation metric;
- dataset size and structure;
- train/test structure;
- groups, users, time, spatial information, or repeated entities;
- expected validation strategy;
- likely leakage risks;
- computational constraints;
- whether the task is tabular, computer vision, NLP, time series, recommender
  systems, reinforcement learning, multimodal, graph learning, optimization, or
  another category;
- unusual task-specific or competition-specific constraints.

Do not treat tasks as equivalent merely because they share a broad label. For
example, grouped and highly imbalanced binary classification evaluated with F1
is more similar to another grouped, imbalanced F1 task than to independent-row
binary classification evaluated with ROC AUC.

## 2. Find relevant completed Kaggle competitions

Search only completed Kaggle competitions.

Select between 3 and 10 competitions that are genuinely relevant to the current
task:

- prioritize quality and similarity over quantity;
- reject competitions that share only a broad category while differing
  substantially in data structure, metric, validation requirements, or
  modeling approach;
- prefer competitions whose strongest solution methods can realistically
  transfer to the current task;
- never add clearly irrelevant competitions merely to reach the maximum.

Rank candidate competitions with a weighted relevance score. Consider:

- task-type similarity;
- target similarity;
- data-modality similarity;
- dataset-structure similarity;
- evaluation-metric similarity;
- validation-design similarity;
- leakage-risk similarity;
- modeling-technique transferability;
- feature-engineering transferability;
- domain similarity;
- computational-constraint similarity.

Use the following default weighting:

| Factor | Weight |
|---|---:|
| Task and target similarity | 25% |
| Data modality and structure | 20% |
| Evaluation metric | 15% |
| Validation structure | 15% |
| Transferable modeling methods | 15% |
| Domain similarity | 5% |
| Computational similarity | 5% |

Weights may be adjusted when justified, but task structure, validation design,
and transferable techniques must remain the highest priorities.

Apply a strict relevance threshold. If fewer than three competitions pass,
relax the threshold slightly until three strong candidates are found. Do not
cross the boundary into clearly irrelevant competitions.

## 3. Select Public Leaderboard solution writeups

For each selected competition, inspect its final leaderboard and publicly
available solution writeups.

Collect exactly five of the highest-ranked teams by Public Leaderboard position
that have a publicly accessible **Solution Writeup**, whenever at least five
such writeups exist.

Selection procedure:

1. Start at Public Rank 1.
2. Preserve every scanned leaderboard row as a complete rank prefix; do not
   export only teams that have writeups.
3. Include the team when it has a qualifying public Solution Writeup.
4. If the team has no qualifying writeup, skip it.
5. Continue downward until five qualifying writeups are collected or no more
   verifiable candidates are available.

The selection is based on leaderboard position, not notebook votes, discussion
popularity, or search-result order.

For every Public-selected solution, record:

- team name;
- Public Leaderboard rank;
- Private Leaderboard rank;
- public score, when available;
- private score, when available;
- direct Kaggle Solution Writeup URL;
- direct competition URL;
- writeup title;
- publication author or team;
- whether source code or a notebook is available;
- links to public code, notebooks, repositories, datasets, and external
  technical resources referenced by the writeup;
- detailed technical summary;
- validation strategy;
- preprocessing;
- feature engineering;
- model architecture or model family;
- training procedure;
- ensembling or blending;
- post-processing;
- leakage prevention;
- important failed experiments;
- methods transferable to the current task.

Do not treat a generic discussion, leaderboard comment, unrelated notebook, or
third-party summary as the team's solution writeup. Prefer the official Kaggle
Solution Writeup associated with the team or competition.

## 4. Select Private Leaderboard solution writeups

For each selected competition, also collect exactly five of the highest-ranked
teams by Private Leaderboard position that have publicly accessible Solution
Writeups, whenever at least five such writeups exist.

Selection procedure:

1. Start at Private Rank 1.
2. Preserve every scanned leaderboard row as a complete rank prefix.
3. Include the team when it has a qualifying public Solution Writeup.
4. If the team has no qualifying writeup, skip it.
5. Continue downward until five qualifying writeups are collected or no more
   verifiable candidates are available.

For every Private-selected solution, record:

- team name;
- Private Leaderboard rank;
- Public Leaderboard rank;
- private score, when available;
- public score, when available;
- direct Kaggle Solution Writeup URL;
- direct competition URL;
- writeup title;
- all available code and notebook links;
- detailed technical summary;
- validation strategy;
- modeling methods;
- ensembling;
- post-processing;
- Private Leaderboard robustness;
- methods transferable to the current task.

Cross-leaderboard ranks are mandatory:

- every Public-selected solution must include its Private rank;
- every Private-selected solution must include its Public rank.

If a cross-leaderboard rank cannot be verified, exclude that candidate from the
selection and report the verification gap. Never infer it.

## 5. Handle overlapping teams

A team may qualify for both Public and Private selection.

Do not duplicate the full analysis. Create one canonical solution entry and:

- mark it as selected through Public, Private, or both;
- include both leaderboard positions;
- reference the canonical entry from both selection tables.

Each competition should normally contain five qualifying Public selections and
five qualifying Private selections. The number of unique teams may be smaller
than ten because of overlap.

## 6. Analyze leaderboard shakeup

For every selected competition, compare the collected teams' Public and Private
ranks.

Calculate or describe:

- absolute rank difference;
- direction of rank change;
- whether the team improved or declined on the Private Leaderboard;
- whether the competition had a substantial leaderboard shakeup;
- techniques associated with robust Private Leaderboard performance;
- techniques that may have overfit the Public Leaderboard;
- validation lessons transferable to the current task.

Do not assume a high Public rank identifies the strongest generalizable
solution. Give greater analytical weight to methods that performed consistently
on the Private Leaderboard.

For a decline, report the positive drop explicitly:

```text
Public rank 4 → Private rank 27 (drop: 23 places)
```

## 7. Build the Markdown knowledge base

Create one Markdown file containing all selected competitions and collected
solutions. The default filename is:

```text
kaggle_relevant_solutions_knowledge_base.md
```

Use the following structure.

### Document title

```markdown
# Kaggle Knowledge Base for the Current Task
```

### Current Task Profile

Include:

- task summary;
- data modality;
- prediction target;
- metric;
- dataset structure;
- validation requirements;
- likely risks;
- recommended competition-search criteria.

### Competition Selection Summary

Use this table:

| Competition | Relevance Score | Main Similarities | Important Differences | Number of Public Solutions | Number of Private Solutions |
|---|---:|---|---|---:|---:|

### One section per competition

For each selected competition, create:

```markdown
## Competition N: Competition Name
```

#### Competition Metadata

Include:

- competition name;
- competition URL;
- competition status;
- task type;
- metric;
- dataset structure;
- relevance score;
- reason for inclusion;
- important differences from the current task.

#### Public Leaderboard Solution Selection

Use this table:

| Selected Position | Team | Public Rank | Private Rank | Public Score | Private Score | Solution Writeup |
|---:|---|---:|---:|---:|---:|---|

Number `Selected Position` from 1 to 5 according to the qualifying Public
Leaderboard writeups, not raw leaderboard rows.

#### Private Leaderboard Solution Selection

Use this table:

| Selected Position | Team | Private Rank | Public Rank | Private Score | Public Score | Solution Writeup |
|---:|---|---:|---:|---:|---:|---|

Number `Selected Position` from 1 to 5 according to the qualifying Private
Leaderboard writeups.

#### Canonical Solution Analyses

For every unique selected team, create:

```markdown
#### Team Name — Public Rank X, Private Rank Y

- Selected through: Public / Private / Both
- Solution Writeup:
- Competition:
- Code or notebook:
- External repository:
- Core idea:
- Validation strategy:
- Preprocessing:
- Feature engineering:
- Models:
- Training procedure:
- Ensembling:
- Post-processing:
- Leakage prevention:
- Failed approaches:
- Compute requirements:
- Public-to-Private rank change:
- Reasons for robustness or shakeup:
- Transferable ideas for the current task:
- Risks of applying this method:
- Confidence in extracted information:
```

#### Competition-Level Lessons

Summarize:

- strongest recurring methods;
- most Private-robust approaches;
- validation lessons;
- leakage risks;
- useful feature-engineering ideas;
- useful model architectures;
- useful ensembling techniques;
- methods that should not be transferred blindly.

Repeat this structure for every selected competition.

### Cross-Competition Synthesis

After all competition sections, include the following subsections.

#### Repeated High-Value Techniques

Identify techniques appearing in multiple successful solutions.

#### Validation Recommendations

Propose a validation design for the current task based on the most relevant
competitions.

#### Model Recommendations

Rank the most promising model families and explain why they fit the current
task.

#### Feature-Engineering Recommendations

Include only features and transformations that can realistically transfer.

#### Ensembling Recommendations

Explain diversity sources and blending methods that were consistently
effective.

#### Public-to-Private Robustness

Identify methods associated with stable Private performance and methods
associated with Public Leaderboard overfitting.

#### Proposed Experimental Plan

Prioritize:

1. a strong baseline;
2. correct local validation;
3. high-value feature engineering;
4. strongest single models;
5. ensemble construction;
6. robustness checks;
7. final submission selection.

### Source Index

End the file with a deduplicated source index containing:

- every competition URL;
- every Solution Writeup URL;
- every notebook URL;
- every code repository URL;
- every external technical source used.

## 8. Quality and verification requirements

The workflow must:

- use only completed Kaggle competitions;
- select between 3 and 10 highly relevant competitions;
- use a strict relevance threshold;
- avoid selection based on superficial keyword overlap;
- collect five qualifying Public writeups per competition whenever at least five
  are publicly available;
- collect five qualifying Private writeups per competition whenever at least
  five are publicly available;
- skip teams without public Solution Writeups and continue down the leaderboard;
- include the Private rank for every Public-selected solution;
- include the Public rank for every Private-selected solution;
- deduplicate teams appearing in both selections;
- preserve direct links to original sources;
- accept rank evidence only from the official HTTPS leaderboard route for the
  same competition;
- never invent ranks, scores, solution details, links, or methods;
- clearly mark missing or unverifiable information;
- distinguish facts explicitly stated in sources from analytical inferences;
- prefer original Kaggle writeups and original code over secondary summaries;
- summarize compactly while preserving important technical detail;
- focus on transferable methods rather than generic competition descriptions;
- avoid placing all raw source content into Codex context;
- save complete research in the Markdown file;
- provide Codex with only a concise task-specific retrieval summary.

When fewer than five qualifying writeups are publicly available for a
leaderboard, include all verified writeups found and report the exact shortage.

## 9. Final skill output

After completing the workflow, return:

1. the path to the generated Markdown knowledge-base file;
2. the selected Kaggle competitions;
3. the relevance score for every competition;
4. the number of qualifying Public and Private solution writeups found;
5. missing or inaccessible information;
6. a compact summary of the most useful techniques for the current task;
7. the recommended validation and modeling strategy;
8. confirmation that leaderboard ranks and links were verified rather than
   inferred.

The final knowledge base must be precise, source-linked, technically useful, and
small enough to avoid unnecessarily filling the Codex context window.
