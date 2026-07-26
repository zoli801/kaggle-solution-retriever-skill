# Task profile schema

`build_knowledge_base.py` can infer a conservative profile from the prompt. For a nuanced task, Codex should write a JSON object with any of the fields below and pass it through `--task-profile`.

```json
{
  "task_summary": "Grouped binary classification of user sessions with severe class imbalance.",
  "task_types": ["binary classification"],
  "target_types": ["binary", "probability"],
  "modalities": ["tabular", "time series"],
  "metrics": ["F1"],
  "dataset_structure": [
    "grouped users",
    "temporal order",
    "repeated entities",
    "class imbalance"
  ],
  "train_test_structure": ["group-disjoint train/test", "time-ordered train/test"],
  "validation_structure": ["GroupKFold", "time-aware holdout"],
  "leakage_risks": ["entity leakage", "temporal leakage"],
  "transferable_methods": ["GBDT", "sequence model"],
  "feature_methods": ["lag features", "aggregation features"],
  "domains": ["retail"],
  "compute_profiles": ["single GPU", "inference constrained"],
  "constraints": ["runtime limit"],
  "profile_source": "Codex analysis of the current prompt"
}
```

Rules:

- Values may be a string or a list of strings; lists are preferred.
- Omit unknown fields. Never guess a metric, target, or split constraint.
- Describe structural properties precisely. For example, use `grouped users` plus `temporal order`, not only `tabular`.
- Keep the profile about the current task. Do not copy candidate-competition metadata into it.
- The override is merged with the deterministic fallback. A supplied field replaces that fallback field.

The weighted competition score is:

| Component | Weight |
|---|---:|
| Task and target similarity | 25% |
| Modality and dataset structure | 20% |
| Evaluation metric | 15% |
| Validation and leakage structure | 15% |
| Transferable modeling and feature methods | 15% |
| Domain | 5% |
| Compute constraints | 5% |

The structural hard gate is evaluated before the numeric threshold. A generic category match cannot rescue a modality or validation mismatch.
