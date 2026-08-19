from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import kagglehub

HANDLE = "alexandervc/cayleypy-tetraminx-tpu-beam-q2"
ROOT = Path("harvest_probe")
ROOT.mkdir(parents=True, exist_ok=True)

report: dict[str, object] = {
    "handle": HANDLE,
    "kaggle_username_set": bool(os.environ.get("KAGGLE_USERNAME")),
    "kaggle_key_set": bool(os.environ.get("KAGGLE_KEY")),
    "kaggle_api_token_set": bool(os.environ.get("KAGGLE_API_TOKEN")),
    "attempts": [],
}


def tree(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not path.exists():
        return out
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out.append({"path": str(p), "size": p.stat().st_size})
    return out


def attempt(label: str, handle: str, out_dir: Path) -> None:
    rec: dict[str, object] = {"label": label, "handle": handle}
    try:
        result = kagglehub.notebook_output_download(
            handle,
            output_dir=str(out_dir),
            force_download=True,
        )
        rec["ok"] = True
        rec["result"] = str(result)
        rec["files"] = tree(out_dir)
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)
        rec["traceback"] = traceback.format_exc()
    report["attempts"].append(rec)
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)


attempt("latest", HANDLE, ROOT / "latest")
attempt("version_1", f"{HANDLE}/versions/1", ROOT / "v1")

report["all_files"] = tree(ROOT)
(Path("harvest_probe_report.json")).write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

if not any(bool(x.get("ok")) for x in report["attempts"]):
    raise SystemExit(2)
