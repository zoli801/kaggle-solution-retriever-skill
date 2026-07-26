from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / ".agents" / "skills" / "kaggle-solution-retriever" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from catalog import import_competitions  # noqa: E402
from kaggle_core import Catalog  # noqa: E402


class CompetitionBootstrapTests(unittest.TestCase):
    def test_meta_kaggle_bootstrap_seeds_completed_metadata_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Competitions.csv").write_text(
                "Id,Slug,Title,Subtitle,DeadlineDate,"
                "EvaluationAlgorithmAbbreviation\n"
                "1,completed-task,Completed Task,Grouped classification,"
                "2020-01-01T00:00:00Z,F1\n"
                "2,active-task,Active Task,Future competition,"
                "2999-01-01T00:00:00Z,AUC\n",
                encoding="utf-8",
            )

            with Catalog(root / "catalog.sqlite3") as catalog:
                catalog.init()
                report = import_competitions(catalog, root)
                status = catalog.status()
                completed = catalog.conn.execute(
                    """
                    SELECT c.status, c.description, cp.metrics
                    FROM competitions c
                    JOIN competition_profiles cp ON cp.competition_id=c.id
                    WHERE c.id='1'
                    """
                ).fetchone()

            self.assertEqual(report["competitions"], 2)
            self.assertEqual(report["completed_competitions"], 1)
            self.assertEqual(status["completed_competitions"], 1)
            self.assertEqual(status["solutions"], 0)
            self.assertEqual(status["leaderboard_teams"], 0)
            self.assertEqual(completed["status"], "completed")
            self.assertIn("Grouped classification", completed["description"])
            self.assertEqual(json.loads(completed["metrics"]), ["F1"])


if __name__ == "__main__":
    unittest.main()
