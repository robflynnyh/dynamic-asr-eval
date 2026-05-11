from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.linear_experiment_callback import build_comment, tail


class LinearExperimentCallbackTest(unittest.TestCase):
    def test_tail_caps_long_transcript_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text("start\nWER: 0.42\n" + ("word " * 50_000) + "FINAL\n")

            excerpt = tail(str(log_path), lines=80, max_chars=512)

        self.assertLessEqual(len(excerpt), 512)
        self.assertIn("truncated", excerpt)
        self.assertIn("FINAL", excerpt)

    def test_build_comment_stays_under_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            log_path.write_text("start\n" + ("token " * 20_000) + "\n")
            args = argparse.Namespace(
                status_code=0,
                branch="branch",
                commit="commit",
                queued_command="with-gpu 1,2 -- wrapper",
                screen_name="screen",
                runner_label="screen:screen",
                log=str(log_path),
                results=tmpdir,
                target_state="Todo",
                note="note",
                tail_lines=80,
                max_log_chars=2_000,
                max_comment_chars=4_000,
            )
            issue = {"identifier": "ROB-56"}

            comment = build_comment(args, issue)

        self.assertLessEqual(len(comment), 4_000)
        self.assertIn("Queued experiment succeeded.", comment)
        self.assertIn("Log path:", comment)


if __name__ == "__main__":
    unittest.main()
