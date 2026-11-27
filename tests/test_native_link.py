import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from specdec.native_campaign import profile_args, rows_from_log
from specdec.native_link import finish_round


class NativeLinkTest(unittest.TestCase):
    def test_round_logs_are_audited_with_full_prompt_and_padded_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rounds.jsonl"
            settings = json.dumps({"profile": profile_args("hdr100"), "log": str(log)})
            calls = []
            with patch("specdec.native_link.wait_until_deadline",
                       side_effect=lambda seconds: calls.append(seconds)):
                for positions, verified, sampled in ((128, 0, [1]), (5, 4, [2, 3, -1])):
                    finish_round(
                        settings, activation_bytes=positions * 3 * 5376 * 2,
                        positions=positions, requests=1, draft_tokens=4, verified_tokens=verified,
                        sampled_count=lambda: sum(token >= 0 for token in sampled),
                        synchronize=lambda: calls.append("sync"),
                        log_rank=True,
                    )
            rows = rows_from_log(log, [{
                "request_id": "one", "prompt_tokens": 128,
                "step_log_start": 0, "step_log_end": log.stat().st_size,
            }], "eagle3", "hdr100")
            self.assertEqual(calls[::2], ["sync", "sync"])
            self.assertAlmostEqual(calls[3] * 1000, 0.03844533333333334)
            self.assertEqual(rows[0]["activation_bytes"], (128 + 5) * 3 * 5376 * 2)
            self.assertEqual(rows[0]["verified_draft_tokens"], 4)
            self.assertEqual(rows[0]["accepted_draft_tokens"], 1)
            self.assertEqual(rows[0]["round_shape"], [[128, 4, 0, 0], [5, 4, 4, 1]])

    def test_ideal_synchronizes_and_logs_zero_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rounds.jsonl"
            settings = json.dumps({"profile": profile_args("ideal"), "log": str(log)})
            calls = []
            with patch("specdec.native_link.wait_until_deadline") as wait:
                finish_round(
                    settings, activation_bytes=32 * 5 * 5376 * 2,
                    positions=32, requests=1, draft_tokens=7, verified_tokens=0,
                    sampled_count=lambda: 1, synchronize=lambda: calls.append("sync"),
                    log_rank=True,
                )
            wait.assert_not_called()
            self.assertEqual(calls, ["sync"])
            self.assertEqual(json.loads(log.read_text())["total_ms"], 0)
