import unittest

from specdec.latency import LatencyProfile, Proposal
from specdec.replay import Replay, TraceEntry, load_trace


class ReplayTest(unittest.TestCase):
    def test_next_lookup_uses_only_committed_tokens(self):
        replay = Replay({
            (7, 8): TraceEntry((1, 3, 4, 5), 1),
            (7, 8, 1, 2): TraceEntry((6, 7, 8, 9), 1),
        }, 4, LatencyProfile())
        self.assertEqual(replay.propose([[8]], [2], [[7, 8, 0, 0]]), [[1, 3, 4, 5]])
        self.assertEqual(
            replay.propose([[1, 2]], [4], [[7, 8, 1, 2, 3, 4, 5]]),
            [[6, 7, 8, 9]],
        )

    def test_batch_charges_one_wait_and_skips_inactive_rows(self):
        waits, logs = [], []
        replay = Replay({
            (7, 8): TraceEntry((1, 2, 3, 4), 1),
            (9, 8): TraceEntry((1, 2), 1),
        }, 4, LatencyProfile(
            round_trip_ms=2, first_token_ms=3, tbt_ms=1,
            upload_bytes_per_second=1000, download_bytes_per_second=1000,
            kv_transfer="initial", kv_bytes_per_position=10,
        ), wait=waits.append, log=logs.append)
        result = replay.propose([[8], [], [8]], [2, 2, 2], [[7, 8], [0, 0], [9, 8]])
        self.assertEqual(result, [[1, 2, 3, 4], [], [1, 2]])
        self.assertEqual(waits, [0.080])
        self.assertEqual(logs[0]["delay"]["upload_bytes"], 48)
        self.assertEqual(logs[0]["delay"]["download_bytes"], 24)

    def test_kv_can_be_sent_on_every_round_as_a_delta(self):
        profile = LatencyProfile(
            kv_transfer="per_round", kv_scope="delta", kv_direction="download",
            kv_bytes_per_position=12, download_bytes_per_second=1000,
        )
        delay = profile.estimate([Proposal(2, 4, 128)])
        self.assertEqual(delay["kv_bytes"], 24)
        self.assertEqual(delay["total_ms"], 40)

    def test_missing_prefix_policy(self):
        for strict in (True, False):
            with self.subTest(strict=strict):
                logs = []
                replay = Replay({}, 4, LatencyProfile(), log=logs.append, strict=strict)
                if strict:
                    with self.assertRaisesRegex(ValueError, r"missing at lengths \[2\]"):
                        replay.propose([[1]], [2], [[0, 1]])
                else:
                    self.assertEqual(replay.propose([[1]], [2], [[0, 1]]), [[]])
                self.assertEqual(logs[0]["misses"], 1)

    def test_trace_conflicts_are_rejected(self):
        rows = [
            '{"prefix_token_ids":[0,1],"draft_token_ids":[2],"prompt_length":1}',
            '{"prefix_token_ids":[0,1],"draft_token_ids":[3],"prompt_length":1}',
        ]
        with self.assertRaisesRegex(ValueError, "Conflicting proposals"):
            load_trace(rows)


if __name__ == "__main__":
    unittest.main()
