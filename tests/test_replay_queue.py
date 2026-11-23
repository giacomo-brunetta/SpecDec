import unittest

from specdec.replay_queue import simulate


class QueueTest(unittest.TestCase):
    def test_queue_charges_service_only_after_upload_and_can_overlap_download(self):
        batch = {
            "arrival_ms": 1_000_000,
            "proposals": 2,
            "delay": {"round_trip_ms": 2, "upload_ms": 1, "download_ms": 5,
                      "first_token_ms": 1, "tbt_ms": 3},
        }
        results = simulate([[batch], [batch]])
        self.assertEqual([row["queue_ms"] for row in results], [0, 4])
        self.assertEqual([row["total_ms"] for row in results], [12, 16])
        self.assertEqual([row["service_ms"] for row in simulate([[batch]], 0.5)], [2])

    def test_queue_handles_empty_rounds(self):
        self.assertEqual(simulate([[{"arrival_ms": 0, "proposals": 0}, {"misses": 1}]]), [])
