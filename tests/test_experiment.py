import unittest

from specdec.benchmark import read_stream, summarize
from specdec.experiment import expected_steps, verify
from specdec.replay import TraceEntry


class ExperimentTest(unittest.TestCase):
    def test_partial_rejection_advances_offline_verifier(self):
        trace = {
            (0, 1): TraceEntry((2, 8, 8, 8), 1),
            (0, 1, 2, 3): TraceEntry((4, 5), 1),
        }
        target = {"prompt_token_ids": [0], "token_ids": [1, 2, 3, 4, 5, 6],
                  "request_id": "partial"}
        steps = expected_steps([0], target["token_ids"], trace, 4)
        self.assertEqual([step["output_offset"] for step in steps], [1, 3])
        self.assertEqual(steps[0]["target_token_id"], 3)
        metrics = {"per_step_accepted": [1, 2], "per_step_drafted": [4, 2],
                   "num_accepted_draft_tokens": 3, "num_draft_tokens": 6,
                   "num_spec_steps": 2}
        self.assertEqual(
            verify([target], [{**target, "speculative_decoding": metrics}], trace, 4)[0]["steps"],
            steps,
        )

    def test_stream_distinguishes_chunk_sizes_and_keeps_final_metrics(self):
        payload, timing = read_stream([
            b'data: {"choices":[{"delta":{"role":"assistant"}}],"prompt_token_ids":[9]}\n',
            b'data: {"choices":[{"token_ids":[1,2]}]}\n',
            b'data: {"choices":[{"token_ids":[3],"finish_reason":"length"}]}\n',
            b'data: {"usage":{"completion_tokens":3},"metrics":{"speculative_decoding":{}}}\n',
            b'data: [DONE]\n',
        ], 0)
        self.assertEqual(payload["choices"][0]["token_ids"], [1, 2, 3])
        self.assertEqual(payload["prompt_token_ids"], [9])
        self.assertIn("speculative_decoding", payload["metrics"])
        self.assertEqual([chunk["tokens"] for chunk in timing["token_chunks"]], [2, 1])
        self.assertEqual(len(timing["inter_chunk_ms"]), 1)
        report = summarize([{
            **timing, "status": "ok", "latency_ms": 2,
            "prompt_tokens": 1, "completion_tokens": 3,
        }], 1)
        self.assertEqual(report["ttft_ms"]["p99"], timing["ttft_ms"])

    def test_rejects_truncated_stream(self):
        with self.assertRaisesRegex(RuntimeError, "before"):
            read_stream([b'data: {"choices":[{"token_ids":[1]}]}\n'], 0)
