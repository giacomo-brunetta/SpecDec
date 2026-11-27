import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from specdec.campaign_sweep import cells, drafter_models, latency_profile
from specdec.study.acceptance import layout, nodes, rounds
from specdec.study.distance import calibration_curve


class StudyTest(unittest.TestCase):
    def test_other_model(self):
        with TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "drafters": {
                            "small": {"model": "org/model", "active_params_b": 3}
                        }
                    }
                )
            )
            models = drafter_models(plan, {"model": "org/target"})
        params = {k: v["active_params_b"] for k, v in models.items()}
        self.assertEqual(len(cells(params)), 20)
        self.assertEqual({m for m, _, _ in cells(params)}, {"small", "oracle"})
        self.assertEqual(
            latency_profile("small", "pcie", "asic", params)["tbt_ms"], 0.6
        )

    def test_variable_length_and_branch_budget(self):
        seq = [3, 2, 1, 2, 1, 1, 3]
        self.assertEqual(list(rounds(seq, 4, 1)), [(1, 4, 2), (4, 2, 2)])
        self.assertEqual(list(rounds([3, 1], 4, 1)), [])
        self.assertEqual(nodes(12, 1), 90)
        self.assertEqual(nodes(32, 1), 560)
        for shape in ("chain", "root leaf", "two chains", "early exits"):
            tree = layout(shape, 16)
            self.assertEqual(len(tree), 16)
            self.assertTrue(all(len(p) == 1 or p[:-1] in tree for p in tree))

    def test_calibration_pooling(self):
        curve = calibration_curve({"1": [5], "5": [8], "9": [6], "12": [9]})
        self.assertEqual(curve, {1: 5, 5: 7, 9: 7, 12: 9})
