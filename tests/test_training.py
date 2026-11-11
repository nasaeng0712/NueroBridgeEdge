"""데이터 누출 차단, 시행별 거부 집계와 합성 모델 정합을 검사한다."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from neurobridge.evaluation import check_splits, trial_metrics


class EvaluationTests(unittest.TestCase):
    def test_split_session_and_trial_overlap(self):
        with self.assertRaises(ValueError):
            check_splits({"training": [{"trial_id": "a", "session_id": "session"}],
                          "final_test": [{"trial_id": "b", "session_id": "session"}]})
        with self.assertRaises(ValueError):
            check_splits({"training": [{"trial_id": "a", "session_id": "a"}],
                          "validation": [{"trial_id": "a", "session_id": "b"}]})

    def test_reject_counted_separately(self):
        report = trial_metrics([0, 0, 1, 1], [0, -1, 1, 0], 2)
        self.assertEqual(report["coverage"], 0.75)
        self.assertEqual(report["balanced_accuracy"], 0.5)
        self.assertEqual(report["all_trial_accuracy"], 0.5)
        self.assertEqual(report["confusion_with_reject_column"], [[1, 0, 1], [1, 1, 0]])
        with self.assertRaises(ValueError):
            trial_metrics([0], [0], 2)

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                         "모델 의존성 미설치: requirements-model.txt 필요")
    def test_synthetic_training_export_and_calibration(self):
        from neurobridge.training import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            report = run(root)
            self.assertLess(report["onnx_max_abs_error"], 1e-5)
            self.assertEqual(report["npu_status"], "BLOCKED")
            self.assertIsNone(report["dxnn_sha256"])
            self.assertGreater(report["evaluation"]["final_test"]["balanced_accuracy"], 0.9)
            dataset = json.loads((root / "synthetic_dataset.json").read_text())
            training_ids = {r["trial_id"] for r in dataset["training"]}
            self.assertEqual(set(report["calibration_trial_ids"]), training_ids)
            from neurobridge.simulation import run as simulate
            integrated = simulate(Path(tmp) / "integrated", model_root=root)
            self.assertEqual(integrated["source"], "SYNTHETIC_CPU_MODEL_REPLAY")
            self.assertEqual(integrated["model_id"], report["model_id"])
            self.assertEqual(integrated["command_count"], 100)
            self.assertIsNotNone(integrated["inference_p95_ms"])
            from neurobridge.backend import SyntheticOnnxBackend
            with (root / "model.onnx").open("ab") as file:
                file.write(b"tampered")
            with self.assertRaises(ValueError):
                SyntheticOnnxBackend(root)
