"""합성 모델 전용 CPU 실행기. 실제 NPU 경로의 자동 대체로 사용하지 않는다."""

import json
from pathlib import Path

import numpy as np

from .recording import file_hash
from .signal import CausalFilter, FEATURE_ORDER, Normalizer, preprocessing_manifest


class SyntheticOnnxBackend:
    def __init__(self, root):
        import onnxruntime as ort

        ort.disable_telemetry_events()
        root = Path(root)
        self.manifest = json.loads((root / "model.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        _, expected_preprocess = preprocessing_manifest(CausalFilter())
        if (manifest.get("source") != "SYNTHETIC_NOT_USER_EEG"
                or manifest.get("feature_order") != list(FEATURE_ORDER)
                or manifest.get("preprocessing_hash") != expected_preprocess
                or manifest.get("input_shapes") != [[1, 12]]
                or manifest.get("input_dtypes") != ["float32"]
                or manifest.get("class_order") != ["SIM_THETA_6HZ", "SIM_ALPHA_10HZ", "SIM_BETA_24HZ"]
                or file_hash(root / "model.onnx") != manifest.get("onnx_sha256")):
            raise ValueError("합성 모델 출처·입력 계약·해시가 일치하지 않습니다.")
        stats = manifest["norm_stats"]
        self.norm = Normalizer(stats["mean"], stats["std"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root / "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != [1, 12] or self.session.get_inputs()[0].type != "tensor(float)":
            raise ValueError("ONNX 실제 입력 shape/dtype이 매니페스트와 다릅니다.")

    def infer(self, vector):
        scores = self.session.run(None, {"features": self.norm.transform(vector)[None, :]})[0]
        if scores.shape != (1, 3) or not np.isfinite(scores).all():
            raise ValueError("합성 모델 출력이 잘못되었습니다.")
        index = int(scores[0].argmax())
        return self.manifest["class_order"][index], float(scores[0, index])
