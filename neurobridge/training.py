"""CPU 합성 신호로 소형 MLP·ONNX 배포 도구를 시험한다. 실제 EEG 학습이 아니다."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .evaluation import check_splits, trial_metrics
from .recording import file_hash
from .signal import CausalFilter, FEATURE_ORDER, Normalizer, features, preprocessing_manifest

CLASS_ORDER = ["SIM_THETA_6HZ", "SIM_ALPHA_10HZ", "SIM_BETA_24HZ"]


def synthetic_dataset(seed=41):
    rng = np.random.default_rng(seed)
    splits = {}
    for split in ("training", "validation", "final_test"):
        rows = []
        for session in range(2):
            for label, frequency in enumerate((6, 10, 24)):
                for trial in range(20):
                    signal = 100 * np.sin(2 * np.pi * frequency * np.arange(1536) / 512 + rng.uniform(0, 6))
                    signal += rng.normal(0, 20, len(signal))
                    filtered = CausalFilter().process(signal, "trial")
                    rows.append({"session_id": f"{split}-{session}",
                                 "trial_id": f"{split}-{session}-{label}-{trial}",
                                 "label": label, "features": features(filtered[-1024:]).tolist()})
        splits[split] = rows
    check_splits(splits)
    return splits


def softmax(logits):
    value = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return value / value.sum(axis=-1, keepdims=True)


def forward(x, weights):
    hidden = np.maximum(x @ weights["w1"] + weights["b1"], 0)
    return softmax(hidden @ weights["w2"] + weights["b2"])


def fit(x, labels, seed=41, epochs=500):
    rng = np.random.default_rng(seed)
    weights = {"w1": (rng.normal(size=(12, 16)) * np.sqrt(2 / 12)).astype(np.float32),
               "b1": np.zeros(16, np.float32),
               "w2": (rng.normal(size=(16, 3)) * np.sqrt(2 / 16)).astype(np.float32),
               "b2": np.zeros(3, np.float32)}
    targets = np.eye(3, dtype=np.float32)[labels]
    for _ in range(epochs):
        hidden = np.maximum(x @ weights["w1"] + weights["b1"], 0)
        error = (softmax(hidden @ weights["w2"] + weights["b2"]) - targets) / len(x)
        grad_w2, grad_b2 = hidden.T @ error, error.sum(axis=0)
        hidden_error = (error @ weights["w2"].T) * (hidden > 0)
        weights["w1"] -= 0.05 * (x.T @ hidden_error)
        weights["b1"] -= 0.05 * hidden_error.sum(axis=0)
        weights["w2"] -= 0.05 * grad_w2
        weights["b2"] -= 0.05 * grad_b2
    return weights


def export_onnx(weights, path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph([
        helper.make_node("Gemm", ["features", "w1", "b1"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["activated"]),
        helper.make_node("Gemm", ["activated", "w2", "b2"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["scores"], axis=1),
    ], "NeuroBridgeSyntheticMLP", [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 12])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(value, name) for name, value in weights.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run(root):
    import onnx
    import onnxruntime as ort

    ort.disable_telemetry_events()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter_ns()
    splits = synthetic_dataset()
    dataset_bytes = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    (root / "synthetic_dataset.json").write_bytes(dataset_bytes)
    train = splits["training"]
    norm = Normalizer.fit([r["features"] for r in train], "training")
    data = {name: np.stack([norm.transform(row["features"]) for row in rows]) for name, rows in splits.items()}
    labels = {name: np.array([row["label"] for row in rows]) for name, rows in splits.items()}
    weights = fit(data["training"], labels["training"])
    np.savez(root / "weights.npz", **weights)
    # 보정 후보에는 학습 분할만 포함한다. DX-COM이 실행된 것은 아니다.
    np.save(root / "calibration-training.npy", data["training"])
    export_onnx(weights, root / "model.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    report, max_error = {}, 0.0
    for name in ("validation", "final_test"):
        actual = np.concatenate([session.run(None, {"features": row[None, :]})[0] for row in data[name]])
        expected = forward(data[name], weights)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        report[name] = trial_metrics(labels[name], actual.argmax(axis=1), 3)
    _, preprocess_hash = preprocessing_manifest(CausalFilter())
    calibration_trials = [r["trial_id"] for r in train]
    manifest = {"model_id": "synthetic-mlp-v1", "source": "SYNTHETIC_NOT_USER_EEG",
                "model_sha256": file_hash(root / "weights.npz"),
                "training_dataset_hash": hashlib.sha256(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "full_synthetic_dataset_hash": hashlib.sha256(dataset_bytes).hexdigest(),
                "calibration_hash": file_hash(root / "calibration-training.npy"),
                "calibration_split": "training", "calibration_trial_ids": calibration_trials,
                "architecture": "Dense(12,16)-ReLU-Dense(16,3)-Softmax",
                "input_names": ["features"], "input_shapes": [[1, 12]], "input_dtypes": ["float32"],
                "output_dtypes": ["float32"], "feature_order": FEATURE_ORDER, "class_order": CLASS_ORDER,
                "preprocessing_hash": preprocess_hash,
                "norm_stats": {"mean": norm.mean.tolist(), "std": norm.std.tolist()},
                "onnx_sha256": file_hash(root / "model.onnx"), "dxnn_sha256": None,
                "sdk_runtime_versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "dxrt": None},
                "firmware_version": None, "compiled_op_report": None,
                "evaluation_protocol_hash": hashlib.sha256(b"synthetic-trial-argmax-v1-no-reject").hexdigest(),
                "npu_status": "BLOCKED", "int8_status": "NOT TESTED", "evaluation": report,
                "onnx_max_abs_error": max_error, "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
                "score_semantics": "합성 클래스 softmax 점수이며 사용자 의도 확률이 아님"}
    (root / "model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="합성 분류기 CPU 학습·ONNX 정합 검사")
    parser.add_argument("--output", type=Path, required=True, help="새 출력 폴더")
    args = parser.parse_args()
    try:
        report = run(args.output)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"합성 모델 시험 실패: {error}\n")
    print(f"합성 모델·ONNX 정합 PASS: 최대 절대오차 {report['onnx_max_abs_error']:.8f}")
    print("실제 EEG 의도 판별·INT8·DX-M1 실행은 검증하지 않았습니다.")


if __name__ == "__main__":
    main()
