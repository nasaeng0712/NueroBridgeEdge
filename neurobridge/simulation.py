"""합성 신호와 사전 정의 명령 통합 경로를 실행한다. EEG 분류기가 아니다."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import time
import uuid

import numpy as np

from .protocol import Decoder, encode
from .recording import Capture, file_hash, verify
from .safety import Policy, SafetyReceiver, Sync
from .signal import CausalFilter, features, preprocessing_manifest
from .thinkgear import encode_raw
from .windowing import WindowBuilder


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    capture = Capture(root / "capture")
    filt, builder = CausalFilter(), WindowBuilder()
    feature_rows = []
    try:
        for index in range(2048):
            sample = int(100 * np.sin(2 * np.pi * 10 * index / 512))
            for packet in capture.ingest(encode_raw(sample)):
                window = builder.feed(capture.segment_id, index, packet.raw_samples[0], 0)
                if window:
                    # 연속 필터의 상태를 중첩 윈도우마다 다시 적용하지 않는다.
                    feature_rows.append(window)
    finally:
        capture.close()
    parsed = verify(capture.root)
    raw = np.fromfile(capture.root / "eeg.i16le", dtype="<i2")
    filtered = filt.process(raw, capture.segment_id)
    manifest, preprocess_hash = preprocessing_manifest(filt)
    with (root / "features.jsonl").open("x", encoding="utf-8") as file:
        for number, window in enumerate(feature_rows):
            file.write(json.dumps({"window_id": f"window-{number}", "segment_id": window.segment_id,
                                  "input_sample_range": [window.sample_start, window.sample_stop],
                                  "source": "SYNTHETIC", "preprocess_sha256": preprocess_hash,
                                  "features": features(filtered[window.sample_start:window.sample_stop]).tolist()}) + "\n")
    identity = {"session_id": capture.session_id, "boot_id": capture.boot_id,
                "sync_epoch": str(uuid.uuid4()), "connection_epoch": str(uuid.uuid4())}
    policy = Policy()
    receiver = SafetyReceiver(identity, "SCRIPTED_NOT_A_MODEL", "0" * 64, policy)
    receiver.prepare(Sync(0, 0), time.monotonic_ns(), recording_ok=not capture.failed,
                     model_ok=True, fresh_samples=True)
    timings, commands = [], []
    # 사전 정의된 명령은 특징·EEG에서 추론한 결과가 아니다.
    actions = ["ROTATE_CW", "STOP", "ROTATE_CCW", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"]
    with socket.socket() as server:
        server.settimeout(2)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2) as sender:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(2)
                decoder = Decoder()
                for number in range(100):
                    source_ns = time.monotonic_ns()
                    window = feature_rows[number % len(feature_rows)]
                    intent = dict(identity, schema_version="1.0", type="INTENT",
                                  stream_id=capture.stream_id, segment_id=capture.segment_id,
                                  trial_id="synthetic-trial", window_id=f"window-{number % len(feature_rows)}",
                                  prediction_id=f"scripted-{number}", intent_id=f"intent-{number}",
                                  model_id="SCRIPTED_NOT_A_MODEL", model_sha256="0" * 64,
                                  input_sample_range=[window.sample_start, window.sample_stop],
                                  decision_status="VALID", action=actions[number % len(actions)], score=1.0,
                                  source_window_end_edge_ns=source_ns, edge_created_mono_ns=time.monotonic_ns(),
                                  error_code=None, command_generation_policy_version="synthetic-v1")
                    sender.sendall(encode(intent))
                    messages = []
                    while not messages:
                        chunk = connection.recv(37)
                        if not chunk:
                            raise RuntimeError("합성 시험 TCP 연결이 종료되었습니다.")
                        messages.extend(decoder.feed(chunk))
                    received_ns = time.monotonic_ns()
                    action = receiver.receive(messages[0], received_ns)
                    if not receiver.armed or action != intent["action"]:
                        raise RuntimeError("합성 명령의 안전 수신 검사에 실패했습니다.")
                    timings.append(received_ns - source_ns)
                    commands.append({"scene_id": "demo", "object_id": "subject", "action": action,
                                     "magnitude": 1.0, "duration_ms": 50, "source_intent_id": intent["intent_id"],
                                     "validation_status": "VALIDATED_SYNTHETIC", "render_policy_version": "synthetic-v1"})
    receiver.stop("TCP_EOF")
    with (root / "scene_commands.jsonl").open("x", encoding="utf-8") as file:
        for command in commands:
            file.write(json.dumps(command) + "\n")
    manifest.update({"source": "SYNTHETIC_SCRIPTED", "policy": asdict(policy),
                     "preprocess_sha256": preprocess_hash, "sample_count": parsed,
                     "window_count": len(feature_rows), "command_count": len(commands),
                     "tcp_p50_ms": float(np.percentile(timings, 50) / 1e6),
                     "tcp_p95_ms": float(np.percentile(timings, 95) / 1e6),
                     "total_wall_ms": (time.perf_counter_ns() - started) / 1e6,
                     "total_cpu_ms": (time.process_time_ns() - cpu_started) / 1e6,
                     "scene_commands_sha256": file_hash(root / "scene_commands.jsonl"),
                     "hardware_status": "NOT TESTED", "neural_decoding_status": "NOT TESTED"})
    (root / "simulation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="합성 신호·사전 정의 명령 통합 검사")
    parser.add_argument("--output", type=Path, required=True, help="존재하지 않는 새 출력 폴더")
    args = parser.parse_args()
    try:
        result = run(args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"합성 통합 시험 실패: {error}\n")
    print(f"합성 통합 PASS: {result['sample_count']}샘플, {result['window_count']}윈도우, {result['command_count']}사전 정의 명령")
    print(f"TCP 수신 p95: {result['tcp_p95_ms']:.3f}ms (추론·렌더 지연 아님)")
    print("실제 EEG 분류와 NPU 실행은 검증하지 않았습니다.")


if __name__ == "__main__":
    main()
