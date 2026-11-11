"""Linux에서 이미 페어링된 MindWave에 직접 연결하는 RFCOMM 수집 진입점."""

import argparse
import json
from pathlib import Path
import re
import socket
import time

from .recording import Capture, verify


class CaptureError(RuntimeError):
    pass


def rfcomm_connect(address, channel):
    if not hasattr(socket, "AF_BLUETOOTH") or not hasattr(socket, "BTPROTO_RFCOMM"):
        raise CaptureError("이 운영체제의 Python은 Bluetooth RFCOMM 소켓을 지원하지 않습니다.")
    connection = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    try:
        connection.settimeout(5)
        connection.connect((address, channel))
        return connection
    except OSError:
        connection.close()
        raise


def collect(output, connector, duration_s=300, max_reconnects=3, clock=time.monotonic, source="MEASURED"):
    if duration_s <= 0 or max_reconnects < 0:
        raise ValueError("수집 시간은 양수, 재연결 상한은 음수가 아니어야 합니다.")
    capture = Capture(output, source=source)
    started, reconnects, chunks, samples = clock(), 0, 0, 0
    event_file = (capture.root / "link.jsonl").open("x", encoding="utf-8")
    connection = None

    def event(state):
        event_file.write(json.dumps({"state": state, "edge_mono_ns": time.monotonic_ns(),
                                     "segment_id": capture.segment_id}, ensure_ascii=False) + "\n")
        event_file.flush()

    try:
        while clock() - started < duration_s:
            try:
                event("CONNECTING")
                connection = connector()
                connection.settimeout(1)
                event("STREAMING")
                while clock() - started < duration_s:
                    try:
                        chunk = connection.recv(4096)
                    except socket.timeout as error:
                        raise CaptureError("원본 수신이 정지되어 연속성을 보장할 수 없습니다.") from error
                    if not chunk:
                        raise CaptureError("Bluetooth 연결이 종료되었습니다.")
                    # 원본 기록이 성공한 뒤만 파싱된 샘플을 수집 결과로 센다.
                    packets = capture.ingest(chunk)
                    samples += sum(len(packet.raw_samples) for packet in packets)
                    chunks += 1
            except (OSError, CaptureError):
                event("DISCONNECTED_STOP_REQUIRED")
                if capture.failed:
                    raise CaptureError("원본 저장에 실패해 수집을 중단했습니다.")
                if reconnects >= max_reconnects:
                    raise CaptureError("재연결 상한을 초과했습니다. 어댑터·페어링·권한을 확인하세요.")
                reconnects += 1
                capture.reconnect()
                event("RECONNECTING")
            finally:
                if connection is not None:
                    connection.close()
                    connection = None
        if samples == 0:
            raise CaptureError("수집 구간에서 유효한 원시 EEG 샘플을 얻지 못했습니다.")
        event("CLOSED")
    finally:
        event_file.close()
        capture.close()
    replay_count = verify(capture.root)
    report = {"source": source, "samples": samples, "chunks": chunks, "reconnects": reconnects,
              "elapsed_s": clock() - started, "replay_sample_count": replay_count,
              "parser_errors": capture.parser.errors,
              "npu_status": "NOT TESTED", "intent_decoding_status": "NOT TESTED"}
    (capture.root / "capture_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Orange Pi 직접 Bluetooth 원본 수집 (실기 미검증)")
    parser.add_argument("--address", required=True, help="이미 페어링된 헤드셋 Bluetooth 주소")
    parser.add_argument("--channel", type=int, default=1, help="실제 서비스에서 확인한 RFCOMM 채널")
    parser.add_argument("--output", type=Path, required=True, help="새 로컬 기록 폴더")
    parser.add_argument("--duration", type=float, default=300, help="수집 제한 시간(초)")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", args.address) or not 1 <= args.channel <= 30:
        parser.error("Bluetooth 주소 또는 채널이 잘못되었습니다.")
    try:
        report = collect(args.output, lambda: rfcomm_connect(args.address, args.channel), args.duration)
    except (CaptureError, OSError, ValueError) as error:
        parser.exit(1, f"원본 수집 실패: {error}\n")
    print(f"원본 수집 완료: {report['samples']}샘플, 재생 {report['replay_sample_count']}샘플 일치")
    print("수집 완료는 T-02 전체 및 의도 판별·NPU 검증의 PASS를 뜻하지 않습니다.")


if __name__ == "__main__":
    main()
