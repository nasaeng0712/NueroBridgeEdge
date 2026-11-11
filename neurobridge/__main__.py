"""장비가 없는 개발 머신에서 합성 원본 기록과 재생을 시험한다."""

import argparse
from pathlib import Path
import tempfile

from .recording import Capture, verify
from .thinkgear import encode_raw


def main():
    parser = argparse.ArgumentParser(description="합성 EEG 기록·재생 검사 (실측 아님)")
    parser.add_argument("--output", type=Path, help="새 기록 폴더")
    args = parser.parse_args()
    root = args.output or Path(tempfile.mkdtemp(prefix="neurobridge-")) / "capture"
    try:
        capture = Capture(root)
    except OSError:
        parser.exit(1, "기록 폴더를 만들 수 없습니다. 새 경로와 쓰기 권한을 확인하세요.\n")
    try:
        for sample in range(-128, 128):
            packet = encode_raw(sample)
            capture.ingest(packet[:5])
            capture.ingest(packet[5:])
    finally:
        capture.close()
    count = verify(root)
    print(f"합성 시험 PASS: {count}개 샘플 원본 해시·재생 일치. 저장 위치: {root}")
    print("실제 EEG 해독·NPU·화면 동작은 검증하지 않았습니다.")


if __name__ == "__main__":
    main()
