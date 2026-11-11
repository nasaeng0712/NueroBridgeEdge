"""P1 실기 재개용 읽기 전용 조사. 장치 인식은 실제 NPU 추론 성공이 아니다."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess


def collect():
    pci = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        entry = {"address": device.name}
        for field in ("vendor", "device", "class"):
            try:
                entry[field] = (device / field).read_text().strip()
            except OSError:
                entry[field] = None
        driver = device / "driver"
        entry["driver"] = driver.resolve().name if driver.exists() else None
        pci.append(entry)
    tools = {}
    for name in ("dxrt-cli", "dxrt_cli", "dx_com", "bluetoothctl", "btmgmt", "lspci"):
        tools[name] = shutil.which(name) is not None
    bluetooth = [p.name for p in Path("/sys/class/bluetooth").glob("hci*")]
    # 이름은 후보 탐색용일 뿐 SDK 지원 장치임을 확정하지 않는다.
    candidates = [p.name for p in Path("/dev").glob("*")
                  if p.name.startswith(("dxrt", "dxnn", "deepx", "rfcomm"))]
    return {"machine_architecture": platform.machine(), "kernel": platform.release(),
            "pci_devices": pci, "device_name_candidates": candidates,
            "bluetooth_adapter_names": bluetooth, "tool_presence": tools,
            "p1_status": "BLOCKED", "actual_npu_inference": "NOT TESTED",
            "explanation": "읽기 전용 조사이며 실제 키트 구성·사용권·SDK·추론 로그를 확인해야 합니다."}


def main():
    parser = argparse.ArgumentParser(description="P1 NPU·Bluetooth 읽기 전용 조사")
    parser.add_argument("--output", type=Path, required=True, help="존재하지 않는 보고서 파일")
    args = parser.parse_args()
    report = collect()
    try:
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
    except OSError:
        parser.exit(1, "새 보고서 파일을 만들 수 없습니다. 경로·권한·기존 파일을 확인하세요.\n")
    print("P1 조사 기록 완료. 실제 NPU 추론·Bluetooth 수신은 NOT TESTED입니다.")


if __name__ == "__main__":
    main()
