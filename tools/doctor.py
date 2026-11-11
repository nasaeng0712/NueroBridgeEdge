"""외부 패키지 없이 개발 머신의 기본 사양을 확인한다."""

import json
import os
import platform
import shutil
import subprocess
import sys


def memory_bytes():
    """운영체제가 제공하는 물리 메모리를 조회하며 실패하면 미확인으로 둔다."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "total_physical", "available_physical", "total_page",
                        "available_page", "total_virtual", "available_virtual", "extended",
                    )],
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total_physical
            return None
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True,
                text=True, check=True, timeout=5,
            )
            return int(result.stdout.strip())
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError, subprocess.SubprocessError):
        return None


def collect_report():
    """사용자명, 호스트명, 환경변수나 인증정보는 수집하지 않는다."""
    return {
        "진단대상": "이 명령을 실행한 머신",
        "운영체제": platform.system(),
        "운영체제버전": platform.release(),
        "아키텍처": platform.machine(),
        "논리CPU수": os.cpu_count(),
        "물리메모리바이트": memory_bytes(),
        "현재작업디스크여유바이트": shutil.disk_usage(os.getcwd()).free,
        "Python버전": platform.python_version(),
        "실행도구": {name: shutil.which(name) is not None for name in ("git", "node")},
        "실제BCI검증": "NOT TESTED",
        "실제NPU검증": "NOT TESTED",
        "설명": "하드웨어 연결 여부와 EEG 해독 능력은 이 진단으로 판단하지 않습니다.",
    }


def main():
    try:
        print(json.dumps(collect_report(), ensure_ascii=False, indent=2))
    except OSError:
        print("진단 실패: 작업 디스크 정보를 조회할 수 없습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
