"""PC 내부 TCP로 Godot 합성 Scene API를 실행하고 명령별 응답을 확인한다."""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import time


def run(commands_path, output, *, headless=False, object_kind="box"):
    godot = shutil.which("godot") or shutil.which("godot4")
    if not godot:
        raise RuntimeError("Godot 4.x 실행 파일을 찾을 수 없습니다.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = [json.loads(line) for line in Path(commands_path).read_text(encoding="utf-8").splitlines()]
    if not commands:
        raise ValueError("합성 명령 파일이 비어 있습니다.")
    env = dict(os.environ, XDG_DATA_HOME=str(output / "data"),
               XDG_CACHE_HOME=str(output / "cache"), XDG_CONFIG_HOME=str(output / "config"))
    report = output / "render.jsonl"
    project = Path(__file__).resolve().parents[1] / "renderer"
    received = []
    timings = []
    with socket.socket() as server:
        server.settimeout(15)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        args = [godot]
        if headless:
            args.append("--headless")
        args += ["--path", str(project), "--", "--port", str(server.getsockname()[1]),
                 "--report", str(report), "--object", object_kind, "--test"]
        with (output / "godot.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(5)
                    # 로컬 파일 응답은 Godot의 실제 명령 적용 기록이다.
                    while not report.exists():
                        if process.poll() is not None:
                            raise RuntimeError("Godot이 보고서를 만들기 전에 종료되었습니다.")
                        time.sleep(0.005)
                    with report.open(encoding="utf-8") as events:
                        for command in commands:
                            body = json.dumps(command, ensure_ascii=False, allow_nan=False).encode()
                            started_ns = time.monotonic_ns()
                            connection.sendall(struct.pack(">I", len(body)) + body)
                            deadline = time.monotonic() + 3
                            line = ""
                            while not line:
                                line = events.readline()
                                if time.monotonic() > deadline or process.poll() is not None:
                                    raise RuntimeError("Godot 명령 적용 응답이 없거나 종료되었습니다.")
                                if not line:
                                    time.sleep(0.001)
                            row = json.loads(line)
                            if not row.get("accepted") or row.get("source_intent_id") != command["source_intent_id"]:
                                raise RuntimeError("Godot이 합성 명령을 거부했습니다.")
                            timings.append(time.monotonic_ns() - started_ns)
                            if not headless:
                                time.sleep(0.05)
                            received.append(row)
                    connection.shutdown(socket.SHUT_WR)
                result = process.wait(timeout=5)
                if result != 0:
                    raise RuntimeError("Godot TCP 실행이 오류로 종료되었습니다.")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[-1].get("reason") != "TCP_EOF" or rows[-1].get("action") != "STOP":
        raise RuntimeError("TCP 단절 후 Godot 정지 기록이 없습니다.")
    summary = {"source": "SYNTHETIC_SCENE_COMMANDS", "command_count": len(received),
               "disconnect_stop": True, "headless": headless,
               "roundtrip_apply_observed_p95_ms": sorted(timings)[max(0, int(len(timings) * .95) - 1)] / 1e6,
               "hardware_status": "NOT TESTED"}
    (output / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Godot 로컬 TCP 합성 명령 시연")
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="새 출력 폴더")
    parser.add_argument("--headless", action="store_true", help="창 없이 장면 로직 검사")
    args = parser.parse_args()
    try:
        report = run(args.commands, args.output, headless=args.headless)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Godot TCP 합성 시험 실패: {error}\n")
    print(f"Godot TCP 합성 PASS: {report['command_count']}명령, 연결 종료 후 STOP")


if __name__ == "__main__":
    main()
