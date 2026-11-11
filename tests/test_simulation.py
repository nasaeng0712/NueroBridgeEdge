"""합성 전체 경로와 Godot 장면 변환을 별도로 검사한다."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from neurobridge.simulation import run


class SimulationTests(unittest.TestCase):
    def test_capture_features_tcp_and_godot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "simulation"
            manifest = run(root)
            self.assertEqual(manifest["sample_count"], 2048)
            self.assertEqual(manifest["window_count"], 9)
            self.assertEqual(manifest["hardware_status"], "NOT TESTED")
            commands = [json.loads(s) for s in (root / "scene_commands.jsonl").read_text().splitlines()]
            self.assertEqual(len(commands), 100)
            godot = shutil.which("godot") or shutil.which("godot4")
            if not godot:
                self.skipTest("합성 Python 경로 통과, Godot 미설치로 장면 검증 미실시")
            env = dict(os.environ, XDG_DATA_HOME=str(root / "data"),
                       XDG_CACHE_HOME=str(root / "cache"), XDG_CONFIG_HOME=str(root / "config"))
            project = Path(__file__).resolve().parents[1] / "renderer"
            for kind in ("box", "sphere"):
                report = root / f"render-{kind}.jsonl"
                result = subprocess.run([godot, "--headless", "--path", str(project), "--",
                                         "--commands", str(root / "scene_commands.jsonl"),
                                         "--report", str(report), "--object", kind, "--test"],
                                        capture_output=True, text=True, timeout=20, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [json.loads(s) for s in report.read_text().splitlines()]
                self.assertEqual(len(rows), 100)
                for command, row in zip(commands, rows):
                    self.assertTrue(row["accepted"])
                    self.assertEqual(row["source_intent_id"], command["source_intent_id"])
                    self.assertEqual(row["action"], command["action"])
                self.assertLess(rows[0]["actual_object_transform"]["rotation"][1], 0)
                # 움직임과 STOP을 단순 문자열 일치가 아닌 변환값으로 확인한다.
                self.assertEqual(rows[0]["actual_object_transform"], rows[1]["actual_object_transform"])
                self.assertLess(rows[3]["actual_object_transform"]["position"][0], 0)
            from neurobridge.live_demo import run as live_run
            live = live_run(root / "scene_commands.jsonl", root / "live", headless=True)
            self.assertEqual(live["command_count"], 100)
            self.assertTrue(live["disconnect_stop"])

    def test_godot_watchdog_and_invalid_frame_stop(self):
        import socket
        import struct
        godot = shutil.which("godot") or shutil.which("godot4")
        if not godot:
            self.skipTest("Godot 미설치")
        project = Path(__file__).resolve().parents[1] / "renderer"
        for fault in ("TIMEOUT", "ERROR_SCHEMA"):
            with tempfile.TemporaryDirectory() as tmp, socket.socket() as server:
                root = Path(tmp)
                server.bind(("127.0.0.1", 0))
                server.listen(1)
                server.settimeout(10)
                env = dict(os.environ, XDG_DATA_HOME=str(root / "data"),
                           XDG_CACHE_HOME=str(root / "cache"), XDG_CONFIG_HOME=str(root / "config"))
                process = subprocess.Popen([godot, "--headless", "--path", str(project), "--",
                                            "--port", str(server.getsockname()[1]), "--report",
                                            str(root / "render.jsonl"), "--test"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                try:
                    connection, _ = server.accept()
                    with connection:
                        if fault == "ERROR_SCHEMA":
                            connection.sendall(struct.pack(">I", 16385))
                        self.assertEqual(process.wait(timeout=5), 1)
                    rows = [json.loads(line) for line in (root / "render.jsonl").read_text().splitlines()]
                    self.assertEqual(rows[-1]["action"], "STOP")
                    self.assertEqual(rows[-1]["reason"], fault)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
