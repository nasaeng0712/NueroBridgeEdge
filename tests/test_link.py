"""가짜 소켓으로 연결 장애·재연결·부분 프레임 보존을 검사한다."""

import json
from pathlib import Path
import tempfile
import unittest

from neurobridge.capture import CaptureError, collect
from neurobridge.thinkgear import encode_raw


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value


class FakeSocket:
    def __init__(self, chunks, clock):
        self.chunks, self.clock, self.closed = iter(chunks), clock, False

    def settimeout(self, timeout):
        pass

    def recv(self, size):
        self.clock.value += 1
        return next(self.chunks, b"")

    def close(self):
        self.closed = True


class LinkTests(unittest.TestCase):
    def test_reconnect_and_preserve_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, clock = Path(tmp) / "capture", Clock()
            first = FakeSocket([encode_raw(1)[:5], b""], clock)
            second = FakeSocket([encode_raw(2), encode_raw(3)], clock)
            sockets = iter([first, second])
            report = collect(root, lambda: next(sockets), 4, clock=clock, source="SYNTHETIC")
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["reconnects"], 1)
            self.assertTrue(first.closed and second.closed)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["source"], "SYNTHETIC")
            rows = [json.loads(s) for s in (root / "chunks.jsonl").read_text().splitlines()]
            self.assertNotEqual(rows[0]["segment_id"], rows[1]["segment_id"])

    def test_connect_failure_is_bounded(self):
        calls = []
        def connector():
            calls.append(1)
            raise OSError("연결 실패")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CaptureError):
                collect(Path(tmp) / "capture", connector, max_reconnects=2, source="SYNTHETIC")
        self.assertEqual(len(calls), 3)
