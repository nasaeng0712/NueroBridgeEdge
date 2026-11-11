"""설계도 V01~V09 및 저장 실패·재생 경로의 합성 검증."""

import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from neurobridge.recording import Capture, verify
from neurobridge.thinkgear import ThinkGearParser, encode_raw


def frame(payload):
    return b"\xaa\xaa" + bytes([len(payload)]) + payload + bytes([~sum(payload) & 255])


class ParserTests(unittest.TestCase):
    def test_signed_boundaries_and_multiple_frames(self):
        values = [-32768, -1, 0, 1, 32767, -21846]
        packets = ThinkGearParser().feed(b"".join(map(encode_raw, values)))
        self.assertEqual([p.raw_samples[0] for p in packets], values)

    def test_every_split_point(self):
        data = encode_raw(-123)
        for split in range(len(data) + 1):
            parser = ThinkGearParser()
            packets = parser.feed(data[:split]) + parser.feed(data[split:])
            self.assertEqual([p.raw_samples for p in packets], [(-123,)])

    def test_bad_checksum_length_and_noise_recover(self):
        parser = ThinkGearParser()
        bad = encode_raw(1)[:-1] + b"\x00"
        packets = parser.feed(b"noise\xaa\xaa\xff" + bad + encode_raw(2))
        self.assertEqual([p.raw_samples for p in packets], [(2,)])
        self.assertGreaterEqual(parser.errors, 2)

    def test_unknown_extended_rows_and_quality(self):
        packets = ThinkGearParser().feed(frame(b"\x55\x80\x02\x12\x34\x02\x00\x90\x01\xff"))
        self.assertEqual(packets[0].raw_samples, ())
        self.assertEqual(packets[0].poor_signal, 0)

    def test_malformed_rows_rejected(self):
        for payload in (b"\x55", b"\x80", b"\x80\x01\x00", b"\x80\x02\x00"):
            parser = ThinkGearParser()
            self.assertEqual(parser.feed(frame(payload)), [])
            self.assertEqual(parser.errors, 1)

    def test_reconnect_discards_incomplete_frame(self):
        parser = ThinkGearParser()
        parser.feed(encode_raw(1)[:5])
        parser.reset()
        self.assertEqual(parser.feed(encode_raw(2))[0].raw_samples, (2,))

    def test_arbitrary_bytes_do_not_crash(self):
        rng = random.Random(42)
        parser = ThinkGearParser()
        for _ in range(1000):
            parser.feed(bytes(rng.randrange(256) for _ in range(rng.randrange(64))))
        self.assertLessEqual(len(parser.buffer), 172)


class CaptureTests(unittest.TestCase):
    def test_record_replay_reconnect_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            capture = Capture(root)
            capture.ingest(encode_raw(1)[:5])
            capture.reconnect()
            capture.ingest(encode_raw(-2))
            capture.close()
            self.assertEqual(verify(root), 1)
            with (root / "transport.bin").open("ab") as file:
                file.write(b"x")
            with self.assertRaises(ValueError):
                verify(root)

    def test_storage_failure_prevents_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = Capture(Path(tmp) / "session")
            with patch.object(capture, "_write", side_effect=OSError("디스크 부족")):
                with self.assertRaises(OSError):
                    capture.ingest(encode_raw(4))
            self.assertTrue(capture.failed)
            self.assertEqual(capture.sample_index, 0)
            with self.assertRaises(RuntimeError):
                capture.ingest(encode_raw(5))
            capture.close()
            self.assertFalse((capture.root / "manifest.json").exists())

    def test_existing_session_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                Capture(tmp)

    def test_storage_failure_stops_receiver(self):
        from test_safety import ready, intent
        receiver = ready()
        self.assertEqual(receiver.receive(intent(), 220), "ROTATE_CW")
        with tempfile.TemporaryDirectory() as tmp:
            capture = Capture(Path(tmp) / "session", on_failure=receiver.stop)
            with patch.object(capture, "_write", side_effect=OSError("디스크 부족")):
                with self.assertRaises(OSError):
                    capture.ingest(encode_raw(1))
            self.assertEqual(receiver.action, "STOP")
            self.assertFalse(receiver.armed)
            capture.close()
