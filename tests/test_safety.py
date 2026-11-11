"""합성 시계와 장애 주입으로 수신 측 정지 계약을 확인한다."""

import unittest

from neurobridge.safety import Policy, SafetyReceiver, Sync
from test_protocol import IDENTITY, heartbeat


def intent(**overrides):
    result = dict(IDENTITY, schema_version="1.0", type="INTENT", stream_id="stream",
                  segment_id="segment", trial_id="trial", window_id="window",
                  prediction_id="prediction", intent_id="intent", model_id="model",
                  model_sha256="a" * 64, input_sample_range=[0, 1024], decision_status="VALID",
                  action="ROTATE_CW", score=0.8, source_window_end_edge_ns=200,
                  edge_created_mono_ns=210, error_code=None,
                  command_generation_policy_version="synthetic-v1")
    result.update(overrides)
    return result


def ready():
    receiver = SafetyReceiver(IDENTITY, "model", "a" * 64)
    assert receiver.prepare(Sync(0, 0), 100, recording_ok=True, model_ok=True, fresh_samples=True)
    return receiver


class SafetyTests(unittest.TestCase):
    def test_valid_and_duplicate_stops(self):
        receiver = ready()
        self.assertEqual(receiver.receive(intent(), 220), "ROTATE_CW")
        self.assertEqual(receiver.receive(intent(), 230), "STOP")
        self.assertFalse(receiver.armed)

    def test_identity_manifest_nan_enum_and_bounds(self):
        cases = [dict(session_id="other"), dict(boot_id="other"), dict(connection_epoch="old"),
                 dict(sync_epoch="old"), dict(model_sha256="b" * 64), dict(score=float("nan")),
                 dict(action="UNKNOWN"), dict(input_sample_range=[2, 1]),
                 dict(decision_status="REST"), dict(source_window_end_edge_ns=True)]
        for changes in cases:
            with self.subTest(changes=changes):
                receiver = ready()
                self.assertEqual(receiver.receive(intent(**changes), 220), "STOP")
                self.assertFalse(receiver.armed)

    def test_reject_invalid_are_not_rest(self):
        for status in ("REJECT", "INVALID"):
            receiver = ready()
            self.assertEqual(receiver.receive(intent(decision_status=status, action="STOP"), 220), "STOP")
            self.assertEqual(receiver.status, status)
            self.assertFalse(receiver.armed)

    def test_stale_future_and_reverse(self):
        self.assertEqual(ready().receive(intent(), 1_000_000_500), "STOP")
        self.assertEqual(ready().receive(intent(source_window_end_edge_ns=300, edge_created_mono_ns=310), 220), "STOP")
        receiver = ready()
        receiver.receive(intent(), 220)
        self.assertEqual(receiver.receive(intent(intent_id="second", edge_created_mono_ns=209), 230), "STOP")

    def test_heartbeat_cannot_keep_stale_movement_or_rearm(self):
        receiver = ready()
        receiver.receive(intent(), 220)
        receiver.receive(heartbeat(), 900_000_000)
        self.assertEqual(receiver.tick(1_000_000_201), "STOP")
        receiver.receive(heartbeat(), 1_000_000_202)
        self.assertFalse(receiver.armed)

    def test_fault_requires_new_preparation_and_fresh_window(self):
        for fault in ("BT_DISCONNECTED", "RECORDING_FAILURE", "NPU_ERROR", "TCP_EOF"):
            for _ in range(10):
                receiver = ready()
                receiver.receive(intent(), 220)
                self.assertEqual(receiver.stop(fault), "STOP")
                self.assertEqual(receiver.receive(intent(intent_id="new"), 230), "STOP")
                self.assertFalse(receiver.prepare(Sync(0, 0), 240, recording_ok=False,
                                                 model_ok=True, fresh_samples=True))
                self.assertTrue(receiver.prepare(Sync(0, 0), 250, recording_ok=True,
                                                 model_ok=True, fresh_samples=True))
                self.assertEqual(receiver.receive(intent(intent_id="new"), 260), "STOP")

    def test_sync_estimate_and_uncertainty(self):
        self.assertEqual(Sync.estimate(100, 120, 130, 150), Sync(0, 20))
        with self.assertRaises(ValueError):
            Sync.estimate(100, 120, 150, 110)
        receiver = ready()
        self.assertFalse(receiver.prepare(Sync(0, float("nan")), 0,
                                         recording_ok=True, model_ok=True, fresh_samples=True))

    def test_unknown_sync_and_expired_sync_stop(self):
        receiver = SafetyReceiver(IDENTITY, "model", "a" * 64)
        self.assertEqual(receiver.receive(intent(), 220), "STOP")
        receiver = ready()
        self.assertEqual(receiver.tick(6_000_000_000), "STOP")
