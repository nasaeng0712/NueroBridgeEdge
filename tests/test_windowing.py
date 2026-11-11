"""구간 혼합, 품질 미상과 손상 입력에서 윈도우 생성이 차단되는지 확인한다."""

import unittest
from neurobridge.windowing import WindowBuilder


class WindowTests(unittest.TestCase):
    def test_default_window_and_hop(self):
        builder = WindowBuilder()
        windows = [w for i in range(1280) if (w := builder.feed("a", i, i, 0))]
        self.assertEqual([(w.sample_start, w.sample_stop) for w in windows],
                         [(0, 1024), (128, 1152), (256, 1280)])
        self.assertEqual(windows[0].samples[-1], 1023)

    def test_gap_and_new_segment_require_fresh_window(self):
        builder = WindowBuilder(4, 2)
        for i in range(3):
            self.assertIsNone(builder.feed("a", i, 1, 0))
        self.assertIsNone(builder.feed("a", 4, 1, 0))
        self.assertIsNone(builder.feed("b", 5, 1, 0))
        self.assertIsNone(builder.feed("b", 6, 1, 0))
        self.assertIsNone(builder.feed("b", 7, 1, 0))
        self.assertEqual(builder.feed("b", 8, 1, 0).sample_start, 5)

    def test_quality_and_nonfinite_clear_history(self):
        for sample, quality in ((1, None), (1, 200), (float("nan"), 0),
                                (float("inf"), 0), (32768, 0), (True, 0)):
            builder = WindowBuilder(2, 1)
            builder.feed("a", 0, 1, 0)
            self.assertIsNone(builder.feed("a", 1, sample, quality))
            self.assertNotEqual(builder.status, "VALID")
            self.assertIsNone(builder.feed("a", 2, 1, 0))
            self.assertIsNotNone(builder.feed("a", 3, 1, 0))

    def test_invalid_configuration(self):
        for args in ((0, 1), (4, 5), (True, 1)):
            with self.assertRaises(ValueError):
                WindowBuilder(*args)
