"""인과성·분할 처리 일치와 수학적으로 정한 특징 벡터를 검사한다."""

import unittest
import numpy as np
from neurobridge.signal import CausalFilter, Normalizer, features, preprocessing_manifest


class SignalTests(unittest.TestCase):
    def test_chunk_processing_matches_continuous(self):
        values = np.random.default_rng(17).normal(size=2048)
        reference = CausalFilter().process(values, "a")
        streaming = CausalFilter()
        actual = np.concatenate([streaming.process(chunk, "a") for chunk in np.array_split(values, 17)])
        np.testing.assert_allclose(actual, reference, atol=1e-12)

    def test_future_values_do_not_affect_prefix(self):
        values = np.arange(128, dtype=float)
        first = CausalFilter().process(values, "a")
        extended = CausalFilter().process(np.concatenate([values, np.ones(100) * 999]), "a")
        np.testing.assert_array_equal(first, extended[:128])

    def test_segment_reset(self):
        filt = CausalFilter()
        filt.process(np.ones(20), "a")
        np.testing.assert_array_equal(filt.process(np.ones(20), "b"), CausalFilter().process(np.ones(20), "b"))

    def test_constant_and_known_sine(self):
        zero = features(np.zeros(1024))
        self.assertTrue(np.isfinite(zero).all())
        self.assertEqual(zero[5], 0)
        self.assertEqual(zero[6], 0)
        x = np.sin(2 * np.pi * 10 * np.arange(1024) / 512)
        result = features(x)
        self.assertAlmostEqual(float(result[1]), np.log(0.5), places=6)
        self.assertAlmostEqual(float(result[6]), np.sqrt(0.5), places=6)
        self.assertGreater(result[1], result[0])
        self.assertEqual(result.shape, (12,))

    def test_bad_data_and_training_only_normalization(self):
        with self.assertRaises(ValueError):
            features([0, np.nan])
        with self.assertRaises(ValueError):
            Normalizer.fit(np.zeros((2, 12)), "final_test")
        norm = Normalizer.fit(np.array([np.zeros(12), np.ones(12) * 2]), "training")
        np.testing.assert_array_equal(norm.transform(np.ones(12)), np.zeros(12))

    def test_preprocessing_hash_changes_with_filter(self):
        _, one = preprocessing_manifest(CausalFilter())
        _, two = preprocessing_manifest(CausalFilter(high_hz=30))
        self.assertNotEqual(one, two)
