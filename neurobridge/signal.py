"""연결 구간마다 상태를 초기화하는 인과 SOS 필터와 고정 순서 특징."""

import hashlib
import json

import numpy as np
from scipy.signal import butter, sosfilt

FEATURE_ORDER = ("theta_logpower", "alpha_logpower", "low_beta_logpower", "high_beta_logpower",
                 "total_logpower", "mean", "std", "peak_to_peak", "line_length",
                 "zero_crossing_rate", "hjorth_mobility", "spectral_entropy")


class CausalFilter:
    def __init__(self, sample_rate=512, low_hz=1, high_hz=40, order=4):
        # 이 기본값은 합성 개발용이며 사용자 EEG의 최종 필터로 동결한 값이 아니다.
        self.sample_rate = sample_rate
        self.sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
        self.segment_id = None
        self.state = np.zeros((len(self.sos), 2))
        self.sample_count = 0

    def process(self, samples, segment_id):
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or not segment_id:
            raise ValueError("필터 입력은 구간 ID와 유한한 1차원 신호여야 합니다.")
        if segment_id != self.segment_id:
            self.state.fill(0)
            self.sample_count = 0
            self.segment_id = segment_id
        if len(values) == 0:
            return values.copy()
        result, self.state = sosfilt(self.sos, values, zi=self.state)
        self.sample_count += len(values)
        return result

    def manifest(self):
        return {"sample_rate": self.sample_rate, "kind": "causal_sos", "sos": self.sos.tolist(), "initial_state": "zeros",
                "segment_reset": True, "status": "SYNTHETIC_DEVELOPMENT"}


def features(samples, sample_rate=512, epsilon=1e-12):
    values = np.asarray(samples, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or not np.isfinite(sample_rate) or sample_rate <= 0
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("특징 입력과 표본 주파수·epsilon이 잘못되었습니다.")
    # 평균 제거·창 함수 없는 단측 periodogram의 bin별 파워를 적분한다.
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2 / len(values) ** 2
    power[1:-1 if len(values) % 2 == 0 else None] *= 2
    frequency = np.fft.rfftfreq(len(values), 1 / sample_rate)
    bands = ((4, 8), (8, 13), (13, 20), (20, 30), (4, 40))
    band_features = [np.log(power[(frequency >= lo) & (frequency < hi)].sum() + epsilon)
                     for lo, hi in bands]
    variance = np.var(values)
    diff = np.diff(values)
    probabilities = power / (power.sum() + epsilon)
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon)) / np.log(len(power))
    rest = [values.mean(), values.std(), np.ptp(values), np.abs(diff).mean(),
            np.count_nonzero(np.signbit(values[1:]) != np.signbit(values[:-1])) / len(diff),
            np.sqrt(np.var(diff) / (variance + epsilon)), entropy]
    output = np.asarray(band_features + rest, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("특징 계산 결과가 유한하지 않습니다.")
    return output


def preprocessing_manifest(causal_filter):
    manifest = {"filter": causal_filter.manifest(), "feature_order": FEATURE_ORDER,
                "bands_hz_half_open": [[4, 8], [8, 13], [13, 20], [20, 30], [4, 40]],
                "epsilon": 1e-12, "sample_rate": causal_filter.sample_rate,
                "spectrum": "demean_rectangular_onesided_power_per_bin",
                "line_length": "mean_absolute_first_difference", "std": "population",
                "entropy": "normalized_natural_log_full_one_sided_spectrum",
                "zero_crossings": "signbit_changes_divided_by_n_minus_one"}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(payload).hexdigest()


class Normalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if (self.mean.shape != (12,) or self.std.shape != (12,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError("정규화 통계는 12개 유한 값과 양의 표준편차여야 합니다.")

    @classmethod
    def fit(cls, rows, split):
        if split != "training":
            raise ValueError("정규화 통계는 학습 분할에서만 계산할 수 있습니다.")
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 12 or len(rows) < 2 or not np.isfinite(rows).all():
            raise ValueError("학습 특징 행렬이 잘못되었습니다.")
        return cls(rows.mean(axis=0), np.maximum(rows.std(axis=0), 1e-12))

    def transform(self, row):
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (12,) or not np.isfinite(row).all():
            raise ValueError("정규화할 특징 벡터가 잘못되었습니다.")
        output = ((row - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("정규화 출력이 유한하지 않습니다.")
        return output
