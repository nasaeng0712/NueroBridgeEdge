"""인과적 수신 순서로 품질과 구간 경계를 검사하고 윈도우를 구성한다."""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Window:
    segment_id: str
    sample_start: int
    sample_stop: int
    samples: tuple[float, ...]


class WindowBuilder:
    def __init__(self, size=1024, hop=128):
        if type(size) is not int or type(hop) is not int or not 0 < hop <= size:
            raise ValueError("윈도우와 이동 크기는 양의 정수이며 이동 크기는 윈도우 이하여야 합니다.")
        self.size, self.hop = size, hop
        self.buffer = deque(maxlen=size)
        self.segment_id = None
        self.next_index = None
        self.valid_count = 0
        self.status = "INVALID"
        self.reason = "WARMUP"

    def reset(self, reason):
        self.buffer.clear()
        self.valid_count = 0
        self.status, self.reason = "INVALID", reason

    def feed(self, segment_id, sample_index, sample, poor_signal):
        if not isinstance(segment_id, str) or not segment_id or type(sample_index) is not int or sample_index < 0:
            self.reset("INVALID_ID")
            raise ValueError("구간 ID 또는 샘플 순번이 잘못되었습니다.")
        if segment_id != self.segment_id or sample_index != self.next_index:
            self.reset("DISCONTINUITY")
        self.segment_id, self.next_index = segment_id, sample_index + 1
        if type(sample) not in (int, float) or not math.isfinite(sample) or not -32768 <= sample <= 32767:
            self.reset("INVALID_SAMPLE")
            return None
        # 실측 기준 확정 전 보수적인 개발 정책: 품질 미상 또는 비영 값은 모두 거부한다.
        if type(poor_signal) is not int or poor_signal != 0:
            self.reset("POOR_SIGNAL")
            self.status = "REJECT"
            return None
        self.buffer.append(float(sample))
        self.valid_count += 1
        if len(self.buffer) < self.size:
            self.status, self.reason = "INVALID", "WARMUP"
            return None
        self.status, self.reason = "VALID", None
        if (self.valid_count - self.size) % self.hop:
            return None
        return Window(segment_id, sample_index + 1 - self.size, sample_index + 1, tuple(self.buffer))
