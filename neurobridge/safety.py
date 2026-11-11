"""수신 명령의 식별·신선도·순서를 확인하며 장애 후 명시적 재준비를 요구한다."""

from dataclasses import dataclass
import math

from .protocol import ProtocolError, validate_envelope

ACTIONS = {"STOP", "ROTATE_CW", "ROTATE_CCW", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"}


@dataclass(frozen=True)
class Sync:
    edge_minus_pc_ns: float
    uncertainty_ns: float

    @classmethod
    def estimate(cls, t1, t2, t3, t4):
        if any(type(t) is not int or t < 0 for t in (t1, t2, t3, t4)) or t4 < t1 or t3 < t2:
            raise ValueError("시계 동기 타임스탬프가 잘못되었습니다.")
        delay = (t4 - t1) - (t3 - t2)
        if delay < 0:
            raise ValueError("시계 동기 왕복 지연이 음수입니다.")
        return cls(((t2 - t1) + (t3 - t4)) / 2, delay / 2)


@dataclass(frozen=True)
class Policy:
    # 합성 개발 정책이며 실기 시험 전 별도 동결해야 한다.
    ttl_ns: int = 1_000_000_000
    heartbeat_ns: int = 1_000_000_000
    max_uncertainty_ns: int = 20_000_000
    max_sync_age_ns: int = 5_000_000_000

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("안전 정책 시간은 양의 정수여야 합니다.")


class SafetyReceiver:
    def __init__(self, identity, model_id, model_sha256, policy=None):
        self.identity = dict(identity)
        if set(self.identity) != {"session_id", "boot_id", "sync_epoch", "connection_epoch"}:
            raise ValueError("수신 연결 식별자가 완전하지 않습니다.")
        if any(not isinstance(v, str) or not v for v in self.identity.values()):
            raise ValueError("수신 연결 식별자가 비어 있습니다.")
        self.model_id, self.model_sha256 = model_id, model_sha256
        self.policy = policy or Policy()
        self.action = "STOP"
        self.status, self.reason = "INVALID", "NOT_READY"
        self.sync = None
        self.armed = False
        self.last_heartbeat_ns = None
        self.last_created_ns = -1
        self.seen = set()
        self.fresh_after_ns = -1
        self.last_source_edge_ns = None

    def stop(self, reason):
        self.action, self.status, self.reason = "STOP", "INVALID", reason
        self.armed = False
        return self.action

    def prepare(self, sync, now_ns, *, recording_ok, model_ok, fresh_samples):
        self.stop("NOT_READY")
        if not (recording_ok and model_ok and fresh_samples):
            return False
        if (not isinstance(sync, Sync) or not math.isfinite(sync.edge_minus_pc_ns)
                or not math.isfinite(sync.uncertainty_ns)
                or not 0 <= sync.uncertainty_ns <= self.policy.max_uncertainty_ns):
            return False
        self.last_source_edge_ns = None
        self.sync = sync
        self.sync_at_ns = now_ns
        self.fresh_after_ns = now_ns + sync.edge_minus_pc_ns + sync.uncertainty_ns
        self.last_heartbeat_ns = now_ns
        self.armed = True
        return True

    def tick(self, now_ns):
        if self.armed and self.last_source_edge_ns is not None:
            age_upper = now_ns - self.last_source_edge_ns + self.sync.edge_minus_pc_ns + self.sync.uncertainty_ns
            if age_upper > self.policy.ttl_ns:
                return self.stop("STALE_INTENT")
        if self.armed and (now_ns < self.sync_at_ns
                or now_ns - self.sync_at_ns > self.policy.max_sync_age_ns
                or now_ns - self.last_heartbeat_ns > self.policy.heartbeat_ns):
            self.stop("TIMEOUT")
        return self.action

    def receive(self, message, now_ns):
        self.tick(now_ns)
        try:
            validate_envelope(message)
            if any(message[k] != value for k, value in self.identity.items()):
                raise ProtocolError("수신 세션 또는 연결 식별자가 다릅니다.")
            if message["type"] == "HEARTBEAT":
                if self.armed:
                    self.last_heartbeat_ns = now_ns
                return self.action
            if message["type"] != "INTENT" or not self.armed:
                raise ProtocolError("준비되지 않은 연결 또는 잘못된 명령 유형입니다.")
            self._validate_intent(message)
            created = message["edge_created_mono_ns"]
            source = message["source_window_end_edge_ns"]
            age = now_ns - (source - self.sync.edge_minus_pc_ns)
            if (source <= self.fresh_after_ns or created < source
                    or age - self.sync.uncertainty_ns < 0
                    or age + self.sync.uncertainty_ns > self.policy.ttl_ns):
                raise ProtocolError("명령의 신선도를 확인할 수 없습니다.")
            if message["intent_id"] in self.seen or created <= self.last_created_ns:
                raise ProtocolError("중복 또는 역순 명령입니다.")
            if len(self.seen) >= 100000:
                raise ProtocolError("수신 추적 용량이 초과되었습니다. 새 연결이 필요합니다.")
            self.seen.add(message["intent_id"])
            self.last_created_ns = created
            self.last_source_edge_ns = source
            self.status = message["decision_status"]
            self.reason = message["error_code"]
            self.action = message["action"] if self.status == "VALID" else "STOP"
            if self.status != "VALID":
                self.armed = False
            return self.action
        except (ProtocolError, KeyError, TypeError, ValueError, OverflowError):
            return self.stop("ERROR_SCHEMA_OR_SAFETY")

    def _validate_intent(self, message):
        for key in ("stream_id", "segment_id", "trial_id", "window_id", "prediction_id",
                    "intent_id", "model_id", "model_sha256", "command_generation_policy_version"):
            if not isinstance(message.get(key), str) or not message[key]:
                raise ProtocolError("필수 식별자가 없습니다.")
        if message["model_id"] != self.model_id or message["model_sha256"] != self.model_sha256:
            raise ProtocolError("모델 식별자가 일치하지 않습니다.")
        bounds = message.get("input_sample_range")
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) is not int for v in bounds) or not 0 <= bounds[0] < bounds[1]):
            raise ProtocolError("입력 샘플 범위가 잘못되었습니다.")
        if message.get("decision_status") not in {"VALID", "REJECT", "INVALID"}:
            raise ProtocolError("판정 상태가 잘못되었습니다.")
        if message.get("action") not in ACTIONS:
            raise ProtocolError("지원하지 않는 동작입니다.")
        if message["decision_status"] != "VALID" and message["action"] != "STOP":
            raise ProtocolError("거부·오류 상태에는 정지만 허용됩니다.")
        score = message.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ProtocolError("점수가 유한한 숫자가 아닙니다.")
        for key in ("source_window_end_edge_ns", "edge_created_mono_ns"):
            if type(message.get(key)) is not int or message[key] < 0:
                raise ProtocolError("명령 시각이 잘못되었습니다.")
        if "error_code" not in message or (message["error_code"] is not None and not isinstance(message["error_code"], str)):
            raise ProtocolError("오류 코드가 잘못되었습니다.")
