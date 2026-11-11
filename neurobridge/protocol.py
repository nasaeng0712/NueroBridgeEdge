"""길이 접두 JSON 프로토콜. 손상된 연결은 호출자가 폐기해야 한다."""

import json
import struct

MAX_BYTES = 16384
TYPES = {"HELLO", "HELLO_ACK", "SYNC_REQ", "SYNC_RESP", "TRIAL_CUE", "TRIAL_ACK",
         "HEARTBEAT", "PREDICTION", "INTENT", "RENDER_ACK", "ERROR"}


class ProtocolError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("JSON 필드가 중복되었습니다.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ProtocolError("유한하지 않은 JSON 숫자입니다.")


def validate_envelope(message):
    if not isinstance(message, dict):
        raise ProtocolError("메시지는 JSON 객체여야 합니다.")
    if message.get("schema_version") != "1.0" or (not isinstance(message.get("type"), str) or message["type"] not in TYPES):
        raise ProtocolError("지원하지 않는 스키마 또는 메시지 유형입니다.")
    for key in ("session_id", "boot_id", "sync_epoch", "connection_epoch"):
        if not isinstance(message.get(key), str) or not message[key]:
            raise ProtocolError("세션·부팅·동기·연결 식별자가 필요합니다.")


def encode(message):
    validate_envelope(message)
    try:
        body = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise ProtocolError("메시지를 JSON으로 변환할 수 없습니다.") from error
    if not 0 < len(body) <= MAX_BYTES:
        raise ProtocolError("메시지 크기가 허용 범위를 벗어났습니다.")
    return struct.pack(">I", len(body)) + body


class Decoder:
    def __init__(self):
        self.buffer = bytearray()
        self.expected = None
        self.failed = False

    def feed(self, chunk):
        if self.failed:
            raise ProtocolError("손상된 연결의 디코더를 재사용할 수 없습니다.")
        messages = []
        # 한 번에 큰 chunk가 와도 내부 버퍼는 프레임 상한을 넘지 않도록 처리한다.
        view = memoryview(chunk)
        cursor = 0
        try:
            while cursor < len(view):
                target = 4 if self.expected is None else self.expected
                count = min(target - len(self.buffer), len(view) - cursor)
                self.buffer.extend(view[cursor:cursor + count])
                cursor += count
                if len(self.buffer) < target:
                    continue
                if self.expected is None:
                    self.expected = struct.unpack(">I", self.buffer)[0]
                    self.buffer.clear()
                    if not 0 < self.expected <= MAX_BYTES:
                        raise ProtocolError("프레임 길이가 허용 범위를 벗어났습니다.")
                else:
                    message = json.loads(self.buffer.decode("utf-8"),
                                         object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                    validate_envelope(message)
                    messages.append(message)
                    self.buffer.clear()
                    self.expected = None
        except (ValueError, TypeError, UnicodeError, RecursionError) as error:
            self.failed = True
            self.buffer.clear()
            raise ProtocolError("TCP 프레임의 형식이 잘못되었습니다.") from error
        return messages

    def eof(self):
        if self.buffer or self.expected is not None:
            self.failed = True
            raise ProtocolError("연결 종료 시 미완성 프레임이 남았습니다.")
