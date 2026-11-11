"""ThinkGear 스트림을 검사한다. 전송 원본은 호출자가 먼저 보존해야 한다."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Packet:
    offset: int
    payload: bytes
    raw_samples: tuple[int, ...]
    poor_signal: int | None


class ThinkGearParser:
    def __init__(self):
        self.buffer = bytearray()
        self.offset = 0
        self.errors = 0

    def reset(self):
        """재연결시 미완성 프레임을 버려 구간 혼합을 방지한다."""
        self.offset += len(self.buffer)
        self.buffer.clear()

    def _discard(self, count):
        del self.buffer[:count]
        self.offset += count

    def feed(self, chunk):
        self.buffer.extend(chunk)
        packets = []
        while self.buffer:
            start = self.buffer.find(b"\xaa\xaa")
            if start < 0:
                self._discard(len(self.buffer) - int(self.buffer[-1] == 0xAA))
                break
            if start:
                self._discard(start)
            if len(self.buffer) < 3:
                break
            length = self.buffer[2]
            if length > 169:
                self.errors += 1
                self._discard(1)
                continue
            size = length + 4
            if len(self.buffer) < size:
                break
            payload = bytes(self.buffer[3:3 + length])
            if self.buffer[size - 1] != (~sum(payload) & 0xFF):
                self.errors += 1
                self._discard(1)
                continue
            try:
                raw, quality = parse_rows(payload)
                packets.append(Packet(self.offset, payload, tuple(raw), quality))
            except ValueError:
                self.errors += 1
            self._discard(size)
        return packets


def parse_rows(payload):
    samples, quality, index = [], None, 0
    while index < len(payload):
        extended = 0
        while index < len(payload) and payload[index] == 0x55:
            extended += 1
            index += 1
        if index >= len(payload):
            raise ValueError("확장 코드 뒤에 데이터 코드가 없습니다.")
        code = payload[index]
        index += 1
        length = 1
        if code >= 0x80:
            if index >= len(payload):
                raise ValueError("데이터 길이가 없습니다.")
            length = payload[index]
            index += 1
        if index + length > len(payload):
            raise ValueError("데이터 행이 잘렸습니다.")
        value = payload[index:index + length]
        index += length
        if not extended and code == 0x80:
            if length != 2:
                raise ValueError("원시 EEG 데이터 길이는 2여야 합니다.")
            samples.append(int.from_bytes(value, "big", signed=True))
        elif not extended and code == 0x02:
            quality = value[0]
    return samples, quality


def encode_raw(sample):
    """시험용 패킷을 생성한다. 실제 EEG 측정값을 생성하지 않는다."""
    payload = b"\x80\x02" + sample.to_bytes(2, "big", signed=True)
    return b"\xaa\xaa\x04" + payload + bytes([~sum(payload) & 0xFF])
