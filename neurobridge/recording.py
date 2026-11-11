"""원본 바이트를 먼저 내구성 있게 기록하고 파싱하는 로컬 수집 경로."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import struct
import time
import uuid

from .thinkgear import ThinkGearParser


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Capture:
    def __init__(self, root, on_failure=None):
        self.on_failure = on_failure
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.parser = ThinkGearParser()
        self.session_id = str(uuid.uuid4())
        self.boot_id = str(uuid.uuid4())
        self.stream_id = str(uuid.uuid4())
        self.segment_id = str(uuid.uuid4())
        self.chunk_seq = self.sample_index = 0
        self.failed = False
        self.closed = False
        self.files = {}
        try:
            for name in ("transport.bin", "chunks.jsonl", "eeg.i16le", "samples.jsonl"):
                self.files[name] = (self.root / (name + ".partial")).open("xb")
        except OSError:
            for file in self.files.values():
                file.close()
            raise

    def _write(self, name, data):
        file = self.files[name]
        if file.write(data) != len(data):
            raise OSError("저장 데이터 일부가 기록되지 않았습니다.")
        file.flush()
        os.fsync(file.fileno())

    def _json(self, name, value):
        self._write(name, (json.dumps(value, ensure_ascii=False) + "\n").encode())

    def ingest(self, chunk):
        if self.failed or self.closed:
            raise RuntimeError("수집이 중단되어 새 데이터를 처리할 수 없습니다.")
        received_ns = time.monotonic_ns()
        offset = self.files["transport.bin"].tell()
        try:
            self._write("transport.bin", chunk)
            self._json("chunks.jsonl", {
                "session_id": self.session_id, "boot_id": self.boot_id,
                "stream_id": self.stream_id, "segment_id": self.segment_id,
                "chunk_seq": self.chunk_seq, "offset": offset, "len": len(chunk),
                "edge_mono_ns": received_ns,
                "edge_wall_utc": datetime.now(timezone.utc).isoformat(),
                "durability": "fsync_completed", "missing_sample_count": None,
            })
            packets = self.parser.feed(chunk)
            for packet in packets:
                for sample in packet.raw_samples:
                    self._write("eeg.i16le", struct.pack("<h", sample))
                    self._json("samples.jsonl", {
                        "segment_id": self.segment_id, "sample_index": self.sample_index,
                        "packet_offset": packet.offset, "chunk_seq": self.chunk_seq,
                    })
                    self.sample_index += 1
            self.chunk_seq += 1
            return packets
        except (OSError, ValueError):
            self.failed = True
            if self.on_failure is not None:
                self.on_failure("RECORDING_FAILURE")
            raise

    def reconnect(self):
        self.parser.reset()
        self.segment_id = str(uuid.uuid4())

    def close(self):
        if self.closed:
            return
        self.closed = True
        for file in self.files.values():
            file.close()
        if self.failed:
            return
        entries = {}
        for name in self.files:
            partial = self.root / (name + ".partial")
            digest = file_hash(partial)
            entries[name] = {"sha256": digest, "size": partial.stat().st_size}
            partial.rename(self.root / name)
        manifest = {"session_id": self.session_id, "source": "SYNTHETIC",
                    "close_state": "closed", "files": entries}
        # 이 구현은 합성 시험 전용이다. 실측 수집기를 연결하기 전 출처 계약을 확장한다.
        with (self.root / "manifest.json").open("x", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {"transport.bin", "chunks.jsonl", "eeg.i16le", "samples.jsonl"}
    if manifest.get("close_state") != "closed" or set(manifest.get("files", {})) != required:
        raise ValueError("완료되지 않았거나 필수 파일이 누락된 기록입니다.")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.parent != root or name not in required or path.is_symlink():
            raise ValueError("허용되지 않은 기록 파일입니다.")
        digest = file_hash(path)
        if digest != expected["sha256"] or path.stat().st_size != expected["size"]:
            raise ValueError("원본 무결성 검사에 실패했습니다.")
    parser = ThinkGearParser()
    count = 0
    segment = None
    with (root / "transport.bin").open("rb") as transport, (root / "eeg.i16le").open("rb") as eeg, (root / "samples.jsonl").open(encoding="utf-8") as refs:
        with (root / "chunks.jsonl").open(encoding="utf-8") as chunks:
            for chunk_seq, line in enumerate(chunks):
                row = json.loads(line)
                if segment != row["segment_id"]:
                    parser.reset()
                    segment = row["segment_id"]
                if transport.tell() != row["offset"] or row["chunk_seq"] != chunk_seq:
                    raise ValueError("원본 인덱스의 오프셋 또는 순번이 맞지 않습니다.")
                if type(row["len"]) is not int or row["len"] < 0:
                    raise ValueError("원본 인덱스의 길이가 잘못되었습니다.")
                data = transport.read(row["len"])
                if len(data) != row["len"]:
                    raise ValueError("원본 인덱스가 실제 파일 길이를 초과합니다.")
                for packet in parser.feed(data):
                    for sample in packet.raw_samples:
                        expected_ref = {"segment_id": segment, "sample_index": count,
                                        "packet_offset": packet.offset, "chunk_seq": chunk_seq}
                        if json.loads(refs.readline()) != expected_ref:
                            raise ValueError("샘플의 원본 계보가 일치하지 않습니다.")
                        if eeg.read(2) != struct.pack("<h", sample):
                            raise ValueError("원본 재생 EEG가 저장된 EEG와 다릅니다.")
                        count += 1
        if transport.read(1) or eeg.read(1) or refs.read(1):
            raise ValueError("인덱스에 없는 원본 또는 EEG·참조 데이터가 남았습니다.")
    return count
