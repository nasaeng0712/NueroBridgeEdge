"""프레임 분할·손상과 실제 루프백 TCP 전달을 검사한다."""

import json
import socket
import struct
import unittest

from neurobridge.protocol import Decoder, ProtocolError, encode

IDENTITY = dict(session_id="s", boot_id="b", sync_epoch="e", connection_epoch="c")


def heartbeat():
    return dict(IDENTITY, schema_version="1.0", type="HEARTBEAT")


class ProtocolTests(unittest.TestCase):
    def test_all_split_points_and_multiple(self):
        wire = encode(heartbeat())
        for split in range(len(wire) + 1):
            decoder = Decoder()
            self.assertEqual(decoder.feed(wire[:split]) + decoder.feed(wire[split:]), [heartbeat()])
            decoder.eof()
        self.assertEqual(Decoder().feed(wire * 3), [heartbeat()] * 3)

    def test_invalid_frames_poison_connection(self):
        invalid = [b"", b"[]", b"{}", b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}', b'{"type":[]}']
        for body in invalid:
            decoder = Decoder()
            with self.assertRaises(ProtocolError):
                decoder.feed(struct.pack(">I", len(body)) + body)
            with self.assertRaises(ProtocolError):
                decoder.feed(encode(heartbeat()))
        with self.assertRaises(ProtocolError):
            Decoder().feed(struct.pack(">I", 16385))

    def test_truncated_eof(self):
        decoder = Decoder()
        decoder.feed(encode(heartbeat())[:-1])
        with self.assertRaises(ProtocolError):
            decoder.eof()

    def test_encode_rejects_invalid_numbers_and_oversize(self):
        for value in (float("nan"), "x" * 16384):
            with self.assertRaises(ProtocolError):
                encode(dict(heartbeat(), extra=value))

    def test_real_tcp_loopback(self):
        with socket.socket() as server:
            server.settimeout(2)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.create_connection(server.getsockname(), timeout=2) as client:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(2)
                    client.sendall(encode(heartbeat()))
                    client.shutdown(socket.SHUT_WR)
                    decoder, messages = Decoder(), []
                    while chunk := connection.recv(7):
                        messages.extend(decoder.feed(chunk))
                    decoder.eof()
                    self.assertEqual(messages, [heartbeat()])
