"""Byte-preserving incremental SSE parser.

Frames are split on byte terminators and decoded only once complete, so a UTF-8
code point split across TCP chunks is handled for free. `raw` is the frame's exact
original bytes -- that is what makes byte-for-byte replay possible.
"""

from __future__ import annotations

from dataclasses import dataclass

_TERMINATORS = (b"\r\n\r\n", b"\n\n")
_DONE = "[DONE]"


@dataclass(frozen=True)
class SSEEvent:
    raw: bytes
    data: str | None
    is_done: bool


def _classify(frame: bytes) -> tuple[str | None, bool]:
    """Decode a frame that is exactly one `data:` line carrying JSON, else None.

    Anything else -- comments, heartbeats, [DONE], multi-line data, unknown fields --
    returns None and must be forwarded verbatim.
    """
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError:
        return None, False

    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line]
    if len(lines) != 1:
        return None, False

    line = lines[0]
    if not line.startswith("data:"):
        return None, False

    payload = line[len("data:") :]
    if payload.startswith(" "):
        payload = payload[1:]
    if payload == _DONE:
        return None, True
    if not payload:
        return None, False
    if not payload.startswith("{"):
        return None, False
    return payload, False


class SSEParser:
    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buffer.extend(chunk)
        events: list[SSEEvent] = []

        while True:
            cut = -1
            width = 0
            for terminator in _TERMINATORS:
                found = self._buffer.find(terminator)
                if found != -1 and (cut == -1 or found < cut):
                    cut = found
                    width = len(terminator)
            if cut == -1:
                break

            raw = bytes(self._buffer[: cut + width])
            del self._buffer[: cut + width]
            data, is_done = _classify(raw[: -width or None])
            events.append(SSEEvent(raw=raw, data=data, is_done=is_done))

        return events

    def flush(self) -> bytes:
        tail = bytes(self._buffer)
        self._buffer.clear()
        return tail
