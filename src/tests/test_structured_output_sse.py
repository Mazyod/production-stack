import time

from vllm_router.services.structured_output.sse import SSEParser


def test_single_frame():
    p = SSEParser()
    events = p.feed(b'data: {"a": 1}\n\n')
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'
    assert events[0].raw == b'data: {"a": 1}\n\n'


def test_frame_split_across_chunks():
    p = SSEParser()
    assert p.feed(b'data: {"a') == []
    events = p.feed(b'": 1}\n\n')
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'


def test_split_utf8_codepoint():
    payload = 'data: {"a": "é"}\n\n'.encode()
    p = SSEParser()
    # split in the middle of the 2-byte e-acute
    cut = payload.index(b"\xc3") + 1
    assert p.feed(payload[:cut]) == []
    events = p.feed(payload[cut:])
    assert len(events) == 1
    assert events[0].data == '{"a": "é"}'


def test_crlf_terminator():
    p = SSEParser()
    events = p.feed(b'data: {"a": 1}\r\n\r\n')
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'
    assert events[0].raw == b'data: {"a": 1}\r\n\r\n'


def test_done_sentinel_has_no_data_payload():
    p = SSEParser()
    events = p.feed(b"data: [DONE]\n\n")
    assert events[0].data is None
    assert events[0].is_done is True
    assert events[0].raw == b"data: [DONE]\n\n"


def test_done_text_in_heartbeat_is_not_the_sentinel():
    event = SSEParser().feed(b": heartbeat [DONE]\n\n")[0]
    assert event.data is None
    assert event.is_done is False


def test_comment_and_multiline_are_opaque():
    p = SSEParser()
    events = p.feed(b": heartbeat\n\n" + b"data: a\ndata: b\n\n")
    assert [e.data for e in events] == [None, None]
    assert events[0].raw == b": heartbeat\n\n"


def test_flush_returns_unterminated_tail():
    p = SSEParser()
    p.feed(b"data: partial")
    assert p.buffered_bytes == len(b"data: partial")
    assert p.flush() == b"data: partial"
    assert p.buffered_bytes == 0


def test_raw_bytes_roundtrip_exactly():
    stream = b'data: {"a": 1}\n\n: hb\n\ndata: [DONE]\n\n'
    p = SSEParser()
    events = p.feed(stream)
    assert b"".join(e.raw for e in events) + p.flush() == stream


def test_raw_bytes_roundtrip_exactly_when_every_byte_is_a_chunk():
    stream = b'data: {"a": 1}\r\n\r\n: hb\n\ndata: [DONE]\n\nunterminated'
    parser = SSEParser()
    events = []
    for byte in stream:
        events.extend(parser.feed(bytes((byte,))))

    assert b"".join(event.raw for event in events) + parser.flush() == stream


def test_unterminated_one_byte_chunks_parse_under_hard_time_bound():
    parser = SSEParser()
    started = time.perf_counter()
    for _ in range(200_000):
        assert parser.feed(b"x") == []
    elapsed = time.perf_counter() - started

    assert parser.buffered_bytes == 200_000
    assert elapsed < 1.0, f"incremental SSE scan took {elapsed:.6f}s"
