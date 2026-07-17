import asyncio
import time
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web

from vllm_router.aiohttp_client import (
    DEFAULT_BACKEND_CLIENT_TIMEOUT,
    DEFAULT_BACKEND_CONNECT_TIMEOUT,
    DEFAULT_BACKEND_READ_TIMEOUT,
    backend_entry_deadline,
    build_backend_client_timeout,
)


def test_build_backend_client_timeout_maps_flags():
    timeout = build_backend_client_timeout(10.0, 300.0)
    assert timeout.total is None
    assert timeout.sock_connect == 10.0
    assert timeout.sock_read == 300.0
    # `connect` bounds DNS resolution + pool acquisition, which sock_connect
    # does not cover; a stalled resolver must not hang the request.
    assert timeout.connect == 10.0


def test_backend_entry_deadline_sums_the_enabled_bounds():
    assert backend_entry_deadline(build_backend_client_timeout(10.0, 300.0)) == 310.0
    assert backend_entry_deadline(build_backend_client_timeout(0, 300.0)) == 300.0


def test_backend_entry_deadline_disabled_when_read_bound_is_off():
    # Disabling the read bound means the operator wants unbounded waits;
    # the entry deadline must not resurrect a bound behind their back.
    assert backend_entry_deadline(build_backend_client_timeout(10.0, 0)) is None
    assert backend_entry_deadline(build_backend_client_timeout(0, 0)) is None


def test_backend_entry_deadline_ignores_non_numeric_timeouts():
    # Test harnesses hand process_request a MagicMock app.state; the deadline
    # must degrade to "no bound" rather than feed garbage to asyncio.timeout.
    assert backend_entry_deadline(MagicMock()) is None


def test_build_backend_client_timeout_zero_disables_each_independently():
    timeout = build_backend_client_timeout(0, 300.0)
    assert timeout.sock_connect is None
    assert timeout.sock_read == 300.0

    timeout = build_backend_client_timeout(10.0, 0)
    assert timeout.sock_connect == 10.0
    assert timeout.sock_read is None

    timeout = build_backend_client_timeout(0, 0)
    assert timeout.total is None
    assert timeout.sock_connect is None
    assert timeout.sock_read is None


def test_build_backend_client_timeout_negative_disables():
    timeout = build_backend_client_timeout(-1, -0.5)
    assert timeout.sock_connect is None
    assert timeout.sock_read is None


def test_default_backend_client_timeout_matches_flag_defaults():
    assert DEFAULT_BACKEND_CLIENT_TIMEOUT.total is None
    assert (
        DEFAULT_BACKEND_CLIENT_TIMEOUT.sock_connect == DEFAULT_BACKEND_CONNECT_TIMEOUT
    )
    assert DEFAULT_BACKEND_CLIENT_TIMEOUT.sock_read == DEFAULT_BACKEND_READ_TIMEOUT


@pytest.fixture
async def stall_server():
    """Real HTTP server whose behavior per path pins aiohttp sock_read semantics."""
    body_gate = asyncio.Event()

    async def stall_before_headers(request):
        await body_gate.wait()
        return web.Response(text="too late")

    async def stall_mid_body(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await body_gate.wait()
        return response

    async def steady_trickle(request):
        response = web.StreamResponse()
        await response.prepare(request)
        for _ in range(15):
            await response.write(b"chunk")
            await asyncio.sleep(0.1)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/stall-before-headers", stall_before_headers)
    app.router.add_get("/stall-mid-body", stall_mid_body)
    app.router.add_get("/steady-trickle", steady_trickle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]

    yield f"http://{host}:{port}"

    body_gate.set()
    await runner.cleanup()


async def test_sock_read_bounds_the_wait_for_response_headers(stall_server):
    """A backend that accepts the connection but never answers fails in
    sock_read seconds instead of hanging forever. This is the non-streaming
    lost-request case: headers only arrive once generation finishes."""
    timeout = build_backend_client_timeout(1.0, 0.2)
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.SocketTimeoutError) as excinfo:
            async with session.get(
                f"{stall_server}/stall-before-headers", timeout=timeout
            ):
                pass
    # process_request routes the failure via `except Exception` -> fail,
    # never via the abort branch.
    assert isinstance(excinfo.value, Exception)
    assert not isinstance(excinfo.value, asyncio.CancelledError)


async def test_sock_read_bounds_mid_stream_silence(stall_server):
    """A stream that goes silent after some bytes fails in sock_read seconds."""
    timeout = build_backend_client_timeout(1.0, 0.2)
    received = []
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.SocketTimeoutError):
            async with session.get(
                f"{stall_server}/stall-mid-body", timeout=timeout
            ) as response:
                async for chunk in response.content.iter_any():
                    received.append(chunk)
    assert b"".join(received) == b"first"


async def test_sock_read_resets_on_every_chunk(stall_server):
    """sock_read is silence-per-read, not a total-duration budget: a stream
    that keeps producing bytes runs past sock_read without tripping it.

    Margins are deliberately wide (0.1s writes vs 0.5s watchdog) so a
    descheduled CI event loop cannot fire the watchdog between two writes.
    The stream also outlives the entry deadline (sock_connect + sock_read =
    1.0s < 1.5s of streaming), pinning that the deadline is disarmed once
    response headers arrive.
    """
    timeout = build_backend_client_timeout(0.5, 0.5)
    assert backend_entry_deadline(timeout) == 1.0
    received = []
    async with aiohttp.ClientSession() as session:
        async with asyncio.timeout(backend_entry_deadline(timeout)) as entry:
            response = await session.get(
                f"{stall_server}/steady-trickle", timeout=timeout
            )
            entry.reschedule(None)
        async with response:
            async for chunk in response.content.iter_any():
                received.append(chunk)
    assert b"".join(received) == b"chunk" * 15


async def test_entry_deadline_bounds_a_stalled_request_body_upload():
    """sock_read only arms once the request body is fully written, so a
    backend that accepts the connection but stops reading hangs a large
    upload forever; the entry deadline is what bounds it."""

    release_handler = asyncio.Event()

    async def never_read(reader, writer):
        # Never reads; released at teardown so Server.wait_closed() (which
        # waits for in-flight handlers since 3.12.1) can complete.
        await release_handler.wait()
        writer.close()

    server = await asyncio.start_server(never_read, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    # Large enough to overflow loopback socket buffers so the write blocks.
    body = b"x" * (64 * 1024 * 1024)
    timeout = build_backend_client_timeout(0.2, 0.3)

    start = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(TimeoutError) as excinfo:
                async with asyncio.timeout(backend_entry_deadline(timeout)):
                    await session.post(
                        f"http://127.0.0.1:{port}/", data=body, timeout=timeout
                    )
        # Bounded by the 0.5s entry deadline, not by the outer pytest timeout.
        assert time.monotonic() - start < 5.0
        # Reaches process_request's `except Exception` fail path, and is
        # disambiguated from client aborts.
        assert isinstance(excinfo.value, Exception)
        assert not isinstance(excinfo.value, asyncio.CancelledError)
    finally:
        release_handler.set()
        server.close()
        await server.wait_closed()
