# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import aiohttp

from vllm_router.log import init_logger

logger = init_logger(__name__)

DEFAULT_BACKEND_CONNECT_TIMEOUT = 10.0
DEFAULT_BACKEND_READ_TIMEOUT = 300.0


def build_backend_client_timeout(
    connect_timeout: Optional[float], read_timeout: Optional[float]
) -> aiohttp.ClientTimeout:
    """Timeout for proxied backend requests.

    total stays None: a healthy streaming response may legitimately run for
    hours. sock_connect and sock_read bound the two ways a lost backend most
    commonly hangs a request forever — a connection that never establishes,
    and a socket that goes silent (sock_read also covers the wait for response
    headers, and is rearmed on every byte received; aiohttp suspends it while
    reading is paused for downstream backpressure). connect is set to the same
    bound as sock_connect because it additionally covers DNS resolution, which
    sock_connect does not. Zero or negative disables a bound.
    """
    connect = connect_timeout if connect_timeout and connect_timeout > 0 else None
    return aiohttp.ClientTimeout(
        total=None,
        connect=connect,
        sock_connect=connect,
        sock_read=read_timeout if read_timeout and read_timeout > 0 else None,
    )


def backend_entry_deadline(timeout: aiohttp.ClientTimeout) -> Optional[float]:
    """Deadline for reaching response headers (connect + upload + header wait).

    sock_read only arms once the request body is fully written, so a backend
    that accepts the connection but stops reading can hang a large upload
    forever with every ClientTimeout field satisfied. The entry deadline
    closes that hole: sock_read plus sock_connect when both are enabled.
    A disabled read bound disables the deadline too — the operator asked for
    unbounded waits. Non-numeric fields (test doubles) also disable it.
    """
    read = timeout.sock_read
    if not isinstance(read, (int, float)) or read <= 0:
        return None
    connect = timeout.sock_connect
    if isinstance(connect, (int, float)) and connect > 0:
        return read + connect
    return read


DEFAULT_BACKEND_CLIENT_TIMEOUT = build_backend_client_timeout(
    DEFAULT_BACKEND_CONNECT_TIMEOUT, DEFAULT_BACKEND_READ_TIMEOUT
)


class AiohttpClientWrapper:

    async_client = None

    def start(self):
        """Instantiate the client. Call from the FastAPI startup hook."""
        # To fully leverage the router's concurrency capabilities,
        # we set the maximum number of connections to be unlimited.
        connector = aiohttp.TCPConnector(limit=0)
        self.async_client = aiohttp.ClientSession(
            connector=connector, connector_owner=True
        )
        logger.info(f"aiohttp ClientSession instantiated. Id {id(self.async_client)}")

    async def stop(self):
        """Gracefully shutdown. Call from FastAPI shutdown hook."""
        logger.info(
            f"aiohttp async_client.closed: {self.async_client.closed} - Now close it. Id (will be unchanged): {id(self.async_client)}"
        )
        await self.async_client.close()
        logger.info(
            f"aiohttp async_client.closed: {self.async_client.closed}. Id (will be unchanged): {id(self.async_client)}"
        )
        self.async_client = None
        logger.info("aiohttp ClientSession closed")

    def __call__(self):
        """Calling the instantiated AiohttpClientWrapper returns the wrapped singleton."""
        # Ensure we don't use it if not started / running
        assert self.async_client is not None
        return self.async_client
