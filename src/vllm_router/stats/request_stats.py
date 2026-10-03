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
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Literal, Optional

from vllm_router.log import init_logger

logger = init_logger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class RequestStats:
    # Number of queries per second
    qps: float
    # Average time-to-first-token (TTFT) in seconds
    ttft: float
    # Total number of requests during prefilling
    in_prefill_requests: int
    # Total number of requests during decoding
    in_decoding_requests: int
    # Total number of requests finished
    finished_requests: int
    # How long the engine has been serving requests (uptime)
    uptime: int
    # Average decoding length (time from first token to completion)
    avg_decoding_length: float
    # Average overall latency (from request arrival to completion)
    avg_latency: float
    # Average inter-token latency (if available; default -1 if not computed)
    avg_itl: float
    # Number of swapped requests (moved from GPU to CPU)
    num_swapped_requests: int


@dataclass(frozen=True)
class RequestHandle:
    """Opaque identity for one request attempt against one engine."""

    _value: int


@dataclass
class _ActiveRequest:
    stage: Literal["prefill", "decoding"]
    engine_url: str
    request_id: str
    start_time: float
    first_token_time: Optional[float] = None


class MovingAverageMonitor:
    """
    Monitors the average of values in a sliding window.
    """

    def __init__(self, sliding_window_size: float):
        self.sliding_window_size = sliding_window_size
        self.timestamps: Deque[float] = deque()
        self.values: Deque[float] = deque()

    def update(self, timestamp: float, value: float):
        """
        Update the throughput monitor with a new timestamp

        Args:
            timestamp: The timestamp of the data point.
            value: The value of the data point.

        This method adds the new data point to the sliding window and
        removes any data point that is older than the sliding window size.
        """
        self.timestamps.append(timestamp)
        self.values.append(value)
        while (
            self.timestamps
            and self.timestamps[0] < timestamp - self.sliding_window_size
        ):
            self.timestamps.popleft()
            self.values.popleft()

    def update_no_value(self, timestamp: float):
        """
        Update the throughput monitor with a new timestamp with no value
        """
        while (
            len(self.timestamps) > 0
            and self.timestamps[0] < timestamp - self.sliding_window_size
        ):
            self.timestamps.popleft()
            self.values.popleft()

    def get_average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else -1

    def get_sum(self) -> float:
        return sum(self.values)


class RequestStatsMonitor(metaclass=SingletonMeta):
    """
    Monitors the request statistics of all serving engines.
    """

    # NOTE (ApostaC): Currently, QPS is calculated based on the number of
    # arrived requests in the sliding window, but the inter_token_latency and
    # ttft are calculated based on the number of completed requests in the
    # sliding window.
    def __init__(self, sliding_window_size: float = None):
        if hasattr(self, "_initialized"):
            return
        if sliding_window_size is None:
            raise ValueError(
                "RequestStatsMonitor must be initialized with sliding_window_size"
            )
        self.sliding_window_size = sliding_window_size
        self._lock = threading.RLock()
        self._next_handle = 0
        self._active_requests: Dict[RequestHandle, _ActiveRequest] = {}
        self.qps_monitors: Dict[str, MovingAverageMonitor] = {}
        self.ttft_monitors: Dict[str, MovingAverageMonitor] = {}

        # These per-handle indexes are retained for observability and are always
        # retired with their authoritative active request record.
        self.request_start_time: Dict[RequestHandle, float] = {}
        self.first_token_time: Dict[RequestHandle, float] = {}

        # Number of requests in different stages (from the start of the router)
        self.in_prefill_requests: Dict[str, int] = {}
        self.in_decoding_requests: Dict[str, int] = {}
        self.finished_requests: Dict[str, int] = {}
        # New monitors for overall latency and decoding length
        self.latency_monitors: Dict[str, MovingAverageMonitor] = {}
        self.decoding_length_monitors: Dict[str, MovingAverageMonitor] = {}

        # Counter for swapped requests
        self.swapped_requests: Dict[str, int] = {}

        self.first_query_time: float = None
        self._initialized = True

    def on_new_request(
        self, engine_url: str, request_id: str, timestamp: float
    ) -> RequestHandle:
        """
        Tell the monitor that a new request has been created.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: the timestamp when the request was created
        """
        with self._lock:
            handle = RequestHandle(self._next_handle)
            self._next_handle += 1
            self._active_requests[handle] = _ActiveRequest(
                stage="prefill",
                engine_url=engine_url,
                request_id=request_id,
                start_time=timestamp,
            )
            self.request_start_time[handle] = timestamp

            self.in_prefill_requests.setdefault(engine_url, 0)
            self.in_prefill_requests[engine_url] += 1

            if engine_url not in self.qps_monitors:
                self.qps_monitors[engine_url] = MovingAverageMonitor(
                    self.sliding_window_size
                )
            self.qps_monitors[engine_url].update(timestamp, 1)

            if engine_url not in self.latency_monitors:
                self.latency_monitors[engine_url] = MovingAverageMonitor(
                    self.sliding_window_size
                )

            if self.first_query_time is None:
                self.first_query_time = timestamp

            return handle

    def on_request_response(self, handle: RequestHandle, timestamp: float):
        """
        Tell the monitor that a response token has been received for a request.

        Args:
            handle: The internal request-attempt handle
            timestamp: The timestamp when the response token was received
        """
        with self._lock:
            request = self._active_requests.get(handle)
            if request is None or request.stage != "prefill":
                return

            engine_url = request.engine_url
            if self.in_prefill_requests.get(engine_url, 0) <= 0:
                logger.error(
                    "Request stats invariant violated: handle %r is prefilling "
                    "but engine %s has no prefill requests",
                    handle,
                    engine_url,
                )
                return

            request.stage = "decoding"
            request.first_token_time = timestamp
            self.first_token_time[handle] = timestamp
            self.in_prefill_requests[engine_url] -= 1
            self.in_decoding_requests.setdefault(engine_url, 0)
            self.in_decoding_requests[engine_url] += 1

            if engine_url not in self.ttft_monitors:
                self.ttft_monitors[engine_url] = MovingAverageMonitor(
                    self.sliding_window_size
                )
            self.ttft_monitors[engine_url].update(
                timestamp, timestamp - request.start_time
            )

    def _finalize_request(
        self, handle: RequestHandle, timestamp: float, *, completed: bool
    ) -> None:
        with self._lock:
            # Pop first so duplicate or re-entrant terminalization is a no-op.
            request = self._active_requests.pop(handle, None)
            if request is None:
                return

            self.request_start_time.pop(handle, None)
            self.first_token_time.pop(handle, None)

            engine_url = request.engine_url
            stage_counts = (
                self.in_prefill_requests
                if request.stage == "prefill"
                else self.in_decoding_requests
            )
            if stage_counts.get(engine_url, 0) <= 0:
                logger.error(
                    "Request stats invariant violated: retiring handle %r from "
                    "%s on engine %s with no matching active count",
                    handle,
                    request.stage,
                    engine_url,
                )
            else:
                stage_counts[engine_url] -= 1

            if not completed:
                return

            self.finished_requests.setdefault(engine_url, 0)
            self.finished_requests[engine_url] += 1
            self.latency_monitors[engine_url].update(
                timestamp, timestamp - request.start_time
            )

    def on_request_complete(self, handle: RequestHandle, timestamp: float):
        """
        Tell the monitor that a request has been completed.

        Args:
            handle: The internal request-attempt handle
            timestamp: The timestamp when the request was completed
        """
        self._finalize_request(handle, timestamp, completed=True)

    def on_request_fail(self, handle: RequestHandle, timestamp: float):
        """Retire a request attempt that failed at the backend or transport."""
        self._finalize_request(handle, timestamp, completed=False)

    def on_request_abort(self, handle: RequestHandle, timestamp: float):
        """Retire a request attempt abandoned or cancelled by the client."""
        self._finalize_request(handle, timestamp, completed=False)

    def on_request_swapped(self, engine_url: str, request_id: str, timestamp: float):
        # This function should be called if a request is determined to be swapped from GPU to CPU.
        """
        Tell the monitor that a request has been swapped from GPU to CPU.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: The timestamp when the request was swapped
        """
        with self._lock:
            if engine_url not in self.swapped_requests:
                self.swapped_requests[engine_url] = 0
            self.swapped_requests[engine_url] += 1

    def get_request_stats(self, current_time: float) -> Dict[str, RequestStats]:
        """
        Get the request statistics for each serving engine

        Args:
            current_time: The current timestamp in seconds

        Returns:
            A dictionary where the key is the serving engine URL and the value
            is the request statistics for that engine.
            The TTFT and inter token latency will be -1 if there is no requests
            finished in the sliding window.
        """
        with self._lock:
            return self._get_request_stats_locked(current_time)

    def _get_request_stats_locked(self, current_time: float) -> Dict[str, RequestStats]:
        ret = {}
        urls = set(self.in_prefill_requests).union(self.in_decoding_requests)
        for engine_url in urls:
            if engine_url not in self.qps_monitors:
                qps = -1
            else:
                # Update the monitors
                self.qps_monitors[engine_url].update_no_value(current_time)
                qps = self.qps_monitors[engine_url].get_sum() / self.sliding_window_size

            if engine_url not in self.ttft_monitors:
                ttft = -1
            else:
                # Update the monitors
                self.ttft_monitors[engine_url].update_no_value(current_time)
                ttft = self.ttft_monitors[engine_url].get_average()

            in_prefill = self.in_prefill_requests.get(engine_url, 0)
            in_decoding = self.in_decoding_requests.get(engine_url, 0)
            finished = self.finished_requests.get(engine_url, 0)

            if engine_url in self.decoding_length_monitors:
                avg_dec_len = self.decoding_length_monitors[engine_url].get_average()
            else:
                avg_dec_len = -1

            if engine_url in self.latency_monitors:
                avg_lat = self.latency_monitors[engine_url].get_average()
            else:
                avg_lat = -1

            # For avg_itl, if not computed, default to -1.
            avg_itl_val = -1

            if engine_url in self.swapped_requests:
                swapped = self.swapped_requests[engine_url]
            else:
                swapped = 0

            ret[engine_url] = RequestStats(
                qps=qps,
                ttft=ttft,
                in_prefill_requests=in_prefill,
                in_decoding_requests=in_decoding,
                finished_requests=finished,
                uptime=(
                    current_time - self.first_query_time if self.first_query_time else 0
                ),
                avg_decoding_length=avg_dec_len,
                avg_latency=avg_lat,
                avg_itl=avg_itl_val,
                num_swapped_requests=swapped,
            )
        return ret


def initialize_request_stats_monitor(sliding_window_size: float):
    return RequestStatsMonitor(sliding_window_size)


def get_request_stats_monitor():
    return RequestStatsMonitor()
