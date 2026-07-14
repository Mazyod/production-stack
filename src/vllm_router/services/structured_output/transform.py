"""Apply boundary repair to backend responses.

Every failure path returns the ORIGINAL body object unchanged -- identity is
asserted in the tests, so a caller can rely on `out is body` meaning "untouched".
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from vllm_router.services.structured_output.contract import OutputContract
from vllm_router.services.structured_output.repair import RepairResult, repair
from vllm_router.services.structured_output.sse import SSEEvent, SSEParser

logger = logging.getLogger(__name__)

CaptureCallback = Callable[[str, "RepairTelemetry"], None]


@dataclass(frozen=True)
class RepairTelemetry:
    status: str
    mode: str
    garbage_prefix_bytes: int


_INCOMPLETE_TELEMETRY = RepairTelemetry("incomplete", "other", 0)
_POISONED_TELEMETRY = RepairTelemetry("poisoned", "other", 0)
_NO_TERMINAL_TELEMETRY = RepairTelemetry("no_terminal", "other", 0)
_CAPPED_TELEMETRY = RepairTelemetry("capped", "other", 0)
_TIMEOUT_TELEMETRY = RepairTelemetry("timeout", "other", 0)
_ERROR_TELEMETRY = RepairTelemetry("error", "other", 0)


def _telemetry(result: RepairResult) -> RepairTelemetry:
    return RepairTelemetry(
        status=result.status,
        mode=result.mode,
        garbage_prefix_bytes=len(result.garbage_prefix.encode("utf-8")),
    )


def _capture_refusal(
    callback: CaptureCallback | None,
    content: str,
    result: RepairResult,
) -> None:
    if callback is None or result.status not in {"ambiguous", "unknown"}:
        return
    try:
        callback(content, _telemetry(result))
    except Exception:  # noqa: BLE001 - diagnostics never affect output
        try:
            logger.warning("structured-output diagnostic capture failed")
        except Exception:  # noqa: BLE001 - logging must not affect output
            pass


def _warn_transform_failure() -> None:
    try:
        logger.warning("structured-output repair failed; passing response through")
    except Exception:  # noqa: BLE001 - logging must not break response pass-through
        pass


def _warn_stream_failure(message: str) -> None:
    try:
        logger.warning(message)
    except Exception:  # noqa: BLE001 - logging must not break stream replay
        pass


def _repair_choice(choice: dict, contract: OutputContract) -> RepairResult | None:
    """Repair one choice's content. None means "nothing to look at here"."""
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None

    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        return None

    return repair(
        content,
        contract.content_schema,
        finish_reason=finish_reason,
    )


def transform_response_body(
    body: bytes,
    contract: OutputContract,
    *,
    capture_callback: CaptureCallback | None = None,
) -> tuple[bytes, list[RepairTelemetry]]:
    """Contain every transform exception, including RecursionError."""
    try:
        return _transform_response_body(
            body, contract, capture_callback=capture_callback
        )
    except Exception:  # noqa: BLE001 - the transform is an availability boundary
        _warn_transform_failure()
        return body, []


def _transform_response_body(
    body: bytes,
    contract: OutputContract,
    *,
    capture_callback: CaptureCallback | None = None,
) -> tuple[bytes, list[RepairTelemetry]]:
    if not contract.engaged:
        return body, []

    try:
        payload = json.loads(body)
        choices = payload["choices"]
        if not isinstance(choices, list):
            return body, []
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        return body, []

    telemetry: list[RepairTelemetry] = []
    changed = False

    try:
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            result = _repair_choice(choice, contract)
            if result is None:
                telemetry.append(_NO_TERMINAL_TELEMETRY)
                return body, telemetry
            _capture_refusal(
                capture_callback,
                choice["message"]["content"],
                result,
            )
            telemetry.append(_telemetry(result))
            if result.status == "repaired":
                choice["message"]["content"] = result.text
                changed = True
            elif result.status != "clean":
                return body, [item for item in telemetry if item.status != "repaired"]
    except Exception:  # noqa: BLE001 - a repair failure must never break a response
        _warn_transform_failure()
        return body, []

    if not changed:
        return body, telemetry

    try:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8"), telemetry
    except (TypeError, ValueError):
        return body, []


_MAX_BUFFERED_BYTES = 1_048_576
_MAX_BUFFER_SECONDS = 30.0


class StreamRepairer:
    """Forward reasoning live; withhold content frames; repair or replay at the end.

    Buffering begins globally at the first frame carrying `delta.content`. With n > 1
    this delays other choices' reasoning -- a deliberate simplification, since
    structured output with multiple choices is rare.
    """

    def __init__(
        self,
        contract: OutputContract,
        max_buffered_bytes: int = _MAX_BUFFERED_BYTES,
        max_buffer_seconds: float = _MAX_BUFFER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        *,
        capture_callback: CaptureCallback | None = None,
    ) -> None:
        self._contract = contract
        self._max_buffered_bytes = max_buffered_bytes
        self._max_buffer_seconds = max_buffer_seconds
        self._clock = clock
        self._capture_callback = capture_callback
        self._parser = SSEParser()
        self._buffering = False
        self._disabled = not (contract.engaged and contract.content_schema is not None)
        self._retained: list[tuple[SSEEvent, dict | None]] = []
        self._retained_bytes = 0
        self._content: dict[int, list[str]] = {}
        self._finish: dict[int, str | None] = {}
        self._buffer_started_at: float | None = None
        self.telemetry: list[RepairTelemetry] = []

    # -- helpers ---------------------------------------------------------------

    def _reset(self) -> None:
        """Clear buffered stream state after replay, disable, abort, or completion."""
        self._retained = []
        self._retained_bytes = 0
        self._content = {}
        self._finish = {}
        self._buffering = False
        self._buffer_started_at = None

    def _replay(self) -> bytes:
        out = b"".join(event.raw for event, _ in self._retained)
        self._reset()
        return out

    def _disable_and_replay(
        self,
        extra: bytes = b"",
        telemetry: tuple[RepairTelemetry, ...] = (),
    ) -> bytes:
        self.telemetry.extend(telemetry)
        out = self._replay() + extra + self._parser.flush()
        self._disabled = True
        return out

    @staticmethod
    def _contains_data_field(event: SSEEvent) -> bool:
        normalized = event.raw.replace(b"\r\n", b"\n")
        return any(
            line == b"data" or line.startswith(b"data:")
            for line in normalized.split(b"\n")
        )

    @classmethod
    def _classify_event(cls, event: SSEEvent) -> tuple[dict | None, bool]:
        """Return the parsed payload and whether the event poisons repair."""
        if event.data is None:
            poison = not event.is_done and cls._contains_data_field(event)
            return None, poison

        try:
            payload = json.loads(event.data)
        except (json.JSONDecodeError, ValueError):
            return None, True

        if not isinstance(payload, dict):
            return None, True
        if "choices" not in payload:
            return payload, False

        choices = payload["choices"]
        if not isinstance(choices, list):
            return payload, True
        for choice in choices:
            if not isinstance(choice, dict):
                return payload, True
            if type(choice.get("index", 0)) is not int:
                return payload, True
            if "delta" not in choice:
                continue
            delta = choice["delta"]
            if not isinstance(delta, dict):
                return payload, True
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                return payload, True
        return payload, False

    @staticmethod
    def _has_content(payload: dict) -> bool:
        for choice in payload.get("choices", []):
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content"):
                return True
        return False

    def _absorb(self, payload: dict) -> RepairTelemetry | None:
        """Absorb a validated payload, returning telemetry when repair must stop."""
        choices = payload.get("choices", [])
        for choice in choices:
            index = choice.get("index", 0)
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if content:
                if index in self._finish:
                    return _POISONED_TELEMETRY
                self._content.setdefault(index, []).append(content)

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if not isinstance(finish_reason, str):
                    return _POISONED_TELEMETRY
                if finish_reason == "length":
                    return _INCOMPLETE_TELEMETRY
                previous = self._finish.get(index)
                if previous is not None and previous != finish_reason:
                    return _POISONED_TELEMETRY
                self._finish[index] = finish_reason
        return None

    @staticmethod
    def _event_terminator(event: SSEEvent) -> bytes:
        if event.raw.endswith(b"\r\n\r\n"):
            return b"\r\n\r\n"
        return b"\n\n"

    def _render_commit(self) -> tuple[bytes | None, tuple[RepairTelemetry, ...]]:
        """Render rewritten frames, or return truthful telemetry for raw replay."""
        repaired: dict[int, str] = {}
        staged_telemetry: list[RepairTelemetry] = []
        for index, parts in self._content.items():
            content = "".join(parts)
            result = repair(
                content,
                self._contract.content_schema,
                finish_reason=self._finish[index],
            )
            _capture_refusal(self._capture_callback, content, result)
            staged_telemetry.append(_telemetry(result))
            if result.status == "repaired":
                repaired[index] = result.text
            elif result.status != "clean":
                declines = tuple(
                    item for item in staged_telemetry if item.status != "repaired"
                )
                return None, declines

        if not repaired:
            return None, tuple(staged_telemetry)

        out = bytearray()
        emitted: set[int] = set()
        for event, payload in self._retained:
            if payload is None:
                out.extend(event.raw)
                continue
            touched = False
            for choice in payload.get("choices", []):
                index = choice.get("index", 0)
                delta = choice.get("delta")
                if (
                    index not in repaired
                    or not isinstance(delta, dict)
                    or not delta.get("content")
                ):
                    continue
                # First content frame for this choice carries the whole repaired
                # document; later ones are emptied. Everything else is preserved.
                if index in emitted:
                    delta["content"] = ""
                else:
                    delta["content"] = repaired[index]
                    emitted.add(index)
                touched = True
            if touched:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                out.extend(b"data: " + encoded + self._event_terminator(event))
            else:
                out.extend(event.raw)

        return bytes(out), tuple(staged_telemetry)

    def _finish_stream(self, extra: bytes = b"") -> bytes:
        """Commit or replay exactly once, now that end-of-stream is known."""
        terminal = bool(self._content) and all(
            index in self._finish for index in self._content
        )
        if not terminal:
            return self._disable_and_replay(
                extra,
                telemetry=(_NO_TERMINAL_TELEMETRY,),
            )

        rewritten, staged_telemetry = self._render_commit()
        if rewritten is None:
            return self._disable_and_replay(extra, telemetry=staged_telemetry)

        self.telemetry.extend(staged_telemetry)
        self._reset()
        self._disabled = True
        return rewritten + extra + self._parser.flush()

    # -- public ----------------------------------------------------------------

    @property
    def buffering(self) -> bool:
        return self._buffering and not self._disabled

    @property
    def seconds_remaining(self) -> float | None:
        if not self.buffering or self._buffer_started_at is None:
            return None
        elapsed = self._clock() - self._buffer_started_at
        return max(0.0, self._max_buffer_seconds - elapsed)

    def feed(self, chunk: bytes) -> bytes:
        if self._disabled:
            return chunk

        out = bytearray()
        events: list[SSEEvent] = []
        next_unaccounted = 0
        try:
            if self.buffering and self.seconds_remaining == 0.0:
                return self._disable_and_replay(telemetry=(_TIMEOUT_TELEMETRY,)) + chunk
        except Exception:  # noqa: BLE001 - preserve the not-yet-fed chunk
            _warn_stream_failure(
                "structured-output deadline check failed; replaying raw"
            )
            return self._disable_and_replay(telemetry=(_ERROR_TELEMETRY,)) + chunk

        try:
            events = self._parser.feed(chunk)
            for position, event in enumerate(events):
                payload, poison = self._classify_event(event)
                if poison:
                    unprocessed = b"".join(e.raw for e in events[next_unaccounted:])
                    out.extend(
                        self._disable_and_replay(
                            unprocessed,
                            telemetry=(_POISONED_TELEMETRY,),
                        )
                    )
                    break

                if not self._buffering:
                    if payload is not None and self._has_content(payload):
                        self._buffering = True
                        self._buffer_started_at = self._clock()
                    else:
                        # Forward the current raw event before the loop can continue.
                        out.extend(event.raw)
                        next_unaccounted = position + 1
                        continue

                # The event becomes accounted only after successful retention.
                self._retained.append((event, payload))
                self._retained_bytes += len(event.raw)
                next_unaccounted = position + 1
                decline = self._absorb(payload) if payload is not None else None
                if decline is not None:
                    unprocessed = b"".join(e.raw for e in events[next_unaccounted:])
                    out.extend(
                        self._disable_and_replay(
                            unprocessed,
                            telemetry=(decline,),
                        )
                    )
                    break

                buffered_bytes = self._retained_bytes + self._parser.buffered_bytes
                decline_telemetry = None
                if buffered_bytes > self._max_buffered_bytes:
                    decline_telemetry = _CAPPED_TELEMETRY
                elif self.seconds_remaining == 0.0:
                    decline_telemetry = _TIMEOUT_TELEMETRY
                if decline_telemetry is not None:
                    unprocessed = b"".join(e.raw for e in events[position + 1 :])
                    out.extend(
                        self._disable_and_replay(
                            unprocessed,
                            telemetry=(decline_telemetry,),
                        )
                    )
                    break

                if event.is_done:
                    unprocessed = b"".join(e.raw for e in events[position + 1 :])
                    out.extend(self._finish_stream(unprocessed))
                    break
        except Exception:  # noqa: BLE001 - never break the stream
            _warn_stream_failure(
                "structured-output stream repair failed; replaying raw"
            )
            unprocessed = b"".join(e.raw for e in events[next_unaccounted:])
            out.extend(
                self._disable_and_replay(
                    unprocessed,
                    telemetry=(_ERROR_TELEMETRY,),
                )
            )

        if not self._disabled and (
            self._retained_bytes + self._parser.buffered_bytes
            > self._max_buffered_bytes
        ):
            out.extend(self._disable_and_replay(telemetry=(_CAPPED_TELEMETRY,)))

        return bytes(out)

    def flush(self) -> bytes:
        if self._disabled:
            return self._parser.flush()

        tail = self._parser.flush()
        if not self._retained:
            self._disabled = True
            return tail
        if tail:
            return self._disable_and_replay(tail, telemetry=(_ERROR_TELEMETRY,))

        try:
            if self.seconds_remaining == 0.0:
                return self._disable_and_replay(telemetry=(_TIMEOUT_TELEMETRY,))
            return self._finish_stream()
        except Exception:  # noqa: BLE001 - never break end-of-stream replay
            _warn_stream_failure("structured-output stream flush failed; replaying raw")
            return self._disable_and_replay(telemetry=(_ERROR_TELEMETRY,))

    def abort(self) -> bytes:
        """A transport error occurred. Give the client back what we withheld."""
        return self._disable_and_replay()
