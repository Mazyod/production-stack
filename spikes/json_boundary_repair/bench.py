"""Micro-benchmark for the boundary JSON repair spike."""

from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable

from repair import RepairResult, repair


SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
    "additionalProperties": False,
}

CASES = {
    "clean": '{"a": 1}',
    "extra_brace": '{{"a": 1}',
    "dup_prefix": '{"{"a": 1}',
    "code_fence": '```json\n{"a": 1}\n```',
    "other": 'noise:{"a": 1}',
}


def _call(content: str) -> RepairResult:
    return repair(content, SCHEMA, finish_reason="stop")


def _microseconds_per_call(
    function: Callable[[], object], *, number: int, repeat: int
) -> float:
    samples = timeit.repeat(function, number=number, repeat=repeat)
    return statistics.median(samples) * 1_000_000 / number


def main() -> None:
    number = 10_000
    repeat = 7
    results = {
        name: _microseconds_per_call(
            lambda content=content: _call(content), number=number, repeat=repeat
        )
        for name, content in CASES.items()
    }

    print(f"median of {repeat} runs x {number:,} calls")
    for name, microseconds in results.items():
        print(f"{name:12} {microseconds:9.2f} us/call")

    corrupt_median = statistics.median(
        duration for name, duration in results.items() if name != "clean"
    )
    assert results["clean"] < corrupt_median
    print(
        "clean fast path is cheaper than the median corrupt path: "
        f"{results['clean']:.2f} < {corrupt_median:.2f} us/call"
    )


if __name__ == "__main__":
    main()
