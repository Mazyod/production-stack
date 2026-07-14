"""Golden-corpus regression harness for structured-output repair.

Run the pytest harness with::

    STRUCTURED_OUTPUT_CORPUS=/path/to/matrix_results.json \
      uv run --no-sync pytest src/tests/test_structured_output_corpus.py -q

Inspect an unfamiliar corpus before running the test with::

    uv run --no-sync python -m src.tests.test_structured_output_corpus \
      "$STRUCTURED_OUTPUT_CORPUS"

The inspector prints only structure and classification labels, never model output.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from vllm_router.services.structured_output.repair import RepairResult, repair

# CORPUS FIELD MAPPING (UNVERIFIED: the production corpus is not in this repo).
# Dotted paths support both the expected flat records and a nested corpus layout.
CONTENT_PATH = "content"
SCHEMA_PATH = "schema"
FINISH_REASON_PATH = "finish_reason"
CLASSIFICATION_PATH = "classification"
CLEAN_CLASSIFICATION = "clean"
CORRUPT_CLASSIFICATION = "corrupt"
# END CORPUS FIELD MAPPING.

CORPUS_ENV_VAR = "STRUCTURED_OUTPUT_CORPUS"
EXPECTED_RECORD_COUNT = 1_536
TRUNCATED_CLASSIFICATION = "truncated"
_MISSING = object()
_NO_DEFAULT = object()


@dataclass(frozen=True)
class CorpusFieldMapping:
    content: str = CONTENT_PATH
    schema: str = SCHEMA_PATH
    finish_reason: str = FINISH_REASON_PATH
    classification: str = CLASSIFICATION_PATH
    clean: str = CLEAN_CLASSIFICATION
    corrupt: str = CORRUPT_CLASSIFICATION

    def describe(self) -> str:
        return (
            f"content={self.content!r}, schema={self.schema!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"classification={self.classification!r}, "
            f"clean={self.clean!r}, corrupt={self.corrupt!r}"
        )


@dataclass(frozen=True)
class CorpusSummary:
    records: int
    classifications: Counter[str]
    statuses: Counter[str]

    def render(self) -> str:
        classifications = ", ".join(
            f"{name}={count}" for name, count in sorted(self.classifications.items())
        )
        statuses = ", ".join(
            f"{name}={count}" for name, count in sorted(self.statuses.items())
        )
        return (
            f"structured-output corpus summary: records={self.records}; "
            f"classifications: {classifications}; statuses: {statuses}"
        )


DEFAULT_MAPPING = CorpusFieldMapping()


def _read_records(corpus_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(corpus_path).open(encoding="utf-8") as handle:
        records = json.load(handle)

    assert isinstance(records, list), "Corpus root must be a JSON array"
    assert records, "Corpus must contain at least one record"
    assert all(
        isinstance(record, dict) for record in records
    ), "Every corpus record must be a JSON object"
    return records


def _dotted_value(
    record: dict[str, Any], path: str, *, default: object = _NO_DEFAULT
) -> Any:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            if default is not _NO_DEFAULT:
                return default
            raise KeyError(path)
        value = value[component]
    return value


def _validate_first_record_mapping(
    record: dict[str, Any], mapping: CorpusFieldMapping
) -> None:
    required_paths = (mapping.content, mapping.schema, mapping.classification)
    missing = [
        path
        for path in required_paths
        if _dotted_value(record, path, default=_MISSING) is _MISSING
    ]
    if missing:
        raise AssertionError(
            "Corpus field mapping did not match the first record. "
            f"Actual top-level keys: {sorted(record)}. "
            f"Expected mapping: {mapping.describe()}. Missing paths: {missing}."
        )


def _schema_error(index: int, error: jsonschema.ValidationError) -> str:
    instance_path = ".".join(str(component) for component in error.absolute_path)
    location = instance_path or "<root>"
    return (
        f"record {index}: repaired value failed schema validation at {location} "
        f"({error.validator})"
    )


def run_corpus(
    corpus_path: str | os.PathLike[str],
    *,
    mapping: CorpusFieldMapping = DEFAULT_MAPPING,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
    repair_fn: Callable[..., RepairResult] = repair,
) -> CorpusSummary:
    """Run all corpus assertions and return non-sensitive aggregate counts."""
    records = _read_records(corpus_path)
    assert (
        len(records) == expected_record_count
    ), f"Expected {expected_record_count} corpus records, found {len(records)}"
    _validate_first_record_mapping(records[0], mapping)

    classifications: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    problems: list[str] = []
    unclassified: list[int] = []

    for index, record in enumerate(records):
        try:
            classification = _dotted_value(record, mapping.classification)
        except KeyError:
            unclassified.append(index)
            continue

        if classification not in {mapping.clean, mapping.corrupt}:
            unclassified.append(index)
            continue

        finish_reason = _dotted_value(record, mapping.finish_reason, default="stop")
        effective_classification = (
            TRUNCATED_CLASSIFICATION if finish_reason == "length" else classification
        )
        classifications[effective_classification] += 1

        try:
            content = _dotted_value(record, mapping.content)
            schema = _dotted_value(record, mapping.schema)
        except KeyError as error:
            problems.append(
                f"record {index}: required mapped field {error.args[0]!r} is missing"
            )
            continue
        if not isinstance(content, str):
            problems.append(f"record {index}: content is not a string")
            continue
        if not isinstance(schema, dict):
            problems.append(f"record {index}: schema is not an object")
            continue

        result = repair_fn(
            content,
            schema,
            finish_reason=finish_reason,
        )
        statuses[result.status] += 1

        if effective_classification == TRUNCATED_CLASSIFICATION:
            if result.status != "incomplete":
                problems.append(
                    f"record {index}: truncated input returned status "
                    f"{result.status!r}, expected 'incomplete'"
                )
            continue

        if classification == mapping.corrupt:
            if result.status != "repaired":
                problems.append(
                    f"record {index}: corrupt input returned status "
                    f"{result.status!r}, expected 'repaired'"
                )
                continue
            if result.text is None or result.text not in content:
                problems.append(
                    f"record {index}: repair did not return a substring of the input"
                )
                continue
            try:
                jsonschema.validate(result.value, schema)
            except jsonschema.ValidationError as error:
                problems.append(_schema_error(index, error))
            continue

        if result.status != "clean":
            problems.append(
                f"record {index}: clean input returned status {result.status!r}, "
                "expected 'clean'"
            )
        if result.text != content or (
            result.text is not None
            and result.text.encode("utf-8") != content.encode("utf-8")
        ):
            problems.append(f"record {index}: clean input was not byte-identical")

    if unclassified:
        classifications["unclassified"] = len(unclassified)
        problems.append(
            f"{len(unclassified)} unclassified record(s) at indices {unclassified}"
        )
    if classifications[mapping.corrupt] == 0:
        problems.append(
            "Found 0 corrupt samples; the classification mapping is probably wrong. "
            f"Expected mapping: {mapping.describe()}. "
            f"Actual top-level keys: {sorted(records[0])}."
        )

    summary = CorpusSummary(len(records), classifications, statuses)
    if problems:
        raise AssertionError(summary.render() + "\n" + "\n".join(problems))
    return summary


def _corpus_path_from_environment() -> str:
    corpus_path = os.environ.get(CORPUS_ENV_VAR)
    if not corpus_path or not os.path.isfile(corpus_path):
        pytest.skip(f"set {CORPUS_ENV_VAR} to matrix_results.json to run")
    return corpus_path


def test_golden_corpus_regression(request: pytest.FixtureRequest) -> None:
    summary = run_corpus(_corpus_path_from_environment())
    terminal_reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(summary.render())


SYNTHETIC_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    records = [
        {
            "content": '{{"summary":"double"}',
            "schema": SYNTHETIC_SCHEMA,
            "finish_reason": "stop",
            "classification": "corrupt",
        },
        {
            "content": '```json{"summary":"fence"}',
            "schema": SYNTHETIC_SCHEMA,
            "finish_reason": "stop",
            "classification": "corrupt",
        },
        {
            "content": '{"{"summary":"quoted-brace"}',
            "schema": SYNTHETIC_SCHEMA,
            "finish_reason": "stop",
            "classification": "corrupt",
        },
        {
            "content": '{"summary":"clean"}',
            "schema": SYNTHETIC_SCHEMA,
            "finish_reason": "stop",
            "classification": "clean",
        },
        {
            "content": '  {"summary":"whitespace"}\n',
            "schema": SYNTHETIC_SCHEMA,
            "classification": "clean",
        },
        {
            "content": '{{"summary":"truncated"',
            "schema": SYNTHETIC_SCHEMA,
            "finish_reason": "length",
            "classification": "corrupt",
        },
    ]
    corpus_path = tmp_path / "matrix_results.json"
    corpus_path.write_text(json.dumps(records), encoding="utf-8")
    return corpus_path


def _run_synthetic(
    corpus_path: Path,
    *,
    mapping: CorpusFieldMapping = DEFAULT_MAPPING,
    repair_fn: Callable[..., RepairResult] = repair,
) -> CorpusSummary:
    return run_corpus(
        corpus_path,
        mapping=mapping,
        expected_record_count=6,
        repair_fn=repair_fn,
    )


def test_harness_passes_complete_synthetic_corpus(synthetic_corpus: Path) -> None:
    summary = _run_synthetic(synthetic_corpus)

    assert summary.classifications == {
        "clean": 2,
        "corrupt": 3,
        "truncated": 1,
    }
    assert summary.statuses == {"clean": 2, "incomplete": 1, "repaired": 3}


def test_harness_supports_dotted_field_mapping(synthetic_corpus: Path) -> None:
    records = _read_records(synthetic_corpus)
    nested_records = []
    for record in records:
        nested_records.append(
            {
                "response": {
                    "content": record["content"],
                    "schema": record["schema"],
                },
                "metadata": {
                    "classification": record["classification"],
                    **(
                        {"finish_reason": record["finish_reason"]}
                        if "finish_reason" in record
                        else {}
                    ),
                },
            }
        )
    synthetic_corpus.write_text(json.dumps(nested_records), encoding="utf-8")
    mapping = CorpusFieldMapping(
        content="response.content",
        schema="response.schema",
        finish_reason="metadata.finish_reason",
        classification="metadata.classification",
    )

    summary = _run_synthetic(synthetic_corpus, mapping=mapping)

    assert summary.classifications == {
        "clean": 2,
        "corrupt": 3,
        "truncated": 1,
    }


def test_harness_fails_when_corrupt_record_does_not_repair(
    synthetic_corpus: Path,
) -> None:
    def fail_corrupt(content: str, schema: dict, **kwargs: Any) -> RepairResult:
        result = repair(content, schema, **kwargs)
        if content.startswith("{{") and kwargs["finish_reason"] != "length":
            return replace(result, status="unknown", text=None, value=None)
        return result

    with pytest.raises(AssertionError, match="corrupt input returned status"):
        _run_synthetic(synthetic_corpus, repair_fn=fail_corrupt)


def test_harness_fails_when_repaired_value_violates_schema(
    synthetic_corpus: Path,
) -> None:
    def return_invalid_value(content: str, schema: dict, **kwargs: Any) -> RepairResult:
        result = repair(content, schema, **kwargs)
        if result.status == "repaired":
            return replace(result, value={"unexpected": True})
        return result

    with pytest.raises(AssertionError, match="failed schema validation"):
        _run_synthetic(synthetic_corpus, repair_fn=return_invalid_value)


def test_harness_fails_when_clean_record_is_altered(
    synthetic_corpus: Path,
) -> None:
    def alter_clean(content: str, schema: dict, **kwargs: Any) -> RepairResult:
        result = repair(content, schema, **kwargs)
        if result.status == "clean":
            return replace(result, text=result.text + " ")
        return result

    with pytest.raises(AssertionError, match="clean input was not byte-identical"):
        _run_synthetic(synthetic_corpus, repair_fn=alter_clean)


def test_harness_fails_when_field_mapping_does_not_match(
    synthetic_corpus: Path,
) -> None:
    wrong_mapping = replace(DEFAULT_MAPPING, content="response.content")

    with pytest.raises(AssertionError) as failure:
        _run_synthetic(synthetic_corpus, mapping=wrong_mapping)

    message = str(failure.value)
    assert "Actual top-level keys" in message
    assert "Expected mapping" in message
    assert "response.content" in message


def test_harness_fails_when_no_corrupt_samples_are_found(
    synthetic_corpus: Path,
) -> None:
    records = _read_records(synthetic_corpus)
    for record in records:
        record["classification"] = "clean"
    synthetic_corpus.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(AssertionError, match="Found 0 corrupt samples"):
        _run_synthetic(synthetic_corpus)


def test_harness_fails_when_record_cannot_be_classified(
    synthetic_corpus: Path,
) -> None:
    records = _read_records(synthetic_corpus)
    records[2]["classification"] = "mystery"
    synthetic_corpus.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(AssertionError, match=r"1 unclassified.*indices \[2\]"):
        _run_synthetic(synthetic_corpus)


def test_harness_skips_when_corpus_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)

    with pytest.raises(pytest.skip.Exception, match=CORPUS_ENV_VAR):
        _corpus_path_from_environment()


def inspect_corpus(
    corpus_path: str | os.PathLike[str],
    *,
    mapping: CorpusFieldMapping = DEFAULT_MAPPING,
) -> str:
    records = _read_records(corpus_path)
    classifications = set()
    for record in records:
        value = _dotted_value(record, mapping.classification, default=_MISSING)
        classifications.add("<missing>" if value is _MISSING else repr(value))
    return "\n".join(
        (
            f"record count: {len(records)}",
            f"first-record top-level keys: {sorted(records[0])}",
            "classification values: " + ", ".join(sorted(classifications)),
            f"expected mapping: {mapping.describe()}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="path to matrix_results.json")
    arguments = parser.parse_args(argv)
    print(inspect_corpus(arguments.corpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
