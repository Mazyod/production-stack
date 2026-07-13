from vllm_router.services.structured_output.contract import extract_output_contract

STRICT = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _req(**kw):
    base = {"model": "m", "messages": []}
    base.update(kw)
    return base


def _rf(schema):
    return {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}


def test_engages_on_strict_object_schema():
    c = extract_output_contract(_req(response_format=_rf(STRICT)))
    assert c.engaged is True
    assert c.content_schema == STRICT


def test_not_engaged_without_response_format():
    assert extract_output_contract(_req()).engaged is False


def test_not_engaged_on_json_object_mode():
    c = extract_output_contract(_req(response_format={"type": "json_object"}))
    assert c.engaged is False


def test_not_engaged_when_logprobs_is_zero():
    contract = extract_output_contract(_req(response_format=_rf(STRICT), logprobs=0))
    assert contract.engaged is False


def test_not_engaged_when_logprobs_is_five():
    contract = extract_output_contract(_req(response_format=_rf(STRICT), logprobs=5))
    assert contract.engaged is False


def test_not_engaged_when_logprobs_is_true():
    contract = extract_output_contract(_req(response_format=_rf(STRICT), logprobs=True))
    assert contract.engaged is False


def test_not_engaged_for_other_logprobs_values():
    for logprobs in (1, [], {}, "requested"):
        contract = extract_output_contract(
            _req(response_format=_rf(STRICT), logprobs=logprobs)
        )
        assert contract.engaged is False


def test_engaged_when_logprobs_is_false():
    contract = extract_output_contract(
        _req(response_format=_rf(STRICT), logprobs=False)
    )
    assert contract.engaged is True


def test_engaged_when_logprobs_is_absent():
    contract = extract_output_contract(_req(response_format=_rf(STRICT)))
    assert contract.engaged is True


def test_engaged_when_logprobs_is_none():
    contract = extract_output_contract(_req(response_format=_rf(STRICT), logprobs=None))
    assert contract.engaged is True


def test_not_engaged_on_scalar_root():
    c = extract_output_contract(_req(response_format=_rf({"type": "integer"})))
    assert c.engaged is False


def test_not_engaged_on_array_root_even_if_it_looks_discriminating():
    # `additionalProperties` is an OBJECT keyword -- it says nothing about an array,
    # so it cannot reject a nested fragment here. Arrays are also where the ambiguity
    # gate is weakest (`[[1, 2]` is a valid incomplete document). Object roots only.
    # See spec §3.3 and §5.1.
    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "additionalProperties": False,
    }
    c = extract_output_contract(_req(response_format=_rf(schema)))
    assert c.engaged is False


def test_not_engaged_on_non_discriminating_schema():
    # No `required`, and additionalProperties is not False -> the oracle cannot
    # reject a nested fragment. See spec §5.1.
    c = extract_output_contract(_req(response_format=_rf({"type": "object"})))
    assert c.engaged is False
    assert c.rejected_non_discriminating is True


def test_engages_on_additional_properties_false_without_required():
    schema = {"type": "object", "additionalProperties": False}
    c = extract_output_contract(_req(response_format=_rf(schema)))
    assert c.engaged is True


def test_tools_alone_do_not_engage_v1():
    c = extract_output_contract(
        _req(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": STRICT,
                    },
                }
            ]
        )
    )
    assert c.engaged is False


def test_same_schema_in_both_carriers_engages():
    contract = extract_output_contract(
        _req(response_format=_rf(STRICT), structured_outputs={"json": STRICT})
    )
    assert contract.engaged is True
    assert contract.content_schema == STRICT


def test_empty_structured_outputs_does_not_conflict_with_response_format():
    contract = extract_output_contract(
        _req(response_format=_rf(STRICT), structured_outputs={})
    )
    assert contract.engaged is True
    assert contract.content_schema == STRICT


def test_null_structured_outputs_schema_does_not_conflict_with_response_format():
    contract = extract_output_contract(
        _req(response_format=_rf(STRICT), structured_outputs={"json": None})
    )
    assert contract.engaged is True
    assert contract.content_schema == STRICT


def test_response_format_without_schema_does_not_conflict_with_structured_outputs():
    response_format = {"type": "json_schema", "json_schema": {"name": "s"}}
    contract = extract_output_contract(
        _req(response_format=response_format, structured_outputs={"json": STRICT})
    )
    assert contract.engaged is True
    assert contract.content_schema == STRICT


def test_non_dict_structured_outputs_schema_conflicts_with_response_format():
    contract = extract_output_contract(
        _req(response_format=_rf(STRICT), structured_outputs={"json": "nonsense"})
    )
    assert contract.engaged is False
    assert contract.rejected_non_discriminating is False


def test_different_discriminating_schemas_in_both_carriers_do_not_engage():
    other_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    contract = extract_output_contract(
        _req(
            response_format=_rf(STRICT),
            structured_outputs={"json": other_schema},
        )
    )
    assert contract.engaged is False
    assert contract.rejected_non_discriminating is False


def test_discriminating_and_non_discriminating_carriers_do_not_engage():
    contract = extract_output_contract(
        _req(
            response_format=_rf(STRICT),
            structured_outputs={"json": {"type": "object"}},
        )
    )
    assert contract.engaged is False
    assert contract.rejected_non_discriminating is False


def test_only_structured_outputs_json_engages():
    c = extract_output_contract(_req(structured_outputs={"json": STRICT}))
    assert c.engaged is True
    assert c.content_schema == STRICT


def test_only_response_format_json_schema_engages():
    contract = extract_output_contract(_req(response_format=_rf(STRICT)))
    assert contract.engaged is True
    assert contract.content_schema == STRICT


def test_malformed_request_json_never_raises():
    assert extract_output_contract({"response_format": "nonsense"}).engaged is False
    assert (
        extract_output_contract({"structured_outputs": {"json": None}}).engaged is False
    )
