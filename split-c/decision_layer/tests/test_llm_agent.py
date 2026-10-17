"""
decision_layer/tests/test_llm_agent.py

Unit tests for LLM priority reasoning agent with FakeLLMClient.
"""

import json
import logging
import pytest

from decision_layer.llm_agent import (
    AnthropicLLMClient,
    FakeLLMClient,
    get_priority_order,
    strip_markdown_fences,
)
from decision_layer.schemas import PriorityOrder, VALID_OBJECTIVES


def test_valid_output_passes():
    """
    Test 1: Valid JSON output from LLM passes validation and returns source='llm'.
    """
    mock_payload = {
        "priority_order": ["utilisation", "cost", "fault_tolerance"],
        "justification": "Steady cluster load, bin-packing efficiency prioritized.",
        "confidence": 0.89,
    }
    client = FakeLLMClient(response=json.dumps(mock_payload))

    decision = get_priority_order("Calm cluster telemetry: 45% util, 0 failures", client=client)

    assert isinstance(decision, PriorityOrder)
    assert decision.source == "llm"
    assert decision.priority_order == ["utilisation", "cost", "fault_tolerance"]
    assert decision.confidence == 0.89
    assert "Steady cluster load" in decision.justification


def test_malformed_json_falls_back():
    """
    Test 2: Malformed JSON output gracefully falls back to rule_based_priority with source='rule'.
    Never raises an exception to the caller.
    """
    client = FakeLLMClient(response="I recommend the following: {priority_order: [broken json...")

    decision = get_priority_order("Telemetry with high errors", client=client)

    assert isinstance(decision, PriorityOrder)
    assert decision.source == "rule"
    assert set(decision.priority_order) == set(VALID_OBJECTIVES)
    assert "Fallback rule applied" in decision.justification


def test_duplicate_objective_output_falls_back():
    """
    Test 3: LLM output with duplicate objectives fails Pydantic validation
    and safely falls back to rule_based_priority with source='rule'.
    """
    mock_payload = {
        "priority_order": ["cost", "cost", "utilisation"],
        "justification": "Budget is tight, so cost twice.",
        "confidence": 0.95,
    }
    client = FakeLLMClient(response=json.dumps(mock_payload))

    decision = get_priority_order("Telemetry: low budget remaining", client=client)

    assert isinstance(decision, PriorityOrder)
    assert decision.source == "rule"
    assert len(set(decision.priority_order)) == 3
    assert "Fallback rule applied" in decision.justification


def test_timeout_falls_back():
    """
    Test 4: LLM call timeout triggers fallback with source='rule'.
    """
    # Fake client simulating 6.0s latency when timeout is 1.0s
    client = FakeLLMClient(
        response=json.dumps({
            "priority_order": ["fault_tolerance", "utilisation", "cost"],
            "justification": "Too late",
            "confidence": 0.9,
        }),
        latency_s=0.2,
    )

    decision = get_priority_order("Telemetry data", timeout_s=0.05, client=client)

    assert isinstance(decision, PriorityOrder)
    assert decision.source == "rule"
    assert set(decision.priority_order) == set(VALID_OBJECTIVES)


def test_fenced_json_gets_parsed():
    """
    Test 5: LLM output enclosed in markdown code fences (```json ... ```)
    is properly stripped and successfully parsed as source='llm'.
    """
    fenced_response = """```json
{
  "priority_order": ["fault_tolerance", "cost", "utilisation"],
  "justification": "Flash-sale traffic surge imminent with unstable worker node.",
  "confidence": 0.85
}
```"""
    client = FakeLLMClient(response=fenced_response)

    decision = get_priority_order("Imminent traffic spike telemetry", client=client)

    assert isinstance(decision, PriorityOrder)
    assert decision.source == "llm"
    assert decision.priority_order == ["fault_tolerance", "cost", "utilisation"]
    assert decision.confidence == 0.85


def test_api_error_falls_back():
    """
    Anthropic API network/service error triggers rule-based fallback.
    """
    client = FakeLLMClient(side_effect=RuntimeError("Anthropic API HTTP 503 Service Unavailable"))

    decision = get_priority_order("Telemetry data", client=client)

    assert isinstance(decision, PriorityOrder)
    assert decision.source == "rule"
    assert "Fallback rule applied" in decision.justification


def test_missing_objective_falls_back():
    """
    Output with only 2 objectives fails validation and falls back.
    """
    incomplete_payload = {
        "priority_order": ["fault_tolerance", "utilisation"],
        "justification": "Forgot cost",
        "confidence": 0.7,
    }
    client = FakeLLMClient(response=json.dumps(incomplete_payload))

    decision = get_priority_order("Telemetry", client=client)

    assert decision.source == "rule"
    assert len(decision.priority_order) == 3


def test_confidence_out_of_range_falls_back():
    """
    Confidence > 1.0 fails validation and falls back.
    """
    invalid_confidence = {
        "priority_order": ["fault_tolerance", "utilisation", "cost"],
        "justification": "Overconfident",
        "confidence": 1.5,
    }
    client = FakeLLMClient(response=json.dumps(invalid_confidence))

    decision = get_priority_order("Telemetry", client=client)

    assert decision.source == "rule"


def test_strip_markdown_fences_variations():
    """Verify various markdown fence wrappers are correctly stripped."""
    # Plain text
    assert strip_markdown_fences('{"a": 1}') == '{"a": 1}'

    # ```json fence
    assert strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    # ``` fence without language tag
    assert strip_markdown_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    # Text before/after fence
    assert strip_markdown_fences('Here is the json:\n```json\n{"a": 1}\n```\nHope it helps!') == '{"a": 1}'


def test_anthropic_client_env_var_validation(monkeypatch):
    """AnthropicLLMClient raises ValueError when environment variables are missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    client = AnthropicLLMClient()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        client.generate("system", "prompt")

    client_with_key = AnthropicLLMClient(api_key="sk-test-key")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        client_with_key.generate("system", "prompt")


def test_logging_latency_and_fallback(caplog):
    """Verify latency and fallback status are logged."""
    client = FakeLLMClient(response="invalid json")
    with caplog.at_level(logging.INFO):
        decision = get_priority_order("Telemetry", client=client)

    assert decision.source == "rule"
    log_text = caplog.text
    assert "fallback_used=True" in log_text
    assert "completed in" in log_text
