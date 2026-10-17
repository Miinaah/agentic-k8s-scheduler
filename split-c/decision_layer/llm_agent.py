"""
decision_layer/llm_agent.py

LLM-based priority order reasoning agent with validation, timeout enforcement,
and rule-based fallback.

Contract:
- Anthropic Python SDK client wrapped behind an injectable protocol.
- Reads API key from ANTHROPIC_API_KEY and model name from LLM_MODEL.
- Uses temperature 0.
- System prompt instructs JSON-only response with the 3 objectives:
  ["fault_tolerance", "utilisation", "cost"].
- 4 few-shot examples: fault-heavy, cost-sensitive, calm/default, conflicting-signals.
- Strips markdown code fences before Pydantic validation.
- On timeout, API error, or invalid JSON, falls back to rule_based_priority from stress.py.
- Logs execution latency and fallback status. Never raises to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Protocol, runtime_checkable

from decision_layer.schemas import (
    ClusterState,
    Objective,
    PriorityOrder,
    VALID_OBJECTIVES,
)
from decision_layer.stress import compute_stress, rule_based_priority

logger = logging.getLogger(__name__)

# ==============================================================================
# System Prompt & Few-Shot Examples
# ==============================================================================
SYSTEM_PROMPT: str = """You are a Kubernetes scheduling policy advisor.
Your role is to evaluate real-time cluster telemetry (resource utilisation, recent failures, event logs, upcoming maintenance or traffic surges) and determine the strict priority order of scheduling objectives.

The three valid objectives are exactly:
- "fault_tolerance"
- "utilisation"
- "cost"

Output MUST be strictly valid JSON without preamble, commentary, or markdown fences. The JSON schema is:
{
  "priority_order": ["<objective_1>", "<objective_2>", "<objective_3>"],
  "justification": "<concise explanation of reasoning>",
  "confidence": <float between 0.0 and 1.0>
}

Constraints:
1. "priority_order" must contain ALL THREE objectives and NO DUPLICATES.
2. "confidence" must be between 0.0 and 1.0.
3. Do not include extra fields or Markdown outside the JSON.

--- FEW-SHOT EXAMPLES ---

[Example 1: Fault-Heavy Cluster]
Telemetry:
- Nodes reporting elevated OOM kills; 4 pod crash-loop-backoffs in the last 15 minutes.
- Cluster CPU utilisation is 78%, memory at 82%. Node condition flapping on worker-02.
Output:
{
  "priority_order": ["fault_tolerance", "utilisation", "cost"],
  "justification": "Elevated OOM kills, crash loops, and flapping node conditions indicate immediate stability risks. Prioritizing fault tolerance to prevent cascading outages.",
  "confidence": 0.95
}

[Example 2: Cost-Sensitive Scenario]
Telemetry:
- Off-peak nocturnal window; monthly cloud budget alert at 94% threshold.
- Non-critical batch analytics pending; all 5 nodes healthy with 0 recent failures; average utilisation at 35%.
Output:
{
  "priority_order": ["cost", "utilisation", "fault_tolerance"],
  "justification": "Zero failure events, low nocturnal traffic, and strict budget alert warrant aggressive cost minimization and spot instance selection.",
  "confidence": 0.92
}

[Example 3: Calm / Default Cluster]
Telemetry:
- All 5 nodes reporting 1.0 health scores, 0 failures across zones.
- Steady baseline CPU at 48% and RAM at 52%. Standard burstable web-app pods incoming.
Output:
{
  "priority_order": ["utilisation", "fault_tolerance", "cost"],
  "justification": "Cluster is fully healthy with balanced steady-state load. Prioritizing resource utilisation to optimize bin-packing efficiency while retaining moderate resilience.",
  "confidence": 0.88
}

[Example 4: Conflicting Signals]
Telemetry:
- Flash sale event starting in 20 minutes (expected 3x traffic surge).
- Worker-03 has sporadic memory pressure warnings, but engineering leadership issued a freeze on expensive on-demand scaling.
Output:
{
  "priority_order": ["fault_tolerance", "cost", "utilisation"],
  "justification": "Imminent high-consequence traffic surge combined with worker memory pressure demands fault tolerance first, while budget freeze keeps cost above utilisation packing.",
  "confidence": 0.83
}
"""


# ==============================================================================
# LLM Client Interface / Protocol
# ==============================================================================
@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations."""

    def generate(self, system: str, prompt: str, timeout_s: float = 5.0) -> str:
        """Generate response text given system instructions and user prompt."""
        ...


class AnthropicLLMClient:
    """Production Anthropic SDK client."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL")

    def generate(self, system: str, prompt: str, timeout_s: float = 5.0) -> str:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        if not self.model:
            raise ValueError("LLM_MODEL environment variable is not set")

        import anthropic  # Lazy import to allow module use without SDK installed if mocked

        client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout_s)
        response = client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from response content blocks
        text_parts = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text
        ]
        return "".join(text_parts)


class FakeLLMClient:
    """
    Fake LLM client for tests and offline simulation.
    """

    def __init__(
        self,
        response: str | None = None,
        side_effect: Exception | None = None,
        latency_s: float = 0.0,
    ) -> None:
        self.response = response
        self.side_effect = side_effect
        self.latency_s = latency_s
        self.call_history: list[dict[str, Any]] = []

    def generate(self, system: str, prompt: str, timeout_s: float = 5.0) -> str:
        self.call_history.append({"system": system, "prompt": prompt, "timeout_s": timeout_s})

        if self.latency_s > 0:
            time.sleep(self.latency_s)

        if timeout_s is not None and self.latency_s > timeout_s:
            raise TimeoutError(f"LLM request timed out after {timeout_s}s (latency: {self.latency_s}s)")

        if self.side_effect is not None:
            raise self.side_effect

        if self.response is None:
            raise ValueError("No mock response configured in FakeLLMClient")

        return self.response


# Default client factory
_default_client: LLMClientProtocol | None = None


def get_default_llm_client() -> LLMClientProtocol:
    """Get or instantiate default Anthropic client."""
    global _default_client
    if _default_client is None:
        _default_client = AnthropicLLMClient()
    return _default_client


def set_default_llm_client(client: LLMClientProtocol | None) -> None:
    """Override or reset default LLM client."""
    global _default_client
    _default_client = client


# ==============================================================================
# Response Parsing and Sanitization
# ==============================================================================
def strip_markdown_fences(raw_text: str) -> str:
    """
    Strip surrounding ```json ... ``` or ``` ... ``` markdown fences and extract JSON.
    """
    text = raw_text.strip()

    # Remove outer triple-backtick fences if present
    fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    fence_match = re.search(fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # If text still contains non-JSON wrappers, locate JSON object bounds
    if not (text.startswith("{") and text.endswith("}")):
        obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if obj_match:
            text = obj_match.group(1).strip()

    return text


def _format_telemetry_prompt(telemetry_summary: Any) -> str:
    """Format input telemetry summary into an LLM prompt string."""
    if isinstance(telemetry_summary, str):
        return telemetry_summary
    if isinstance(telemetry_summary, ClusterState):
        return telemetry_summary.model_dump_json(indent=2)
    if isinstance(telemetry_summary, dict):
        return json.dumps(telemetry_summary, indent=2)
    return str(telemetry_summary)


def _execute_rule_fallback(
    telemetry_summary: Any,
    error_reason: str,
) -> PriorityOrder:
    """Execute rule-based fallback from stress.py and ensure source='rule'."""
    # Determine base stress
    if isinstance(telemetry_summary, ClusterState) or (
        isinstance(telemetry_summary, dict) and "nodes" in telemetry_summary
    ):
        try:
            stress = compute_stress(telemetry_summary)
            fallback = rule_based_priority(stress)
        except Exception:
            fallback = rule_based_priority("medium")
    elif isinstance(telemetry_summary, str):
        lower_text = telemetry_summary.lower()
        if any(w in lower_text for w in ("fail", "crash", "oom", "stress", "high", "reboot")):
            fallback = rule_based_priority("high")
        elif any(w in lower_text for w in ("idle", "calm", "budget", "cost", "low")):
            fallback = rule_based_priority("low")
        else:
            fallback = rule_based_priority("medium")
    else:
        fallback = rule_based_priority("medium")

    # Construct safe fallback PriorityOrder with source="rule"
    justification = f"Fallback rule applied ({error_reason}): {fallback.justification}"
    return PriorityOrder(
        priority_order=fallback.priority_order,
        justification=justification,
        confidence=fallback.confidence,
        source="rule",
    )


# ==============================================================================
# Main LLM Agent Function
# ==============================================================================
def get_priority_order(
    telemetry_summary: Any,
    timeout_s: float = 5.0,
    client: LLMClientProtocol | None = None,
) -> PriorityOrder:
    """
    Obtain a PriorityOrder by invoking the LLM reasoning agent with validation,
    timeout handling, and deterministic rule-based fallback.

    Parameters:
    - telemetry_summary: Cluster telemetry string, dict, or ClusterState.
    - timeout_s: Maximum seconds allowed for the LLM request (default: 5.0).
    - client: Optional injectable LLMClientProtocol (for tests / fakes).

    Returns:
    - PriorityOrder: Validated scheduling priority order (source='llm' or 'rule').
      Guaranteed never to raise exceptions to caller.
    """
    start_time = time.perf_counter()
    fallback_used = False
    llm_client = client if client is not None else get_default_llm_client()

    prompt = _format_telemetry_prompt(telemetry_summary)

    try:
        # Call LLM client
        raw_response = llm_client.generate(
            system=SYSTEM_PROMPT,
            prompt=prompt,
            timeout_s=timeout_s,
        )

        # Strip markdown fences and extract JSON
        cleaned_json = strip_markdown_fences(raw_response)

        # Parse JSON
        parsed_data = json.loads(cleaned_json)
        if not isinstance(parsed_data, dict):
            raise ValueError(f"LLM output is not a JSON object: {type(parsed_data).__name__}")

        # Set source to 'llm' before Pydantic validation
        parsed_data["source"] = "llm"

        # Validate with Pydantic PriorityOrder schema
        # (checks 3 objectives, no duplicates, valid objective names, confidence in [0, 1])
        decision = PriorityOrder.model_validate(parsed_data)

    except Exception as exc:
        fallback_used = True
        error_name = type(exc).__name__
        logger.warning(
            "LLM agent failed (%s: %s). Falling back to rule_based_priority.",
            error_name,
            exc,
        )
        decision = _execute_rule_fallback(telemetry_summary, error_reason=error_name)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "get_priority_order completed in %.2f ms | fallback_used=%s | source=%s | order=%s",
        elapsed_ms,
        fallback_used,
        decision.source,
        decision.priority_order,
    )

    return decision
