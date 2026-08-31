"""
utils.py
--------
Small, focused helper functions:
- Safely parsing JSON that an LLM returned (LLMs sometimes wrap JSON in
  markdown fences, or add stray text before/after it).
- Validating that parsed JSON matches the schema we expect.
- A couple of generic error-handling helpers.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_KEYS = [
    "financial_summary",
    "financial_health_score",
    "spending_analysis",
    "risk_level",
    "top_priorities",
    "budget_recommendations",
    "savings_strategy",
    "next_month_action_plan",
]


def extract_json_block(raw_text: str) -> Optional[str]:
    """
    Try to pull a JSON object out of raw LLM text, even if the model added
    markdown code fences or extra commentary around it.
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # Remove markdown code fences like ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Fall back to grabbing the first { ... last } span in the text.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return None


def safe_parse_llm_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Attempt to safely parse JSON returned by the LLM.

    Returns:
        (parsed_dict, error_message)
        Exactly one of the two will be None.
    """
    candidate = extract_json_block(raw_text)
    if candidate is None:
        return None, "No JSON object could be found in the AI response."

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"The AI response was not valid JSON ({exc})."

    is_valid, validation_error = validate_analysis_schema(parsed)
    if not is_valid:
        return None, validation_error

    return parsed, None


def validate_analysis_schema(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check that parsed JSON has the keys/types we require. Never raises."""
    if not isinstance(data, dict):
        return False, "AI response was not a JSON object."

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        return False, f"AI response is missing required fields: {', '.join(missing)}"

    # Clamp / coerce the score defensively instead of failing outright.
    try:
        score = float(data["financial_health_score"])
        data["financial_health_score"] = max(0, min(100, round(score)))
    except (TypeError, ValueError):
        return False, "financial_health_score was not a valid number."

    if not isinstance(data.get("spending_analysis"), list):
        return False, "spending_analysis must be a list."

    for list_field in [
        "top_priorities",
        "budget_recommendations",
        "savings_strategy",
        "next_month_action_plan",
    ]:
        if not isinstance(data.get(list_field), list):
            return False, f"{list_field} must be a list."

    if data.get("risk_level") not in ("LOW", "MEDIUM", "HIGH"):
        # Don't hard-fail on this -- normalize instead, since models
        # sometimes use lowercase or slightly different wording.
        risk_text = str(data.get("risk_level", "")).upper()
        data["risk_level"] = risk_text if risk_text in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"

    return True, None


def validate_positive_number(value: Any, field_name: str) -> Tuple[bool, Optional[str]]:
    """Generic numeric validation used by the input form."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} must be a valid number."
    if number < 0:
        return False, f"{field_name} cannot be negative."
    return True, None


def format_currency(amount: float, currency: str) -> str:
    """Format a number as a currency string for display, e.g. '1,234.00 USD'."""
    try:
        return f"{amount:,.2f} {currency}"
    except (TypeError, ValueError):
        return f"{amount} {currency}"
