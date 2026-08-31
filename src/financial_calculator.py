"""
financial_calculator.py
------------------------
ONLY deterministic financial math lives here. Nothing in this file talks
to the LLM, Streamlit, or the network. That separation is intentional:

    Same inputs -> always the same outputs.

This is what lets us clearly tell the user "this number came from Python,
not from AI" -- it is 100% reproducible arithmetic.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from src.config import EXPENSE_CATEGORIES, EXPENSE_LABELS


@dataclass
class FinancialInputs:
    """Raw values collected from the Streamlit form."""

    monthly_income: float
    expenses: Dict[str, float]  # keys match EXPENSE_CATEGORIES
    savings: float
    financial_goal: str
    currency: str


@dataclass
class FinancialResults:
    """Everything Python calculates from FinancialInputs."""

    total_expenses: float
    remaining_income: float
    savings_ratio: float
    expense_ratio: float
    debt_ratio: float
    preliminary_score: float
    risk_level: str
    expense_breakdown: List[Dict[str, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _safe_float(value) -> float:
    """
    Convert a form value to a non-negative float.

    Streamlit number_input already restricts input, but we defend here too
    in case this function is ever called from somewhere else (tests, CLI).
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    return number


def calculate_financials(inputs: FinancialInputs) -> FinancialResults:
    """
    Run all deterministic financial calculations required by the assignment:

        total_expenses     = sum(all expense categories)
        remaining_income    = monthly_income - total_expenses
        savings_ratio        = (savings / monthly_income) * 100
        expense_ratio        = (total_expenses / monthly_income) * 100

    Also computes a 0-100 preliminary financial score and a risk level.
    Handles income = 0, missing values, and invalid input gracefully.
    """
    warnings: List[str] = []

    income = _safe_float(inputs.monthly_income)
    savings = _safe_float(inputs.savings)

    # Normalize expenses: make sure every expected category exists,
    # defaulting missing/invalid values to 0 instead of crashing.
    clean_expenses: Dict[str, float] = {}
    for category in EXPENSE_CATEGORIES:
        clean_expenses[category] = _safe_float(inputs.expenses.get(category, 0))

    total_expenses = sum(clean_expenses.values())
    remaining_income = income - total_expenses

    # --- Ratios (guarded against division by zero) -----------------------
    if income == 0:
        savings_ratio = 0.0
        expense_ratio = 0.0
        debt_ratio = 0.0
        warnings.append(
            "Monthly income is 0. Ratios that depend on income cannot be "
            "meaningfully calculated and are shown as 0%."
        )
    else:
        savings_ratio = round((savings / income) * 100, 2)
        expense_ratio = round((total_expenses / income) * 100, 2)
        debt_ratio = round((clean_expenses["debt"] / income) * 100, 2)

    if remaining_income < 0:
        warnings.append(
            "Total expenses exceed monthly income. Remaining income is negative."
        )

    # --- Preliminary 0-100 financial score --------------------------------
    # Weighted heuristic, calculated entirely in Python (no AI involved):
    #   20% -> savings ratio        (more saved relative to income = better)
    #   35% -> remaining income     (positive and healthy vs income = better)
    #   30% -> expense ratio        (lower spending relative to income = better)
    #   15% -> debt burden          (lower debt relative to income = better)
    if income == 0:
        preliminary_score = 0.0
    else:
        # Savings component: 20% savings ratio or higher earns full marks.
        savings_component = min(savings_ratio / 20.0, 1.0) * 20

        # Remaining income component: remaining income >= 30% of income
        # earns full marks; negative remaining income earns 0.
        remaining_ratio = (remaining_income / income) * 100
        remaining_component = max(min(remaining_ratio / 30.0, 1.0), 0.0) * 35

        # Expense ratio component: 50% expense ratio or lower earns full
        # marks; 100%+ or higher earns 0.
        expense_component = max(1.0 - max(expense_ratio - 50, 0) / 50.0, 0.0) * 30
        expense_component = min(expense_component, 30.0)

        # Debt burden component: 0% debt ratio earns full marks; 30%+ earns 0.
        debt_component = max(1.0 - (debt_ratio / 30.0), 0.0) * 15

        preliminary_score = round(
            savings_component + remaining_component + expense_component + debt_component,
            1,
        )
        preliminary_score = max(0.0, min(100.0, preliminary_score))

    # --- Risk level ---------------------------------------------------------
    if income == 0 or remaining_income < 0 or expense_ratio >= 100:
        risk_level = "HIGH"
    elif expense_ratio >= 80 or debt_ratio >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # --- Expense breakdown (for display + for the LLM prompt) --------------
    expense_breakdown = []
    for category in EXPENSE_CATEGORIES:
        amount = clean_expenses[category]
        share = round((amount / total_expenses) * 100, 1) if total_expenses > 0 else 0.0
        expense_breakdown.append(
            {
                "category": EXPENSE_LABELS[category],
                "amount": amount,
                "share_of_expenses_pct": share,
            }
        )

    return FinancialResults(
        total_expenses=round(total_expenses, 2),
        remaining_income=round(remaining_income, 2),
        savings_ratio=savings_ratio,
        expense_ratio=expense_ratio,
        debt_ratio=debt_ratio,
        preliminary_score=preliminary_score,
        risk_level=risk_level,
        expense_breakdown=expense_breakdown,
        warnings=warnings,
    )


def score_band_label(score: float, bands) -> str:
    """Map a 0-100 score to its descriptive band label (Strong/Healthy/etc.)."""
    for low, high, label in bands:
        if low <= score <= high:
            return label
    return "Unknown"


def format_expense_breakdown_text(results: FinancialResults) -> str:
    """
    Turn the expense breakdown into a short human-readable string so it can
    be inserted into the LLM prompt as the `expense_breakdown` variable.
    """
    lines = [
        f"- {item['category']}: {item['amount']:.2f} "
        f"({item['share_of_expenses_pct']}% of total expenses)"
        for item in results.expense_breakdown
        if item["amount"] > 0
    ]
    return "\n".join(lines) if lines else "No expenses were entered."
