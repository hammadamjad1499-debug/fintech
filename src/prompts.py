"""
prompts.py
----------
All prompt engineering lives here: the reusable PromptTemplate, the
ChatPromptTemplate (system + human messages), the safety rules, and the
JSON schema description we ask the model to follow.

Nothing in this file calls the LLM -- it only BUILDS prompts. The actual
calling happens in chains.py.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# ---------------------------------------------------------------------------
# JSON SCHEMA (as text, inserted into the prompt so the model knows exactly
# what shape of JSON we expect back)
# ---------------------------------------------------------------------------

JSON_SCHEMA_DESCRIPTION = """
Return ONLY a single valid JSON object (no markdown fences, no extra text)
with EXACTLY this structure:

{{
  "financial_summary": "string - a short 2-3 sentence overview",
  "financial_health_score": 0,
  "spending_analysis": [
    {{
      "category": "string",
      "observation": "string",
      "recommendation": "string"
    }}
  ],
  "risk_level": "LOW | MEDIUM | HIGH",
  "top_priorities": ["string", "string"],
  "budget_recommendations": ["string", "string"],
  "savings_strategy": ["string", "string"],
  "next_month_action_plan": ["string", "string"]
}}

Rules for the JSON:
- "financial_health_score" must be an integer between 0 and 100.
- Use ONLY the numbers provided to you. Do not invent financial data.
- Do not include any text before or after the JSON object.
"""

# ---------------------------------------------------------------------------
# SAFETY / SYSTEM INSTRUCTIONS
# Reused by both the JSON analysis chain and the streaming chain so the
# assistant behaves consistently everywhere.
# ---------------------------------------------------------------------------

SAFETY_SYSTEM_INSTRUCTIONS = """
You are FinWise AI, an educational financial analysis assistant embedded in
a student FinTech project.

Follow these safety rules at all times:
1. You provide EDUCATIONAL information only, not professional financial advice.
2. Never claim a guaranteed financial outcome.
3. Never claim to execute financial transactions.
4. Never claim to connect to real bank accounts.
5. Never provide investment guarantees or promise specific returns.
6. Always encourage the user to consult a qualified financial professional
   for actual financial decisions.
7. Base your analysis strictly on the numbers provided to you in the prompt.
8. Do not invent financial data, account balances, or transactions that
   were not given to you.
"""

# ---------------------------------------------------------------------------
# REUSABLE PromptTemplate
# Demonstrates the "classic" LangChain PromptTemplate with named variables.
# This is used to build the HUMAN portion of the analysis request.
# ---------------------------------------------------------------------------

FINANCIAL_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=[
        "monthly_income",
        "total_expenses",
        "remaining_income",
        "savings",
        "savings_ratio",
        "expense_ratio",
        "financial_goal",
        "expense_breakdown",
    ],
    template="""
Analyze the following personal finances and respond with educational
insights only.

Monthly income: {monthly_income}
Total expenses: {total_expenses}
Remaining income (income - expenses): {remaining_income}
Current monthly savings: {savings}
Savings ratio: {savings_ratio}%
Expense ratio: {expense_ratio}%
Financial goal: {financial_goal}

Expense breakdown:
{expense_breakdown}

"""
    + JSON_SCHEMA_DESCRIPTION,
)


def build_analysis_prompt_text(values: dict) -> str:
    """Fill the reusable PromptTemplate with real values and return the text."""
    return FINANCIAL_ANALYSIS_PROMPT.format(**values)


# ---------------------------------------------------------------------------
# ChatPromptTemplate (system + human) -- used for both the JSON chain and
# the streaming recommendations chain.
# ---------------------------------------------------------------------------

ANALYSIS_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SAFETY_SYSTEM_INSTRUCTIONS),
        (
            "human",
            "Here is my financial data:\n\n"
            "Monthly income: {monthly_income}\n"
            "Total expenses: {total_expenses}\n"
            "Remaining income: {remaining_income}\n"
            "Current monthly savings: {savings}\n"
            "Savings ratio: {savings_ratio}%\n"
            "Expense ratio: {expense_ratio}%\n"
            "Financial goal: {financial_goal}\n\n"
            "Expense breakdown:\n{expense_breakdown}\n\n"
            + JSON_SCHEMA_DESCRIPTION,
        ),
    ]
)

# A lighter-weight prompt used specifically for the streaming "AI
# Recommendations" section -- plain text, not JSON, so it can stream nicely
# with a typing effect.
STREAMING_RECOMMENDATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SAFETY_SYSTEM_INSTRUCTIONS),
        (
            "human",
            "Based on this financial snapshot, write a short, warm, "
            "encouraging set of educational recommendations in plain "
            "text (use short paragraphs or a simple bullet list, no JSON):\n\n"
            "Monthly income: {monthly_income}\n"
            "Total expenses: {total_expenses}\n"
            "Remaining income: {remaining_income}\n"
            "Savings ratio: {savings_ratio}%\n"
            "Expense ratio: {expense_ratio}%\n"
            "Financial goal: {financial_goal}\n\n"
            "Expense breakdown:\n{expense_breakdown}\n\n"
            "Remember: educational only, no guaranteed outcomes, "
            "recommend consulting a qualified financial professional.",
        ),
    ]
)
