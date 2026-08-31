"""
config.py
---------
Central place for application configuration:
- Environment variable loading
- Model configuration
- Dropdown / form options
- Score bands and risk thresholds

Keeping all of this in one file means if we ever need to change the
OpenAI model name, add a new currency, or add a new financial goal,
we only need to edit this file.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file into the environment. The OpenAI
# API key itself is NOT read from here anymore -- it is entered by the user
# directly in the Streamlit sidebar and kept only in st.session_state for
# the lifetime of the browser session. .env is only used for the optional
# default model override below.
load_dotenv()

# ---------------------------------------------------------------------------
# API / MODEL SETTINGS
# ---------------------------------------------------------------------------

# Selectable models shown in the sidebar "Model Settings" dropdown.
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

# The chat model used for both the JSON analysis chain and the streaming
# recommendations. Kept in one place so it's easy to swap models. Can be
# overridden via the FINWISE_MODEL env var; otherwise it defaults to the
# first entry in MODEL_OPTIONS.
DEFAULT_MODEL_NAME: str = os.getenv("FINWISE_MODEL", MODEL_OPTIONS[0])

DEFAULT_TEMPERATURE: float = 0.3

# ---------------------------------------------------------------------------
# FORM OPTIONS
# ---------------------------------------------------------------------------

FINANCIAL_GOALS = [
    "Save money",
    "Emergency fund",
    "Pay off debt",
    "Vacation",
    "Start a business",
    "Improve budgeting",
]

CURRENCIES = [
    "USD",
    "EUR",
    "GBP",
    "PKR",
    "INR",
    "AED",
    "CAD",
    "AUD",
]

EXPENSE_CATEGORIES = [
    "housing",
    "food",
    "transportation",
    "utilities",
    "education",
    "healthcare",
    "entertainment",
    "debt",
    "other",
]

# Human-friendly labels for the expense categories above, used in the UI
# and when building the expense breakdown text sent to the LLM.
EXPENSE_LABELS = {
    "housing": "Housing/Rent",
    "food": "Food",
    "transportation": "Transportation",
    "utilities": "Utilities",
    "education": "Education",
    "healthcare": "Healthcare",
    "entertainment": "Entertainment",
    "debt": "Loan/Debt",
    "other": "Other",
}

# ---------------------------------------------------------------------------
# SCORE BANDS (used for both the Python preliminary score and the
# AI financial health score so the two are displayed consistently)
# ---------------------------------------------------------------------------

SCORE_BANDS = [
    (80, 100, "Strong"),
    (60, 79, "Generally Healthy"),
    (40, 59, "Needs Improvement"),
    (0, 39, "High Attention"),
]

# ---------------------------------------------------------------------------
# CACHE OPTIONS
# ---------------------------------------------------------------------------

CACHE_OPTIONS = ["In-memory cache", "SQLite cache"]
SQLITE_CACHE_PATH = os.path.join(os.getcwd(), ".finwise_cache.db")

# ---------------------------------------------------------------------------
# DISCLAIMER TEXT (single source of truth, reused across the whole app)
# ---------------------------------------------------------------------------

EDUCATIONAL_DISCLAIMER = (
    "This application is an educational prototype for informational "
    "purposes only. It does not provide guaranteed financial advice, "
    "execute financial transactions, connect to real bank accounts, or "
    "guarantee financial outcomes. Consult a qualified financial "
    "professional for financial decisions."
)
