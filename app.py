"""
app.py
------
FinWise AI -- AI-Powered Personal Financial Analysis and Smart Budget Assistant

This is the main Streamlit entry point. It is intentionally "thin": it
collects form input, calls into src/financial_calculator.py for math and
src/chains.py for AI analysis, and renders the results. Business logic
lives in the src/ modules, not here.

The app is gated behind a centered "Connect to FinWise AI" screen: no
dashboard, form, or analysis is reachable until the user enters a valid
OpenAI API key. The key lives only in st.session_state for the duration
of the browser session -- it is never written to .env, disk, cache, or
logs -- and "Clear API Key" in the sidebar drops the user straight back
to the gate screen.

App flow:
    App starts -> Centered API Key Screen -> Connect AI ->
    Validate/Initialize LLM -> Main FinWise Dashboard

Run with:  streamlit run app.py
"""

import streamlit as st

from src.config import (
    CACHE_OPTIONS,
    CURRENCIES,
    EDUCATIONAL_DISCLAIMER,
    EXPENSE_CATEGORIES,
    EXPENSE_LABELS,
    FINANCIAL_GOALS,
    MODEL_OPTIONS,
    SCORE_BANDS,
)
from src.financial_calculator import (
    FinancialInputs,
    calculate_financials,
    format_expense_breakdown_text,
    score_band_label,
)
from src.cache_manager import configure_cache, reset_cache
from src.chains import run_financial_analysis, stream_recommendations, validate_api_key
from src.utils import format_currency

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinWise AI",
    page_icon="\U0001F4B0",
    layout="wide",
)

# ---------------------------------------------------------------------------
# GLOBAL STYLE
# Card-based, modern FinTech look. Streamlit's own widgets are used for
# every interactive element -- this CSS only styles layout/typography.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .fw-hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
        padding: 2rem 2.4rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 28px rgba(15, 32, 39, 0.25);
    }
    .fw-hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
    .fw-hero p { margin: 0.4rem 0 0 0; font-size: 1rem; opacity: 0.85; }

    .fw-section-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin: 1.6rem 0 0.7rem 0;
    }

    .fw-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.2px;
    }
    .fw-badge-low { background: #e6f7ee; color: #0a7d3a; }
    .fw-badge-medium { background: #fff6e0; color: #9a6b00; }
    .fw-badge-high { background: #fdecea; color: #b3261e; }

    .fw-caption { color: #6b7280; font-size: 0.88rem; }

    /* Centered API-key gate screen */
    .fw-gate-wrap {
        display: flex;
        justify-content: center;
        margin-top: 4vh;
    }
    .fw-gate-card {
        max-width: 460px;
        width: 100%;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 2.4rem 2.2rem 2rem 2.2rem;
        box-shadow: 0 12px 40px rgba(15, 32, 39, 0.10);
        text-align: center;
    }
    .fw-gate-logo {
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
    }
    .fw-gate-title {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 0.2rem 0;
        background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .fw-gate-subtitle {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
        margin: 1.1rem 0 0.2rem 0;
    }
    .fw-gate-desc {
        color: #6b7280;
        font-size: 0.92rem;
        margin: 0 0 1.2rem 0;
    }
    .fw-gate-note {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------------------------

if "cache_choice" not in st.session_state:
    st.session_state.cache_choice = CACHE_OPTIONS[0]
    configure_cache(st.session_state.cache_choice)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_OPTIONS[0]

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "ai_connected" not in st.session_state:
    st.session_state.ai_connected = False
if "ai_status" not in st.session_state:
    st.session_state.ai_status = "⚠️ API Key Required"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "financial_results" not in st.session_state:
    st.session_state.financial_results = None
if "prompt_values" not in st.session_state:
    st.session_state.prompt_values = None
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None
if "raw_llm_text" not in st.session_state:
    st.session_state.raw_llm_text = ""


def reset_session():
    """Clear all analysis results and reset the active cache. Does NOT
    touch the API key -- that is only cleared explicitly via 'Clear API Key'."""
    st.session_state.analysis_result = None
    st.session_state.financial_results = None
    st.session_state.prompt_values = None
    st.session_state.analysis_error = None
    st.session_state.raw_llm_text = ""
    reset_cache(st.session_state.cache_choice)


def connect_ai():
    """Validate the entered key and, if valid, activate it for this session."""
    entered_key = st.session_state.get("api_key_input", "")
    is_valid, message = validate_api_key(entered_key)
    st.session_state.ai_connected = is_valid
    st.session_state.openai_api_key = entered_key.strip() if is_valid else ""
    st.session_state.ai_status = message


def clear_api_key():
    """Remove the key from session state entirely and reset connection status."""
    st.session_state.openai_api_key = ""
    st.session_state.ai_connected = False
    st.session_state.ai_status = "⚠️ API Key Required"
    st.session_state.api_key_input = ""


# ---------------------------------------------------------------------------
# GATE SCREEN -- shown instead of everything else until a valid API key is
# connected. This is the application's entry point: no dashboard, no form,
# no sidebar settings are rendered before this point.
# ---------------------------------------------------------------------------

if not st.session_state.ai_connected:
    st.markdown('<div class="fw-gate-wrap">', unsafe_allow_html=True)
    _, gate_col, _ = st.columns([1, 1.3, 1])
    with gate_col:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center;">
                    <div class="fw-gate-logo">\U0001F4B0</div>
                    <div class="fw-gate-title">FinWise AI</div>
                    <div class="fw-gate-subtitle">\U0001F510 Connect to FinWise AI</div>
                    <div class="fw-gate-desc">Enter your OpenAI API key to continue.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_input(
                "OpenAI API Key",
                type="password",
                key="api_key_input",
                placeholder="sk-...",
                label_visibility="collapsed",
            )
            st.button(
                "\U0001F50C Connect AI",
                on_click=connect_ai,
                use_container_width=True,
                type="primary",
            )
            if st.session_state.ai_status and st.session_state.ai_status != "⚠️ API Key Required":
                st.error(st.session_state.ai_status)
            st.markdown(
                '<div class="fw-gate-note">Your API key is used only for '
                "this session and is not stored by the application.</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR (only reached once the AI connection is established)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## \U0001F4B0 FinWise AI")
    st.caption("AI-Powered Personal Financial Analysis & Smart Budget Assistant")

    st.markdown("### \U0001F510 AI Connection")
    st.success("✓ AI Connected")
    if st.button("\U0001F5D1️ Clear API Key", use_container_width=True):
        clear_api_key()
        st.rerun()

    st.markdown("### ⚙️ Model Settings")
    st.selectbox(
        "OpenAI model",
        MODEL_OPTIONS,
        key="selected_model",
        help="Used for both the structured analysis and streaming recommendations.",
    )

    st.markdown("### \U0001F5C4️ Cache Settings")
    new_cache_choice = st.selectbox(
        "LLM response cache",
        CACHE_OPTIONS,
        index=CACHE_OPTIONS.index(st.session_state.cache_choice),
        help=(
            "In-memory cache: RAM only, very fast, cleared on restart.\n\n"
            "SQLite cache: saved to disk, survives restarts, reuses "
            "previous identical responses to save API calls."
        ),
    )
    if new_cache_choice != st.session_state.cache_choice:
        st.session_state.cache_choice = new_cache_choice
        cache_status = configure_cache(new_cache_choice)
        st.success(cache_status)

    with st.expander("How does caching work?"):
        st.markdown(
            "- **InMemoryCache**: stored in RAM, very fast, lost when the "
            "app restarts. Good for a single session.\n"
            "- **SQLiteCache**: stored on disk, survives restarts, "
            "slightly slower than RAM. Repeated identical prompts can "
            "reuse a cached response instead of calling the API again."
        )

    st.markdown("---")
    if st.button("\U0001F504 Reset Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("### ⚠️ Educational Disclaimer")
    st.info(EDUCATIONAL_DISCLAIMER)

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="fw-hero">
        <h1>\U0001F4B0 FinWise AI</h1>
        <p>Smart Budget Assistant &mdash; combining deterministic Python
        calculations with AI-powered educational insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(EDUCATIONAL_DISCLAIMER)

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------

st.markdown('<div class="fw-section-title">\U0001F4DD Your Financial Information</div>', unsafe_allow_html=True)

with st.form("financial_form"):
    with st.container(border=True):
        st.markdown("##### \U0001F4B5 Income & Savings")
        inc_col, sav_col, cur_col, goal_col = st.columns(4)
        with inc_col:
            monthly_income = st.number_input(
                "Monthly Income", min_value=0.0, step=100.0, value=0.0, format="%.2f"
            )
        with sav_col:
            savings = st.number_input(
                "Current Monthly Savings", min_value=0.0, step=50.0, value=0.0, format="%.2f"
            )
        with cur_col:
            currency = st.selectbox("Currency", CURRENCIES)
        with goal_col:
            financial_goal = st.selectbox("Financial Goal", FINANCIAL_GOALS)

    with st.container(border=True):
        st.markdown("##### \U0001F9FE Monthly Expenses")
        expenses = {}
        expense_cols = st.columns(3)
        for i, category in enumerate(EXPENSE_CATEGORIES):
            target_col = expense_cols[i % 3]
            with target_col:
                expenses[category] = st.number_input(
                    EXPENSE_LABELS[category],
                    min_value=0.0,
                    step=10.0,
                    value=0.0,
                    format="%.2f",
                    key=f"expense_{category}",
                )

    st.markdown("")
    submitted = st.form_submit_button("✨ Analyze My Finances", use_container_width=True)

if submitted:
    if monthly_income == 0:
        st.error(
            "Monthly income is 0. You can still see calculations, but "
            "ratios that depend on income won't be meaningful."
        )

    inputs = FinancialInputs(
        monthly_income=monthly_income,
        expenses=expenses,
        savings=savings,
        financial_goal=financial_goal,
        currency=currency,
    )
    results = calculate_financials(inputs)
    st.session_state.financial_results = results

    for warning_text in results.warnings:
        st.warning(warning_text)

    prompt_values = {
        "monthly_income": monthly_income,
        "total_expenses": results.total_expenses,
        "remaining_income": results.remaining_income,
        "savings": savings,
        "savings_ratio": results.savings_ratio,
        "expense_ratio": results.expense_ratio,
        "financial_goal": financial_goal,
        "expense_breakdown": format_expense_breakdown_text(results),
    }
    st.session_state.prompt_values = prompt_values

    if st.session_state.openai_api_key:
        with st.spinner("Analyzing your finances with AI..."):
            parsed, error, raw_text = run_financial_analysis(
                prompt_values,
                api_key=st.session_state.openai_api_key,
                model_name=st.session_state.selected_model,
            )
        st.session_state.analysis_result = parsed
        st.session_state.analysis_error = error
        st.session_state.raw_llm_text = raw_text
    else:
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.session_state.raw_llm_text = ""

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

results = st.session_state.financial_results
analysis = st.session_state.analysis_result
analysis_error = st.session_state.analysis_error
prompt_values = st.session_state.prompt_values

if results is not None:
    st.markdown('<div class="fw-section-title">\U0001F4CA Financial Overview</div>', unsafe_allow_html=True)

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    with overview_col1, st.container(border=True):
        st.metric("Monthly Income", format_currency(prompt_values["monthly_income"], currency))
    with overview_col2, st.container(border=True):
        st.metric("Total Expenses", format_currency(results.total_expenses, currency))
    with overview_col3, st.container(border=True):
        st.metric(
            "Remaining Balance",
            format_currency(results.remaining_income, currency),
            delta=None if results.remaining_income >= 0 else "Negative",
        )
    with overview_col4, st.container(border=True):
        st.metric("Current Savings", format_currency(prompt_values["savings"], currency))

    st.markdown('<div class="fw-section-title">\U0001F4C8 Financial Ratios</div>', unsafe_allow_html=True)
    ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
    with ratio_col1, st.container(border=True):
        st.metric("Savings Ratio", f"{results.savings_ratio}%")
    with ratio_col2, st.container(border=True):
        st.metric("Expense Ratio", f"{results.expense_ratio}%")
    with ratio_col3, st.container(border=True):
        st.metric("Debt Burden", f"{results.debt_ratio}%")

    st.markdown('<div class="fw-section-title">\U0001F3AF Financial Health</div>', unsafe_allow_html=True)
    score_col1, score_col2 = st.columns(2)

    with score_col1, st.container(border=True):
        st.markdown("**Preliminary Python Score**")
        st.caption("Calculated deterministically in Python from your numbers only.")
        st.progress(int(results.preliminary_score) / 100)
        st.write(
            f"{results.preliminary_score} / 100 "
            f"({score_band_label(results.preliminary_score, SCORE_BANDS)})"
        )

    with score_col2, st.container(border=True):
        st.markdown("**AI Financial Health Score**")
        st.caption("Generated by the LLM based on the same numbers, plus qualitative judgment.")
        if analysis:
            ai_score = analysis["financial_health_score"]
            st.progress(int(ai_score) / 100)
            st.write(f"{ai_score} / 100 ({score_band_label(ai_score, SCORE_BANDS)})")
        else:
            st.info("Run an analysis to see the AI score.")

    risk_display = analysis["risk_level"] if analysis else results.risk_level
    risk_badge_class = {"LOW": "fw-badge-low", "MEDIUM": "fw-badge-medium", "HIGH": "fw-badge-high"}.get(
        risk_display, "fw-badge-medium"
    )
    st.markdown(
        f'<span class="fw-badge {risk_badge_class}">\U0001F6A6 Risk Level: {risk_display}</span>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # AI ANALYSIS SECTIONS
    # -----------------------------------------------------------------
    if analysis_error:
        st.markdown('<div class="fw-section-title">\U0001F916 AI Analysis</div>', unsafe_allow_html=True)
        st.error(f"The AI analysis could not be displayed: {analysis_error}")
        with st.expander("Show raw AI response (for debugging)"):
            st.code(st.session_state.raw_llm_text or "(empty response)")
    elif analysis:
        st.markdown('<div class="fw-section-title">\U0001F916 AI Analysis</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.write(analysis["financial_summary"])

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "\U0001F50D Spending Analysis",
                "\U0001F3AF Top Priorities",
                "\U0001F4A1 Budget Recommendations",
                "\U0001F4B0 Savings Strategy",
                "\U0001F4C5 Next Month Plan",
            ]
        )

        with tab1:
            if analysis["spending_analysis"]:
                for item in analysis["spending_analysis"]:
                    with st.expander(f"\U0001F4CB {item.get('category', 'Category')}"):
                        st.write(f"**Observation:** {item.get('observation', '')}")
                        st.write(f"**Recommendation:** {item.get('recommendation', '')}")
            else:
                st.info("No category-level analysis was returned.")

        with tab2:
            for priority in analysis["top_priorities"]:
                st.markdown(f"- {priority}")

        with tab3:
            for rec in analysis["budget_recommendations"]:
                st.markdown(f"- {rec}")

        with tab4:
            for strategy in analysis["savings_strategy"]:
                st.markdown(f"- {strategy}")

        with tab5:
            for step in analysis["next_month_action_plan"]:
                st.markdown(f"- {step}")

        st.markdown('<div class="fw-section-title">\U0001F916 Streaming AI Recommendations</div>', unsafe_allow_html=True)
        st.caption("Streamed live from the model -- educational insights only.")
        if st.button("✨ Generate Streaming Recommendations"):
            st.write_stream(
                stream_recommendations(
                    prompt_values,
                    api_key=st.session_state.openai_api_key,
                    model_name=st.session_state.selected_model,
                )
            )

    st.markdown("---")
    st.caption(EDUCATIONAL_DISCLAIMER)
else:
    st.info("Fill in the form above and click **Analyze My Finances** to get started.")
