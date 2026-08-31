https://hammadamjad1499-debug-fintech-app-ixwpkz.streamlit.app/

# FinWise AI -- AI-Powered Personal Financial Analysis & Smart Budget Assistant

## 1. Project Overview

FinWise AI is a Streamlit web application that takes a snapshot of a
user's monthly income, expenses, savings, and financial goal, and produces:

1. **Deterministic Python calculations** -- total expenses, remaining
   income, savings ratio, expense ratio, debt ratio, a preliminary 0-100
   financial score, and a risk level.
2. **AI-generated educational insights** -- via LangChain + OpenAI: a
   structured JSON analysis (summary, spending analysis, priorities,
   budget recommendations, savings strategy, action plan) plus a streamed,
   plain-text "AI Recommendations" section.

The app never executes real transactions, never connects to a bank
account, and never guarantees financial outcomes. It is an **educational
prototype** for a university FinTech / AI assignment.

## 2. Features

- Professional multi-column input form (income, 9 expense categories,
  savings, financial goal, currency).
- Python-only deterministic financial calculations, fully reproducible.
- LangChain `PromptTemplate` and `ChatPromptTemplate` with a safety-first
  system message.
- Structured JSON output from the LLM, validated and safely parsed (never
  crashes on malformed JSON).
- Streaming AI recommendations with a live typing effect
  (`st.write_stream`).
- Switchable LangChain LLM cache: `InMemoryCache` or `SQLiteCache`.
- Full dashboard: metrics, progress bars, tabs, expanders, and
  info/warning/error boxes.
- Graceful error handling for missing/invalid API keys, network errors,
  invalid input, and malformed AI output.
- Visible educational disclaimer throughout the app.

## 3. Technologies

- Python 3.10+
- Streamlit
- LangChain (`langchain`, `langchain-core`, `langchain-community`)
- `langchain-openai` (`ChatOpenAI`)
- OpenAI API
- `python-dotenv`

## 4. Folder Structure

```
finwise_ai/
│
├── app.py                     # Main Streamlit application
├── requirements.txt
├── .env.example
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Settings, env vars, form options
│   ├── prompts.py               # PromptTemplate / ChatPromptTemplate / JSON schema
│   ├── financial_calculator.py  # ONLY deterministic math, no AI
│   ├── chains.py                 # ChatOpenAI, messages, chain, streaming
│   ├── cache_manager.py           # InMemoryCache / SQLiteCache
│   └── utils.py                    # Safe JSON parsing & validation
│
└── docs/
    └── SPEC_SOURCE.md
```

## 5. Installation Instructions

### 5.1 Clone or download the project

```bash
cd finwise_ai
```

### 5.2 Create and activate a virtual environment

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.4 API key setup -- no `.env` editing required

FinWise AI no longer reads an OpenAI API key from `.env` or any
environment variable. Instead, each user enters their **own** key directly
in the app:

1. Run the app (see Section 6).
2. In the sidebar, paste your key into the **"OpenAI API Key"** field.
3. Click **"Connect"**. The key is validated and, once confirmed, kept only
   in that browser session's memory (`st.session_state`) -- it is never
   written to disk, `.env`, cache, or logs.
4. Click **"Clear Key"** at any time to remove it from the session.

`.env` / `.env.example` are only used for the optional `FINWISE_MODEL`
override; they no longer contain any secret.

## 6. Running the Application

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) --
open it in your browser, then connect your API key from the sidebar as
described in Section 5.4.

## 7. How the Python Calculations Work

All math lives in `src/financial_calculator.py` and only that file. Given
the same inputs, it always returns the same outputs -- there is no
randomness and no AI involved.

- `total_expenses = sum(all expense categories)`
- `remaining_income = monthly_income - total_expenses`
- `savings_ratio = (savings / monthly_income) * 100`
- `expense_ratio = (total_expenses / monthly_income) * 100`
- `debt_ratio = (debt_expense / monthly_income) * 100`

The **preliminary Python score** (0-100) is a weighted heuristic:

| Component        | Weight | Best when...                      |
|-------------------|--------|-------------------------------------|
| Savings ratio      | 40%    | savings ratio >= 20% of income       |
| Remaining income    | 30%    | remaining income >= 20% of income     |
| Expense ratio        | 20%    | expenses <= 50% of income               |
| Debt burden            | 10%    | debt is 0% of income                     |

Income = 0, missing values, and invalid input are all handled without
crashing (see `_safe_float()` and the guarded division logic).

## 8. What is LangChain, and Why Use It?

LangChain is a framework for building applications on top of large
language models. It provides reusable building blocks -- prompt
templates, chat message types, model wrappers, output parsing, caching --
so you don't have to hand-roll string formatting and HTTP calls every
time you want to talk to an LLM. Using it here keeps the AI logic modular
and swappable (e.g. changing models later requires touching one line in
`config.py`).

## 9. PromptTemplate

`src/prompts.py` defines `FINANCIAL_ANALYSIS_PROMPT`, a classic
`PromptTemplate` with named variables (`monthly_income`, `total_expenses`,
etc.). Calling `.format(**values)` on it fills in those variables and
returns a single formatted string. This demonstrates the simplest form of
LangChain prompt engineering: a template with placeholders.

## 10. ChatPromptTemplate

`ANALYSIS_CHAT_PROMPT` and `STREAMING_RECOMMENDATIONS_PROMPT` are
`ChatPromptTemplate` objects, which model a **conversation** rather than a
single string -- a system message (the assistant's rules/persona) plus a
human message (the user's data). This is the format modern chat models
like `gpt-4o-mini` expect, and it's what actually gets sent to OpenAI in
this project.

## 11. SystemMessage / HumanMessage / AIMessage

`src/chains.py` includes `demonstrate_messages()`, a small commented
example showing how a conversation is represented as a list of message
objects:

- **SystemMessage** -- sets the assistant's role and safety rules.
- **HumanMessage** -- represents something the user said.
- **AIMessage** -- represents a previous assistant reply (used to give the
  model conversation history).

This function isn't used in the main analysis flow -- it exists purely to
demonstrate the concept clearly, which is useful for a viva explanation.

## 12. LLMChain

The assignment asks for `LLMChain`. In recent LangChain versions,
`LLMChain` is deprecated in favor of **LCEL** (LangChain Expression
Language), where you compose a prompt and a model with the `|` (pipe)
operator: `prompt | llm`. This project's real chain
(`build_analysis_chain()` in `src/chains.py`) uses that modern approach.

`demonstrate_legacy_llmchain()` additionally tries to import and build a
classic `LLMChain` for educational purposes, and falls back to explaining
the LCEL equivalent if the legacy class isn't available in your installed
LangChain version -- so the assignment concept is preserved either way.

## 13. JSON Output

The LLM is instructed (via the schema description embedded in the prompt)
to return **only** a JSON object matching a fixed structure:
`financial_summary`, `financial_health_score`, `spending_analysis`,
`risk_level`, `top_priorities`, `budget_recommendations`,
`savings_strategy`, `next_month_action_plan`.

Structured JSON is used instead of free text because it lets the
Streamlit dashboard render each piece (score, priorities, action plan) in
its own UI element, rather than trying to parse a paragraph of prose.

`src/utils.py` safely extracts and validates this JSON -- handling
markdown code fences, stray text, missing fields, and invalid types --
without ever crashing the app.

## 14. Streaming

`stream_recommendations()` in `src/chains.py` formats the streaming
prompt, calls `llm.stream(...)` (instead of `.invoke(...)`), and `yield`s
each chunk's text as it arrives. `app.py` passes this generator directly
to `st.write_stream()`, which renders the text with a live typing effect
in the "AI Recommendations" section.

## 15. Caching

LangChain's global LLM cache is configured in `src/cache_manager.py`:

- **InMemoryCache** -- stored in RAM, very fast, lost when the app
  restarts. Good for a single working session.
- **SQLiteCache** -- stored on disk (`.finwise_cache.db`), survives app
  restarts, slightly slower than RAM, but lets you reuse a previous
  response for an identical prompt instead of calling the OpenAI API
  again (saving time and cost).

Switch between them from the sidebar. The "Reset Session" button clears
both the on-screen results and rebuilds a fresh cache.

## 16. Python Score vs. AI Score

| | Preliminary Python Score | AI Financial Health Score |
|---|---|---|
| **Calculated by** | Pure Python arithmetic | The LLM (OpenAI model) |
| **Reproducibility** | Always identical for identical inputs | May vary slightly between calls |
| **Basis** | A fixed weighted formula | The same numbers, interpreted qualitatively |
| **Purpose** | A trustworthy, deterministic baseline | Educational, human-readable interpretation |

Both use the same score bands: 80-100 Strong, 60-79 Generally Healthy,
40-59 Needs Improvement, below 40 High Attention.

## 17. Testing Scenarios

| # | Income | Key Expense | Expected Result |
|---|--------|--------------|-------------------|
| 1 | 8000 | ~2000 total expenses | High score, LOW risk, growth-focused advice |
| 2 | 2000 | ~2600 total expenses | Negative remaining income, expense ratio > 100%, HIGH risk |
| 3 | 5000 | 2500 debt | High debt ratio, MEDIUM/HIGH risk, debt-reduction priorities |
| 4 | 4000 | 1200 savings | ~30% savings ratio, high score, LOW risk |
| 5 | 3000 | 3000 total expenses | Remaining income = 0, MEDIUM/HIGH risk |

Enter these values into the form to verify the app behaves as expected.

## 18. Educational Disclaimer

> This application is an educational prototype for informational purposes
> only. It does not provide guaranteed financial advice, execute
> financial transactions, connect to real bank accounts, or guarantee
> financial outcomes. Consult a qualified financial professional for
> financial decisions.

This text is defined once in `src/config.py` and shown in the sidebar and
on the main dashboard.

## 19. Troubleshooting

| Problem | Fix |
|---|---|
| "API Key Required" / AI section locked | Paste your key into the sidebar and click **Connect**. |
| "Invalid API key" after clicking Connect | Double-check the key value on platform.openai.com; make sure there's no extra whitespace. |
| App hangs / times out | Check your internet connection; OpenAI's API may be temporarily unavailable. |
| "AI response was not valid JSON" | The app has already caught this safely -- try re-running the analysis. |
| Cache file locked (SQLite) | Close other running instances of the app, or delete `.finwise_cache.db`. |
| `ModuleNotFoundError` | Make sure your virtual environment is activated and `pip install -r requirements.txt` completed successfully. |

## 20. GitHub Submission Instructions

```bash
git init
git add .
git commit -m "FinWise AI - initial submission"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Double-check before pushing:
- `.env` is **not** committed (only `.env.example` should be).
- `.finwise_cache.db` is not committed.
- `requirements.txt` is up to date.
