"""
chains.py
---------
Everything related to actually talking to the LLM lives here:

- get_chat_model(): builds a configured ChatOpenAI instance from a
  user-supplied API key.
- validate_api_key(): a lightweight, low-cost check used by the sidebar's
  "Connect/Activate AI" button to confirm a key actually works.
- demonstrate_messages(): a small, commented example of
  SystemMessage / HumanMessage / AIMessage, for teaching purposes.
- build_analysis_chain(): the reusable chain that takes financial data
  and returns the LLM's raw text response.
- demonstrate_legacy_llmchain(): shows the classic LLMChain API for
  educational purposes, with a safe fallback if it isn't available in the
  installed LangChain version.
- run_financial_analysis(): the high-level function app.py calls to get
  validated JSON back.
- stream_recommendations(): a generator that yields text chunks for the
  "AI Recommendations" streaming section.

Why a chain at all instead of just calling the model directly?
A "chain" bundles together: (1) a prompt template, (2) the model, and
(3) whatever comes after (parsing). That makes it reusable -- the same
chain can be called from a Streamlit form, a test script, or a CLI.

The OpenAI API key is never read from config/env here -- every function in
this module takes it as an explicit `api_key` argument supplied by app.py
from st.session_state, and it is never logged, cached, or included in any
returned error message.
"""

from typing import Any, Dict, Iterator, Tuple, Optional

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE
from src.prompts import (
    ANALYSIS_CHAT_PROMPT,
    SAFETY_SYSTEM_INSTRUCTIONS,
    STREAMING_RECOMMENDATIONS_PROMPT,
)
from src.utils import safe_parse_llm_json


class MissingAPIKeyError(RuntimeError):
    """Raised when no OpenAI API key has been provided for this session."""


def get_chat_model(
    api_key: str,
    model_name: str = DEFAULT_MODEL_NAME,
    streaming: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOpenAI:
    """
    Build a configured ChatOpenAI instance from a user-supplied API key.

    Raises MissingAPIKeyError early (instead of letting a confusing network
    error happen later) if no API key is present.
    """
    if not api_key:
        raise MissingAPIKeyError(
            "No OpenAI API key provided. Enter your API key in the sidebar "
            "and click 'Connect / Activate AI'."
        )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        streaming=streaming,
    )


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Cheaply verify that a user-supplied OpenAI API key is well-formed and
    actually authenticates, without generating any completion tokens.

    Uses the raw OpenAI client's models.list() endpoint, which only checks
    authentication and does not consume completion/token quota.

    Returns:
        (is_valid, safe_status_message) -- the message never echoes the key
        or any raw exception text that might contain sensitive details.
    """
    key = (api_key or "").strip()
    if not key:
        return False, "⚠️ API Key Required"
    if not key.startswith("sk-") or len(key) < 20:
        return False, "⚠️ That doesn't look like a valid OpenAI API key."

    try:
        client = openai.OpenAI(api_key=key)
        client.models.list()
    except openai.AuthenticationError:
        return False, "⚠️ Invalid API key. Please check the key and try again."
    except openai.APIConnectionError:
        return False, "⚠️ Could not reach OpenAI. Check your internet connection."
    except Exception:  # noqa: BLE001 - never surface raw exception details
        return False, "⚠️ Could not verify the API key right now. Please try again."

    return True, "✓ AI Connected"


# ---------------------------------------------------------------------------
# MESSAGE TYPE DEMONSTRATION
# A small, self-contained example of the three core LangChain message types.
# Not used in the main analysis flow -- purely for teaching / the viva.
# ---------------------------------------------------------------------------

def demonstrate_messages() -> list:
    """
    Build a tiny example conversation using LangChain's message classes.

    - SystemMessage: sets the assistant's role/behavior (the AI never
      "speaks" this out loud, it just follows the instructions).
    - HumanMessage: represents what the user said.
    - AIMessage: represents what the assistant previously said (useful for
      giving the model conversation history).

    Returns a list of message objects that could be passed directly to
    llm.invoke(messages).
    """
    messages = [
        # 1) The system message sets the assistant's persona and rules.
        SystemMessage(content=SAFETY_SYSTEM_INSTRUCTIONS),
        # 2) A human message: something the user asked.
        HumanMessage(content="What does a savings ratio mean, in simple terms?"),
        # 3) An AI message: a previous reply, shown here as an example of
        #    how conversation history is represented (not actually
        #    generated by calling the model).
        AIMessage(
            content=(
                "Your savings ratio is the percentage of your income that "
                "you keep as savings each month. For example, saving $200 "
                "out of $2,000 income is a 10% savings ratio."
            )
        ),
        # A follow-up human message, showing how a multi-turn conversation
        # is simply a growing list of these message objects.
        HumanMessage(content="Is that a healthy percentage?"),
    ]
    return messages


# ---------------------------------------------------------------------------
# STRUCTURED JSON ANALYSIS CHAIN
# ---------------------------------------------------------------------------

def build_analysis_chain(api_key: str, model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE):
    """
    Build the reusable financial-analysis chain using LangChain Expression
    Language (LCEL): `prompt | llm`.

    LCEL's `|` (pipe) operator is the modern, actively-maintained way to
    compose a prompt template with a model -- it replaces the older
    `LLMChain` class for most new projects. See
    `demonstrate_legacy_llmchain()` below for the classic API.
    """
    llm = get_chat_model(api_key=api_key, model_name=model_name, streaming=False, temperature=temperature)
    chain = ANALYSIS_CHAT_PROMPT | llm
    return chain


def demonstrate_legacy_llmchain(api_key: str, model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Educational demonstration of the classic `LLMChain` class that the
    assignment asks about.

    In recent LangChain releases `LLMChain` has been deprecated in favor of
    LCEL (`prompt | llm`), and in some installed versions the import path
    below may not exist at all. We try the legacy import, and if it's
    unavailable we fall back to an LCEL-equivalent while explaining why --
    the assignment concept (a reusable prompt+model chain) is preserved
    either way.
    """
    try:
        from langchain.chains import LLMChain  # legacy API
        from src.prompts import FINANCIAL_ANALYSIS_PROMPT

        llm = get_chat_model(api_key=api_key, model_name=model_name, streaming=False, temperature=temperature)
        legacy_chain = LLMChain(llm=llm, prompt=FINANCIAL_ANALYSIS_PROMPT)
        return (
            "LLMChain imported and constructed successfully from "
            "langchain.chains. Call legacy_chain.run(**inputs) to use it."
        )
    except ImportError:
        return (
            "LLMChain is not available in this installed LangChain version "
            "(it has been deprecated). This project uses the modern LCEL "
            "equivalent instead: `prompt | llm`, built in "
            "build_analysis_chain(). The underlying concept -- a reusable "
            "prompt bound to a model -- is the same."
        )


def run_financial_analysis(
    prompt_values: Dict[str, Any],
    api_key: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """
    High-level function used by app.py:
    1. Build the chain.
    2. Send the financial data through the prompt to the model.
    3. Parse + validate the JSON response.

    Returns:
        (parsed_json_or_None, error_message_or_None, raw_response_text)

    Errors are converted to safe, generic messages -- the raw exception
    (which could echo request details) is never returned to the UI.
    """
    try:
        chain = build_analysis_chain(api_key=api_key, model_name=model_name)
        response = chain.invoke(prompt_values)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except MissingAPIKeyError as exc:
        return None, str(exc), ""
    except openai.AuthenticationError:
        return None, "Invalid API key. Please reconnect with a valid OpenAI API key.", ""
    except openai.APIConnectionError:
        return None, "Could not reach the AI service. Check your internet connection.", ""
    except Exception:  # noqa: BLE001 - never surface raw exception details
        return None, "The AI service could not complete this request. Please try again.", ""

    parsed, parse_error = safe_parse_llm_json(raw_text)
    if parsed is None:
        return None, parse_error, raw_text

    return parsed, None, raw_text


# ---------------------------------------------------------------------------
# STREAMING RECOMMENDATIONS
# ---------------------------------------------------------------------------

def stream_recommendations(
    prompt_values: Dict[str, Any],
    api_key: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Iterator[str]:
    """
    Generator that yields text chunks for the "AI Recommendations" section,
    meant to be passed directly to st.write_stream().

    Steps:
    1. Format the ChatPromptTemplate with the user's financial data.
    2. Call the model using `.stream()` instead of `.invoke()`.
    3. Yield each chunk's text content as it arrives.

    If anything goes wrong (missing key, network error), we yield a single
    friendly, safe error message instead of raising or echoing raw
    exception text -- so the Streamlit app never crashes mid-stream and
    never leaks sensitive request details.
    """
    try:
        llm = get_chat_model(api_key=api_key, model_name=model_name, streaming=True)
        formatted_messages = STREAMING_RECOMMENDATIONS_PROMPT.format_messages(**prompt_values)
    except MissingAPIKeyError as exc:
        yield f"\u26a0\ufe0f {exc}"
        return
    except Exception:  # noqa: BLE001
        yield "\u26a0\ufe0f Could not prepare the AI request. Please try again."
        return

    try:
        for chunk in llm.stream(formatted_messages):
            content = getattr(chunk, "content", "")
            if content:
                yield content
    except openai.AuthenticationError:
        yield "\n\n\u26a0\ufe0f Invalid API key. Please reconnect with a valid OpenAI API key."
    except Exception:  # noqa: BLE001
        yield "\n\n\u26a0\ufe0f The AI response was interrupted. Please try again."
