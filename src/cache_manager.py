"""
cache_manager.py
-----------------
Configures LangChain's global LLM cache.

Two options are supported, matching the assignment requirements:

    InMemoryCache -> stored in RAM, very fast, lost on restart.
    SQLiteCache   -> stored on disk, survives restarts, slightly slower.

Caching means that if the exact same prompt is sent twice, LangChain can
return the previously-generated response instead of calling the OpenAI API
again -- saving time and API cost.
"""

from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

from src.config import SQLITE_CACHE_PATH


def configure_cache(cache_choice: str) -> str:
    """
    Set LangChain's global cache based on the user's sidebar selection.

    Args:
        cache_choice: "In-memory cache" or "SQLite cache"

    Returns:
        A short human-readable description of the active cache, so the UI
        can display confirmation to the user.
    """
    if cache_choice == "SQLite cache":
        set_llm_cache(SQLiteCache(database_path=SQLITE_CACHE_PATH))
        return f"SQLite cache active (file: {SQLITE_CACHE_PATH})"

    # Default to in-memory cache.
    set_llm_cache(InMemoryCache())
    return "In-memory cache active (cleared on app restart)"


def reset_cache(cache_choice: str) -> str:
    """
    Re-create a fresh cache of the same type. Used by the 'Reset Session'
    button so old cached responses don't linger.
    """
    return configure_cache(cache_choice)
