"""Model factory for the 3 LLMs used in the thesis grid.

Every model uses the same temperature / top_p so any difference in summary
quality reflects the model, not the sampling config. `seed` varies per run
(1, 2, 3) so stochastic runs are reproducibly different.
"""
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


SUPPORTED_MODELS = ("gemini", "grok", "o4mini")

TEMPERATURE = 0.7
TOP_P = 0.95
NUM_CTX = 40960


def get_model(name: str, seed: int) -> Any:
    # ── OpenRouter-backed models. Reasoning models ignore temp/top_p/seed,
    #    so we omit those params and let the model's internal sampling
    #    drive variance across runs 1/2/3.
    if name == "gemini":
        import truststore
        truststore.inject_into_ssl()
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="google/gemini-2.5-flash",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        )
    if name == "grok":
        import truststore
        truststore.inject_into_ssl()
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="x-ai/grok-4.3",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        )
    if name == "o4mini":
        import truststore
        truststore.inject_into_ssl()
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="openai/o4-mini",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Unknown model {name!r}; expected one of {SUPPORTED_MODELS}")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def invoke(chain, payload: dict) -> str:
    """Run a LangChain chain and return clean text.
    OllamaLLM returns a str; ChatOpenAI returns an AIMessage with .content.
    """
    result = chain.invoke(payload)
    text = result.content if hasattr(result, "content") else result
    return _THINK_RE.sub("", text).strip()
