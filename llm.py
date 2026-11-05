"""One place that talks to the LLM provider (any OpenAI-compatible API)."""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

DEFAULT_MODEL = "openrouter/free"


def _short(exc):
    return str(exc).replace("\n", " ")[:300]


def chat(messages, task="general", temperature=0.1):
    """Return the model's reply, or None if every model/retry failed.
    The real error is printed in the terminal so you can see what went wrong."""
    model_list = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
    models = [m.strip() for m in model_list.split(",") if m.strip()]
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("[llm] No LLM_API_KEY found. Is there a file named exactly .env in the project folder?")
        return None
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=key,
        max_retries=0,
        timeout=60,
    )
    for model in models:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                print(f"[llm] {task}: {model} rate limited (try {attempt + 1}/3): {_short(e)}")
                time.sleep(5 * (attempt + 1))
            except Exception as e:  # wrong key, wrong model name, network, ...
                print(f"[llm] {task}: {model} failed: {type(e).__name__}: {_short(e)}")
                break  # try the next model in the list
    return None