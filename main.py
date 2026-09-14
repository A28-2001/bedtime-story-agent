import json
import os
import re
import time

from openai import OpenAI

"""
Before submitting the assignment, describe here in a few sentences what you
would have built next if you spent 2 more hours on this project:

(TODO: fill in before submitting)
"""

MODEL = "gpt-3.5-turbo"  # fixed by the assignment rules; do not change

client = OpenAI()  # reads OPENAI_API_KEY from the environment


def call_model(prompt: str, system: str = None, temperature: float = 0.7,
               max_tokens: int = 1200, retries: int = 2) -> str:
    """Single entry point for every LLM call in the pipeline.

    - `system` lets each component (storyteller, judge, ...) define its role
    - `temperature` is per-call: high for creative writing, low for judging
    - basic retry with backoff so one network hiccup doesn't kill a story
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))


def parse_json_safely(text: str) -> dict:
    """Parse JSON from model output, tolerating markdown fences and stray text.

    gpt-3.5-turbo occasionally wraps JSON in ```json fences or adds a sentence
    around it; this strips that before parsing instead of crashing.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ---------------------------------------------------------------- interpreter

INTERPRETER_SYSTEM = """You are the request interpreter for a children's \
bedtime story service (ages 5-10). Turn the user's story request into a \
story brief as strict JSON with exactly these keys:
{
  "category": one of "adventure", "friendship", "animals", "fantasy", "everyday-life",
  "characters": list of short character descriptions from the request (invent one gentle protagonist if none given),
  "setting": one-sentence setting (invent a cozy one if not given),
  "theme": the gentle lesson or feeling the story should carry,
  "adjustments": list of changes you made to keep things calm and age-appropriate (empty list if none)
}

Rules:
- Keep every detail the user asked for unless it is unsuitable for ages 5-10.
- Soften anything scary, violent, or sad into a gentle version (e.g. a \
"terrifying monster" becomes a "big shy creature") and record it in "adjustments".
- Respond with JSON only. No markdown, no commentary."""


def interpret_request(user_request: str) -> dict:
    """Turn a raw request into a structured story brief (with safety softening)."""
    raw = call_model(user_request, system=INTERPRETER_SYSTEM, temperature=0.2,
                     max_tokens=400)
    return parse_json_safely(raw)


def main():
    user_input = input("What kind of story do you want to hear? ")
    brief = interpret_request(user_input)
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()