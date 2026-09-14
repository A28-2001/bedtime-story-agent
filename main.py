import os
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


def main():
    # temporary test; the real pipeline replaces this in the next phase
    print(call_model("Tell a two-sentence story about a cat.", temperature=0.9))


if __name__ == "__main__":
    main()