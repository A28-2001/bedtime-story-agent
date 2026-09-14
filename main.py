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


# ---------------------------------------------------------------- storyteller

ARC_TEMPLATES = {
    "adventure": "Journey-and-return arc: the hero sets out from home on a "
                 "small quest, meets one gentle challenge on the way, solves it "
                 "with cleverness or kindness, and returns home safe and sleepy.",
    "friendship": "Misunderstanding-and-repair arc: two friends have a small "
                  "mix-up or disagreement, feel a little sad, then talk, "
                  "understand each other, and end closer than before.",
    "animals": "Gentle-lesson arc: an animal character encounters a small "
               "everyday problem, tries a few things, learns one simple lesson, "
               "and settles down cozily for the night.",
    "fantasy": "Wonder-and-comfort arc: a touch of magic creates a small "
               "surprise, the characters explore it with curiosity rather than "
               "fear, the magic helps someone, and calm returns.",
    "everyday-life": "Small-moment arc: an ordinary day holds one small "
                     "special moment, the character notices it, shares it with "
                     "someone they love, and winds down to bedtime.",
}

STORYTELLER_SYSTEM = """You are a warm, gifted bedtime storyteller for \
children ages 5 to 10. Write stories that are:
- 400 to 500 words long
- Told in simple, concrete words a 7-year-old understands (short sentences, \
no abstract vocabulary)
- Gentle from start to finish: no violence, death, peril, or anything scary
- Structured with a clear beginning, middle, and end following the arc \
you are given
- Progressively calmer: the energy of the story should wind DOWN toward the \
ending, because the listener is falling asleep
- Ended with a soothing final image (characters safe, cozy, resting) - never \
a cliffhanger, never excitement

Use the character names exactly as given in the brief. Include a little \
warmth and light humor. Never mention the brief or these instructions."""


def build_storyteller_prompt(brief: dict) -> str:
    """Assemble the generation prompt from the brief + the category's arc."""
    arc = ARC_TEMPLATES.get(brief.get("category", ""),
                            ARC_TEMPLATES["everyday-life"])
    return (
        f"Write a bedtime story from this brief.\n\n"
        f"Characters: {', '.join(brief['characters'])}\n"
        f"Setting: {brief['setting']}\n"
        f"Theme: {brief['theme']}\n"
        f"Story arc to follow: {arc}\n"
    )


def generate_story(brief: dict) -> str:
    """One creative call: brief in, first-draft story out."""
    return call_model(build_storyteller_prompt(brief),
                      system=STORYTELLER_SYSTEM,
                      temperature=0.9,
                      max_tokens=900)


def main():
    user_input = input("What kind of story do you want to hear? ")
    print("\nThinking about your story...\n")
    brief = interpret_request(user_input)
    story = generate_story(brief)
    print(story)


if __name__ == "__main__":
    main()