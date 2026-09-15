import json
import re
import time

from openai import OpenAI

"""
Before submitting the assignment, describe here in a few sentences what you
would have built next if you spent 2 more hours on this project:

1. Story memory: persist each story's characters to a local JSON file so a
   child can ask for "another story about Fluffy" and sequels inherit
   established characters. Touches only the brief-construction step.
2. Mode-aware judging: condition the bedtime_suitability rubric on the
   sleepy/storytime mode, after collecting per-mode score baselines so the
   rubric change is driven by data.
3. Deterministic text cleanup: a non-LLM post-processing pass for the
   missing-space artifacts gpt-3.5-turbo produces, cheaper and more
   reliable than asking the model to proofread itself.
4. A small evaluation harness: run a fixed suite of requests N times and
   report score distributions per pipeline stage, turning "the judge
   improves stories" from an anecdote into a measured claim.
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

STORY_MODES = {
    "sleepy": "MODE: Wind-down story. Make the final third of the story "
              "progressively quieter and slower. Sentences in the last two "
              "paragraphs should be short and soft, like a lullaby in prose. "
              "End with the characters falling asleep.",
    "storytime": "MODE: Story-time story. Keep the gentle, cozy ending, but "
                 "the energy can stay a little brighter - end with a warm "
                 "smile rather than falling asleep.",
}


def build_storyteller_prompt(brief: dict, mode: str = "sleepy") -> str:
    """Assemble the generation prompt from the brief + arc + story mode."""
    arc = ARC_TEMPLATES.get(brief.get("category", ""),
                            ARC_TEMPLATES["everyday-life"])
    return (
        f"Write a bedtime story from this brief.\n\n"
        f"Characters: {', '.join(brief['characters'])}\n"
        f"Setting: {brief['setting']}\n"
        f"Theme: {brief['theme']}\n"
        f"Story arc to follow: {arc}\n"
        f"{STORY_MODES.get(mode, STORY_MODES['sleepy'])}\n"
    )


def generate_story(brief: dict, mode: str = "sleepy") -> str:
    """One creative call: brief in, first-draft story out."""
    return call_model(build_storyteller_prompt(brief, mode),
                      system=STORYTELLER_SYSTEM,
                      temperature=0.9,
                      max_tokens=900)

# ---------------------------------------------------------------- judge

JUDGE_SYSTEM = """You are a strict quality judge for children's bedtime \
stories (ages 5-10). You will receive a story brief and a story. Evaluate \
the story on five dimensions, each scored 1-10:

1. age_appropriate_language: simple, concrete words a 7-year-old understands;
   short sentences; no abstract or advanced vocabulary
2. story_arc: clear beginning, middle, and end; follows the arc described in
   the brief; no plot holes or abrupt jumps
3. engagement: charming, warm, a little humor; characters feel alive; not a
   flat list of events; minimal cliches and repeated phrases
4. bedtime_suitability: energy winds DOWN toward the ending; final image is
   soothing and safe; nothing exciting or unresolved at the end
5. safety: nothing violent, scary, sad, or inappropriate for ages 5-10
   anywhere in the story

Also report mechanical problems (missing spaces, typos, broken sentences) -
these belong under "revision_notes" as fixes, and cap age_appropriate_language
at 7 if present.

Respond as strict JSON only, with exactly these keys:
{
  "scores": {"age_appropriate_language": int, "story_arc": int,
             "engagement": int, "bedtime_suitability": int, "safety": int},
  "verdict": "approve" or "revise",
  "revision_notes": list of specific, actionable instructions (what to change
                    and where); empty list if verdict is "approve"
}

Be demanding. A first draft should rarely score above 8 on engagement.
Verdict is "approve" only if EVERY score is 8 or higher."""


APPROVAL_THRESHOLD = 8  # every rubric dimension must reach this


def judge_story(brief: dict, story: str) -> dict:
    """Score the story against the brief; return scores + revision notes."""
    prompt = (
        f"STORY BRIEF:\n{json.dumps(brief, indent=2)}\n\n"
        f"STORY:\n{story}\n\n"
        f"Evaluate now."
    )
    result = call_model(prompt, system=JUDGE_SYSTEM, temperature=0.1,
                        max_tokens=600)
    return parse_json_safely(result)


def needs_revision(judgment: dict) -> bool:
    """True if the judge's verdict or any sub-threshold score demands a rewrite.

    Checks the numeric scores directly instead of trusting the verdict string
    alone: 3.5-turbo sometimes says "approve" while scoring a dimension 6.
    """
    scores = judgment.get("scores", {})
    if judgment.get("verdict") == "revise":
        return True
    return any(v < APPROVAL_THRESHOLD for v in scores.values())

# ---------------------------------------------------------------- revision loop

MAX_REVISIONS = 2  # hard cap: cost control + diminishing returns after 2 passes

REVISER_SYSTEM = STORYTELLER_SYSTEM + """

You will receive a story you already wrote, plus specific revision notes from
an editor. Rewrite the story, applying every note while keeping everything
that already works: same characters, same plot skeleton, same cozy ending.
Also silently fix any typos, missing spaces, or broken sentences you notice.
Return only the revised story text.
Keep the story between 400 and 500 words - simplify the language, never shorten the story."""


def revise_story(brief: dict, story: str, revision_notes: list,
                 mode: str = "sleepy") -> str:
    """One rewrite pass: story + notes in, improved story out (mode-aware)."""
    notes = "\n".join(f"- {n}" for n in revision_notes)
    prompt = (
        f"ORIGINAL BRIEF:\n{json.dumps(brief, indent=2)}\n\n"
        f"CURRENT STORY:\n{story}\n\n"
        f"REVISION NOTES:\n{notes}\n\n"
        f"{STORY_MODES.get(mode, STORY_MODES['sleepy'])}\n\n"
        f"Rewrite the story now."
    )
    return call_model(prompt, system=REVISER_SYSTEM, temperature=0.7,
                      max_tokens=900)

def draft_quality(judgment: dict) -> tuple:
    """Sortable quality: worst dimension first, then total. Higher is better."""
    scores = judgment.get("scores", {})
    if not scores:
        return (0, 0)
    return (min(scores.values()), sum(scores.values()))

def create_story(user_request: str, mode: str = "sleepy", verbose: bool = True) -> dict:
    """Full pipeline: request -> brief -> draft -> judge -> revise (<=2x).

    Returns a dict with the final story plus the full history for transparency:
    every draft, every judge report, and how many revisions ran.
    """
    log = lambda msg: print(msg) if verbose else None

    log("\nUnderstanding your request...")
    brief = interpret_request(user_request)
    if brief.get("adjustments"):
        log("(Gently adjusted: " + "; ".join(brief["adjustments"]) + ")")

    log("Writing your story...")
    story = generate_story(brief, mode)
    history = []
    best_story, best_judgment = None, None

    for round_num in range(MAX_REVISIONS + 1):
        log(f"Checking story quality (round {round_num + 1})...")
        judgment = judge_story(brief, story)
        history.append({"draft": story, "judgment": judgment})
        if best_judgment is None or draft_quality(judgment) > draft_quality(best_judgment):
            best_story, best_judgment = story, judgment
        if not needs_revision(judgment):
            break
        if round_num == MAX_REVISIONS:
            break  # cap reached: ship the best draft seen, not the last one
        log("Making it even better...")
        story = revise_story(brief, story,
                             judgment.get("revision_notes", []), mode)

    return {"story": best_story, "brief": brief, "history": history,
            "revisions_used": len(history) - 1, "mode": mode}
    
def apply_user_feedback(brief: dict, story: str, feedback: str,
                        mode: str = "sleepy") -> str:
    """Route the child's change request through the reviser, then re-judge.

    User feedback is treated like judge notes - same reviser, different
    source. The judge still checks the result so a request like 'make it
    scarier' cannot push the story out of the safe zone.
    """
    revised = revise_story(brief, story, [f"The listener asked: {feedback}"],
                           mode)
    judgment = judge_story(brief, revised)
    if needs_revision(judgment):
        revised = revise_story(brief, revised,
                               judgment.get("revision_notes", []), mode)
    return revised


def main():
    user_input = input("What kind of story do you want to hear? ")
    mode_answer = input("Is this for winding down to sleep? (yes/no) ").strip().lower()
    mode = "sleepy" if mode_answer in ("y", "yes", "") else "storytime"

    result = create_story(user_input, mode=mode)
    story = result["story"]
    print("\n" + "=" * 60 + "\n")
    print(story)
    print("\n" + "=" * 60)

    while True:
        feedback = input(
            "\nWould you like to change anything about the story? "
            "(describe it, or press Enter to finish) ").strip()
        if not feedback:
            print("\nSweet dreams!")
            break
        print("\nUpdating your story...\n")
        story = apply_user_feedback(result["brief"], story, feedback,
                                    result["mode"])
        print("=" * 60 + "\n")
        print(story)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()