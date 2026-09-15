"""Generate sample_output.md: one full pipeline run with all drafts and
judge reports, as reviewable evidence of the judge-revision loop."""
import json

from main import create_story

REQUEST = "A story about a scary dragon that burns down a village"
MODE = "sleepy"


def main():
    result = create_story(REQUEST, mode=MODE)
    lines = [
        "# Sample run: full pipeline transcript",
        "",
        f"**Request:** {REQUEST}",
        f"**Mode:** {MODE}",
        "",
        "## Story brief (interpreter output)",
        "```json",
        json.dumps(result["brief"], indent=2),
        "```",
        "",
    ]
    for i, entry in enumerate(result["history"], 1):
        j = entry["judgment"]
        lines += [
            f"## Round {i} draft",
            "",
            entry["draft"],
            "",
            f"### Round {i} judge report",
            "```json",
            json.dumps(j, indent=2),
            "```",
            "",
        ]
    lines += [
        "## Final story (best draft shipped)",
        "",
        result["story"],
        "",
        f"*Revisions used: {result['revisions_used']}*",
    ]
    with open("sample_output.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote sample_output.md ({len(result['history'])} judged rounds)")


if __name__ == "__main__":
    main()