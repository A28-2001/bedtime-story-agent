"""Unit tests for the pipeline's pure functions (no API calls needed).

The LLM stages are covered by scenario testing (see sample_output.md and
the category sweep described in the README); these tests cover the
deterministic logic the pipeline's control flow depends on.
"""
import json

import pytest

from main import parse_json_safely, draft_quality, needs_revision


# ---------------------------------------------------------- parse_json_safely

def test_parses_clean_json():
    assert parse_json_safely('{"a": 1}') == {"a": 1}


def test_strips_markdown_fences():
    raw = '```json\n{"category": "fantasy"}\n```'
    assert parse_json_safely(raw) == {"category": "fantasy"}


def test_extracts_json_from_surrounding_chatter():
    raw = 'Here is the evaluation you asked for:\n{"verdict": "approve"}\nHope this helps!'
    assert parse_json_safely(raw) == {"verdict": "approve"}


def test_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_json_safely("not json at all")


# ------------------------------------------------------------- draft_quality

def test_worst_dimension_ranks_first():
    balanced = {"scores": {"a": 8, "b": 8, "c": 8}}       # min 8, total 24
    spiky = {"scores": {"a": 10, "b": 10, "c": 6}}        # min 6, total 26
    assert draft_quality(balanced) > draft_quality(spiky)


def test_total_breaks_ties():
    low_total = {"scores": {"a": 8, "b": 8}}              # min 8, total 16
    high_total = {"scores": {"a": 8, "b": 10}}            # min 8, total 18
    assert draft_quality(high_total) > draft_quality(low_total)


def test_empty_scores_rank_lowest():
    assert draft_quality({}) == (0, 0)
    assert draft_quality({"scores": {}}) == (0, 0)


# ------------------------------------------------------------ needs_revision

def test_revise_verdict_triggers_revision():
    j = {"verdict": "revise", "scores": {"a": 9, "b": 9}}
    assert needs_revision(j) is True


def test_low_score_overrides_approve_verdict():
    # the judge sometimes says approve while scoring below threshold;
    # the control logic must catch the contradiction
    j = {"verdict": "approve", "scores": {"a": 9, "b": 6}}
    assert needs_revision(j) is True


def test_clean_approval_passes():
    j = {"verdict": "approve", "scores": {"a": 8, "b": 9, "c": 10}}
    assert needs_revision(j) is False