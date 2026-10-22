"""Unit tests for template heuristics (hand-made examples)."""

from __future__ import annotations

from saedm.data.templates import classify_template, is_pattern


def test_incrementing_hex_list_flagged() -> None:
    text = "23: 0xf1, 24: 0xf2, 25: 0xf3, 26: 0xf4"
    is_inc, is_rep = is_pattern(text)
    assert is_inc is True
    assert is_rep is False
    assert classify_template(text)["is_template"] is True


def test_repeated_phrase_flagged() -> None:
    text = "Go Go Go Go Go Go Go Go Go Go Go Go"
    _is_inc, is_rep = is_pattern(text)
    # Character-level repeating / short-cycle detection.
    flags = classify_template(text)
    assert is_rep or flags["compressible"]
    assert flags["is_template"] is True


def test_random_word_sequence_not_flagged() -> None:
    text = (
        "orchid balcony quartz meadow lantern silver copper harvest "
        "velvet canyon marble whisper galaxy timber"
    )
    flags = classify_template(text)
    assert flags["is_incrementing"] is False
    assert flags["is_repeating"] is False
    assert flags["is_template"] is False
