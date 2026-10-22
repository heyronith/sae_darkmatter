"""Template / reconstruction heuristics from Prashanth et al. (ICLR 2025).

Source: EleutherAI/semantic-memorization @ 84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2
  - filters/pattern.py (incrementing / repeating)
  - filters/huffman_coding_filter.py (Huffman average code length)

We port the pure-Python helpers (no Spark). Compressibility flag uses the
Huffman average code length with a documented threshold (lower = more
compressible).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from typing import Any

import numpy as np

# Cite this commit when reporting template filters.
SEMANTIC_MEM_COMMIT = "84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2"
SEMANTIC_MEM_REPO = "https://github.com/EleutherAI/semantic-memorization"

# Average Huffman code length (bits/char). Values at or below this are flagged
# as highly compressible templates. Chosen to catch near-uniform short alphabets
# (repeats / counters) while leaving diverse English above the line; see notes.
HUFFMAN_TEMPLATE_THRESHOLD = 2.5


def find_if_incrementing_or_repeating(
    splits: list,
    test_repeating: bool = False,
) -> tuple[bool, float | None]:
    """Port of semantic-memorization ``find_if_incrementing_or_repeating``."""
    if len(splits) < 3 and not test_repeating:
        return False, 0
    if len(splits) == 3 and not test_repeating:
        if not all(type(i) in [float, int] for i in splits):
            return False, 0
        return (splits[2] - 2 * splits[1] + splits[0]) < 1e-5, splits[2] - splits[1]

    if len(splits) > 4 and not test_repeating:
        splits = splits[1:-1]

    diff = None
    for temp_len in range(1, len(splits) // 2 + 1):
        is_inc = True
        diff_valid = False
        for every_i in range(temp_len):
            curr_diff = None
            for i in range(every_i + temp_len, len(splits), temp_len):
                if type(splits[i]) is not type(splits[i - temp_len]):
                    is_inc = False
                    break

                if curr_diff is None and type(splits[i]) in [int, float]:
                    curr_diff = splits[i] - splits[i - temp_len]
                elif type(splits[i]) in [int, float]:
                    is_curr_inc = abs(splits[i] - splits[i - temp_len] - curr_diff) < 1e-10
                    if not is_curr_inc:
                        is_inc = False
                        break
                    diff_valid = True
                elif type(splits[i]) is str and splits[i] != splits[i - temp_len]:
                    is_inc = False
                    break

            if not is_inc:
                break

            if diff is None and curr_diff is not None and curr_diff != 0:
                diff = curr_diff
        if is_inc:
            if diff is not None and diff_valid:
                return True, diff
            if diff is None:
                return True, None

    return False, 0


def split_text(text: str, split_type: str = "incrementing") -> list:
    """Port of semantic-memorization ``split_text``."""
    if split_type == "repeating":
        return list(text)
    if split_type != "incrementing":
        raise ValueError("Invalid Split Type")

    text = re.sub(r"\s+", " ", text)
    splits: list[str] = []
    text_recon = ""
    for word in text.split(" "):
        try:
            text_recon += str(int(word, 0)) + " "
        except ValueError:
            text_recon += word + " "

    text_recon = repr(text_recon.strip())
    for word in text_recon.split(" "):
        word = word.replace("\\n", "")
        word = word.replace("\\x", "")
        word = word.replace("\\", "")
        word = word.replace("'", "")
        word = word.replace('"', "")
        word_parts = re.split("([0-9]+)", word)
        splits.extend(word_parts)

    splits_new: list = []
    to_continue = False
    for word in splits:
        word = word.strip("'")
        if to_continue:
            to_continue = not to_continue
            continue
        if word.strip(" ") == "":
            continue
        if word == "":
            continue
        try:
            splits_new.append(int(word))
        except ValueError:
            splits_new.append(word)
    return splits_new


def is_pattern(text: str) -> tuple[bool, bool]:
    """Return ``(is_incrementing, is_repeating)`` — port of ``is_pattern``."""
    splits = split_text(text)
    is_inc, diff = find_if_incrementing_or_repeating(splits)
    if is_inc and diff is not None and diff != 0:
        return True, False

    splits = split_text(text, split_type="repeating")
    is_inc, _diff = find_if_incrementing_or_repeating(splits)
    if is_inc:
        return False, True
    return False, False


def huffman_code_length(text: str) -> float:
    """Average Huffman code length (bits/char); port of their filter."""
    if not text:
        return 0.0
    char_counts = np.array(list(Counter(text).values()), dtype=np.float64)
    char_probs = char_counts / len(text)
    char_code_lengths = np.ceil(-np.log2(char_probs))
    return float(np.sum(char_code_lengths * char_probs))


def zlib_ratio(text: str) -> float:
    """Compressed size / raw UTF-8 size (secondary compressibility metric)."""
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, level=9)) / len(raw)


def classify_template(
    text: str,
    *,
    huffman_threshold: float = HUFFMAN_TEMPLATE_THRESHOLD,
) -> dict[str, Any]:
    """Flag template if incrementing OR repeating OR low Huffman length."""
    is_inc, is_rep = is_pattern(text)
    huff = huffman_code_length(text)
    zratio = zlib_ratio(text)
    compressible = huff <= huffman_threshold
    is_template = bool(is_inc or is_rep or compressible)
    return {
        "is_incrementing": bool(is_inc),
        "is_repeating": bool(is_rep),
        "huffman_code_length": huff,
        "zlib_ratio": zratio,
        "compressible": compressible,
        "huffman_threshold": huffman_threshold,
        "is_template": is_template,
    }
