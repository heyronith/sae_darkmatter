# P1.4 notes

- Template heuristics ported from [https://github.com/EleutherAI/semantic-memorization](https://github.com/EleutherAI/semantic-memorization)
  commit `84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2` (`filters/pattern.py`, `filters/huffman_coding_filter.py`).
- Flag = incrementing OR repeating OR (Huffman average code length ≤ 2.5).
- Huffman length is bits/character (lower ⇒ more compressible). zlib ratio is reported
  as a secondary diagnostic only and is **not** used in the flag.
