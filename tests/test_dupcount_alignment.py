"""Unit tests for EOS boundary detection and Llama span alignment."""

from __future__ import annotations

from saedm.data.dupcount import (
    GPT_NEOX_EOS_ID,
    continuation_char_span,
    llama_query_ids_for_continuation,
    longest_eos_free_continuation,
    neox_token_char_spans,
)


class _FakeNeoX:
    """Deterministic decode: each id i maps to the character ``chr(65 + i % 26)``."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (i % 26)) for i in ids)


class _FakeLlama:
    """Character-level fake Llama tokenizer with offset mappings."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        ids = list(range(len(text)))
        offsets = [(i, i + 1) for i in range(len(text))]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_longest_eos_free_no_boundary() -> None:
    toks = list(range(10, 74))  # 64 tokens, no EOS
    cont, crosses, usable = longest_eos_free_continuation(toks)
    assert crosses is False
    assert usable == 32
    assert cont == toks[32:64]


def test_longest_eos_free_with_eos_in_continuation() -> None:
    prompt = list(range(1, 33))
    # continuation: 5 toks, EOS, 10 toks, EOS, 3 toks, EOS-padded -> longest is 10
    cont = (
        [100, 101, 102, 103, 104, GPT_NEOX_EOS_ID]
        + list(range(200, 210))
        + [GPT_NEOX_EOS_ID, 300, 301, 302]
        + [GPT_NEOX_EOS_ID] * 12
    )
    assert len(cont) == 32
    toks = prompt + cont
    got, crosses, usable = longest_eos_free_continuation(toks)
    assert crosses is True
    assert usable == 10
    assert got == list(range(200, 210))


def test_neox_char_spans_cover_full_text() -> None:
    tok = _FakeNeoX()
    ids = [0, 1, 2, 3, 4]
    spans = neox_token_char_spans(tok, ids)
    text = tok.decode(ids)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    # contiguous
    for (_a, b), (c, _d) in zip(spans, spans[1:], strict=False):
        assert b == c


def test_continuation_char_span_starts_at_token_32() -> None:
    tok = _FakeNeoX()
    ids = list(range(64))
    text, start, end = continuation_char_span(tok, ids)
    assert text == tok.decode(ids)
    assert start == 32  # one char per token in fake tokenizer
    assert end == 64


def test_llama_query_keeps_only_continuation_interior() -> None:
    neox = _FakeNeoX()
    llama = _FakeLlama()
    ids = list(range(64))
    out = llama_query_ids_for_continuation(neox, llama, ids, edge_drop=2, min_len=16)
    # continuation chars [32,64); drop 2 each end -> chars 34..61 inclusive -> 28 tokens
    assert out["too_short"] is False
    assert out["query_len_tokens"] == 28
    assert out["query_ids"][0] == 34
    assert out["query_ids"][-1] == 61
    assert out["crosses_doc_boundary"] is False


def test_llama_query_marks_too_short() -> None:
    neox = _FakeNeoX()
    llama = _FakeLlama()
    # Only 5 usable continuation tokens after EOS trimming
    prompt = list(range(1, 33))
    cont = [10, 11, 12, 13, 14] + [GPT_NEOX_EOS_ID] * 27
    out = llama_query_ids_for_continuation(
        neox, llama, prompt + cont, edge_drop=2, min_len=16
    )
    assert out["crosses_doc_boundary"] is True
    assert out["too_short"] is True
    assert out["query_len_tokens"] < 16
