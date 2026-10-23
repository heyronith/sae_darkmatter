"""Random-access reader for EleutherAI/pile-deduped-pythia-preshuffled shards."""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

PILE_MMAP_DS = "EleutherAI/pile-deduped-pythia-preshuffled"
PILE_MMAP_REVISION = "4647773ea142ab1ff5694602fa104bbf49088408"
N_SHARDS = 21  # document-00000-of-00020.bin … document-00020-of-00020.bin
SEQ_TOKENS = 2049  # packed training windows

_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def shard_name(i: int) -> str:
    return f"document-{i:05d}-of-00020.bin"


def load_index(cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Load ``document.idx`` sizes/pointers (memmap; ~1.7 GB on disk)."""
    path = hf_hub_download(
        PILE_MMAP_DS,
        "document.idx",
        repo_type="dataset",
        revision=PILE_MMAP_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    with open(path, "rb") as stream:
        magic = stream.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(f"unexpected idx magic {magic!r}")
        version = struct.unpack("<Q", stream.read(8))
        if version != (1,):
            raise ValueError(f"unexpected idx version {version}")
        (dtype_code,) = struct.unpack("<B", stream.read(1))
        dtype = _DTYPES[dtype_code]
        n = struct.unpack("<Q", stream.read(8))[0]
        doc_count = struct.unpack("<Q", stream.read(8))[0]
        offset = stream.tell()
    mm = np.memmap(path, mode="r", order="C")
    sizes = np.frombuffer(mm, dtype=np.int32, count=n, offset=offset)
    pointers = np.frombuffer(
        mm, dtype=np.int64, count=n, offset=offset + sizes.nbytes
    )
    return {
        "path": path,
        "dtype": dtype,
        "dtype_code": dtype_code,
        "n": int(n),
        "doc_count": int(doc_count),
        "sizes": sizes,
        "pointers": pointers,
        "element_size": int(np.dtype(dtype).itemsize),
        "dataset_id": PILE_MMAP_DS,
        "revision": PILE_MMAP_REVISION,
    }


@lru_cache(maxsize=1)
def byte_ranges_from_hub() -> tuple[tuple[int, int, str], ...]:
    api = HfApi()
    info = api.repo_info(
        PILE_MMAP_DS,
        repo_type="dataset",
        revision=PILE_MMAP_REVISION,
        files_metadata=True,
    )
    siblings = {s.rfilename: s for s in (info.siblings or [])}
    ranges: list[tuple[int, int, str]] = []
    cursor = 0
    for i in range(N_SHARDS):
        name = shard_name(i)
        sib = siblings[name]
        size = int(sib.size)
        ranges.append((cursor, cursor + size, name))
        cursor += size
    return tuple(ranges)


def ensure_shard(i: int, cache_dir: str | Path | None = None) -> str:
    return hf_hub_download(
        PILE_MMAP_DS,
        shard_name(i),
        repo_type="dataset",
        revision=PILE_MMAP_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def pointer_to_shard(
    pointer: int,
    ranges: tuple[tuple[int, int, str], ...] | list[tuple[int, int, str]],
) -> tuple[int, int]:
    """Map a merged-bin byte pointer to ``(shard_index, offset_in_shard)``."""
    for i, (start, end, _name) in enumerate(ranges):
        if start <= pointer < end:
            return i, pointer - start
    raise IndexError(f"pointer {pointer} outside shard ranges")


def indices_in_shard(index: dict[str, Any], shard_i: int) -> np.ndarray:
    ranges = byte_ranges_from_hub()
    start, end, _ = ranges[shard_i]
    ptrs = index["pointers"]
    return np.flatnonzero((ptrs >= start) & (ptrs < end))


class ShardWindowReader:
    """Read the first ``n_tokens`` of training windows from one downloaded shard."""

    def __init__(
        self,
        index: dict[str, Any],
        shard_i: int,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.index = index
        self.shard_i = int(shard_i)
        self.ranges = byte_ranges_from_hub()
        self.shard_path = ensure_shard(shard_i, cache_dir=cache_dir)
        self.shard_start, self.shard_end, _ = self.ranges[shard_i]
        self._file = open(self.shard_path, "rb")
        self.local_indices = indices_in_shard(index, shard_i)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> ShardWindowReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def read_prefix(self, seq_index: int, n_tokens: int = 64) -> list[int]:
        ptr = int(self.index["pointers"][seq_index])
        size = int(self.index["sizes"][seq_index])
        if n_tokens > size:
            raise ValueError(f"seq {seq_index} has only {size} tokens")
        shard_i, local = pointer_to_shard(ptr, self.ranges)
        if shard_i != self.shard_i:
            raise ValueError(f"seq {seq_index} not in shard {self.shard_i}")
        elem = self.index["element_size"]
        self._file.seek(local)
        buf = self._file.read(n_tokens * elem)
        arr = np.frombuffer(buf, dtype=self.index["dtype"], count=n_tokens)
        return [int(x) for x in arr.tolist()]
