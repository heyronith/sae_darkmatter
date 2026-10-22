# SAE Dark Matter

Research codebase for studying SAE (sparse autoencoder) "dark matter" — features and activations that may reflect memorization rather than generalizable structure. Every experiment is keyed by a task ID and must be reproducible from this repo alone via pinned configs, seeds, and `manifest.json` provenance.

## Setup

```bash
uv sync
```

## Tests

```bash
uv run pytest
```

## Smoke run

```bash
uv run python scripts/smoke_pythia.py
```

## Conventions

- Task IDs follow `P<phase>.<n>` (e.g. `P0.1`).
- Outputs live in `results/<task-id>/` and always include `manifest.json`.
- Large artifacts (weights, caches, data) never go in git; `data/` is gitignored.
- **Commit code first, then produce results from a clean tree, then commit results in a second commit with the same task ID.** `write_manifest` refuses dirty code (ignoring `results/`) unless `allow_dirty=True`.
