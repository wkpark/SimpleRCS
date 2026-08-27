# AGENTS.md

Guidance for AI coding agents working on this repository.

## What this is

SimpleRCS is a small, dependency-free, reverse-delta version control library for
single files (text or binary), with an RCS-inspired plain-text storage format, a
hash chain, and optional GPG signatures. Python >= 3.13, environment managed by
`uv`. Single maintainer (wkpark).

## Commands

```bash
uv run pytest tests/unit_tests/ -q     # full test suite (~30s) — must stay green
uv run ruff check                      # lint — must stay clean (line-length 120)
uv run tools/bench_diff.py             # compare all diff engines (time + memory)
uv run tools/srcs_commit.py FILE -m "msg" --no-sign   # CLI smoke test
```

The Cython extensions (`simple_rcs/_myersdiff_{ses,dmp}.pyx`) are the **default
diff backend** and are built by `uv sync` when a C toolchain is present.
`setup.py` downgrades a failed compile to a warning, so they must stay
optional: `simple_rcs/matchers.py` resolves a backend at import time and falls
back. Rebuild after editing a `.pyx` file. `SIMPLE_RCS_NO_EXT=1` skips the
build (used to produce a pure-Python wheel).

## Layout

| Path | Role |
|---|---|
| `simple_rcs/simple_rcs.py` | Core `SimpleRCS` class: commit/checkout/log/blame/diff/verify/sign_head, block parsing, backward file scanning |
| `simple_rcs/codec.py` | Stateless format primitives: binary payload encode/decode, `@`-escaping, block hash |
| `simple_rcs/pydifflib.py` | `StreamSequenceMatcher` — the production text-diff engine (hash-based greedy + difflib refinement) |
| `simple_rcs/pybsdiff.py` | BSDIFF-style binary deltas |
| `simple_rcs/myersdiff*.py` | Pure-Python Myers variants — first-cut reference implementations, **never a fallback target** (see Gotchas) |
| `simple_rcs/matchers.py` | Backend registry: resolves `SIMPLE_RCS_MATCHER` -> Cython Myers -> `StreamSequenceMatcher` at import, and adapts line-oriented matchers to the stream interface |
| `simple_rcs/_myersdiff_{ses,dmp}.pyx` | Cython Myers matchers — the default backend when compiled; optional by design |
| `simple_rcs/adapters.py` | File-like adapter for psycopg2 large objects |
| `simple_rcs/simple_rcs_gpg.py` | GPG sign/verify callbacks used by the CLIs |
| `tools/` | CLI scripts (`srcs_commit/log/diff/blame/verify/sign_head`, benchmarks) — run from repo root via `uv run tools/...` |
| `scripts/` | One-off exploratory/design benchmark scripts (e.g. wiki-backend storage experiments) not tied to a specific CLI command — run via `uv run scripts/...` |
| `tests/unit_tests/` | pytest suite |
| `docs/` | Design/benchmark notes (e.g. `parser_benchmark.md`) |

Full storage-format details (block layout, header, v1 vs v2, reverse-delta
mechanics) live in the project wiki's [Storage Format](https://github.com/wkpark/SimpleRCS/wiki/Storage-Format)
page, not here — this file stays focused on what changes how you write code.

## Conventions

- Commit messages: English only, conventional commits, concise and technical
  (senior-engineer audience). No prose that restates the diff.
- Library code logs via `logging`; it never `print`s. `tools/` scripts are
  exempt and start with `# ruff: noqa: T201, ANN201`.
- **Deliberately-kept reference implementations exist.** Example:
  `_parse_block_content` (the regex parser) is unused on the hot path *on
  purpose* — it is the readable specification of the block format, pinned by
  `test_parse_block_content_matches_no_regex`. Do not flag or remove
  "dead-looking" private methods without checking intent with the maintainer.
- The repo root contains untracked scratch files (`aa.py`, `bb.py`, `trash/`,
  test PDFs, `.srcs/`...). Leave them alone; never `git add -A` — stage files
  explicitly.
- **YAGNI by default, especially for patches/PRs.** Don't add abstractions,
  config knobs, or generalized interfaces for needs that don't exist yet.
  A contribution should solve the problem in front of it, not a hypothetical
  future one.
  - Exception: if following YAGNI would leave clearly duplicated logic (the
    same fix copy-pasted across call sites, the same parsing/validation
    rewritten twice, etc.), prefer the small optimization/refactor that
    removes the duplication over keeping it split just to avoid an
    abstraction. Judge case by case — one or two similar lines is not
    duplication worth abstracting; the same non-trivial logic repeated is.

## Gotchas

- `commit(str)` takes the text path (line deltas); `commit(bytes)` takes the
  binary path (BSDIFF). The library does **not** auto-detect — callers decide.
  `tools/srcs_commit.py` applies a NUL-byte + UTF-8-decode heuristic (like git).
- Block hashes cover the **logical full content**, not the stored delta, so a
  hash stays valid when HEAD is later demoted to a delta block. Keep this in
  mind if you touch hashing code — hashing the stored bytes instead breaks
  the chain across a HEAD demotion.
- Corrupted delta payloads raise `SimpleRCSCorruptionError` (a `ValueError`
  subclass); `verify()` returns `False` instead of raising. Don't assume
  these two error paths are interchangeable.
- `commit()` reads head info at its **start**; after committing,
  `self.head_info` is stale — call `_load_head(force=True)` before relying on it.
- `get_opcodes()` on every matcher is an **uncached generator** — consuming it
  twice runs the whole diff twice.
- The regex parser degrades sharply on large blocks (10–30x slower than
  `_parse_block_content_no_regex`); never wire it into a hot path.
- Pure-Python Myers matchers hit their O(ND) worst case on dissimilar inputs
  (seconds on ~500KB with 10% changes); the Cython builds do not. Check
  `docs/parser_benchmark.md` and `tools/bench_diff.py` before performance work.
  This is why the fallback chain in `matchers.py` ends at
  `StreamSequenceMatcher` and never at a `myersdiff_*.py` twin, despite the
  twins being drop-in compatible. They are selectable by name only.
- Backends emit different but equally valid opcode sequences, so the **stored
  delta bytes differ between backends**. Block hashes cover logical content, so
  histories stay verifiable across a backend switch — don't "fix" a delta-byte
  mismatch between two backends, assert on content instead.
