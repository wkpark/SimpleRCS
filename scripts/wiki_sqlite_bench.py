#!/usr/bin/env python3
# ruff: noqa: T201
"""Empirical test: SimpleRCS history stored as a single growing BLOB column
in SQLite (design A from the "Wiki Backend Design" project wiki page), vs. a
normalized per-revision table (design B proxy), measured with real sqlite3
UPDATE/INSERT + commit (including fsync) cost as history grows.
"""
import os
import sqlite3
import time

from simple_rcs.simple_rcs import SimpleRCS

# Written next to the repo (not system /tmp, which may be a small/shared
# partition) and removed again at the end of the run — see .gitignore.
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".bench_tmp")
DB_PATH = os.path.join(BENCH_DIR, "wiki_bench.sqlite3")
N = 2000
LINES = 50

os.makedirs(BENCH_DIR, exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute("""
CREATE TABLE pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    author TEXT,
    version TEXT,
    current_text TEXT,
    srcs_blob BLOB NOT NULL
)
""")
conn.execute("""
CREATE TABLE revisions (
    page_id INTEGER NOT NULL,
    version TEXT NOT NULL,
    block_bytes BLOB NOT NULL,
    author TEXT,
    log TEXT,
    date TEXT,
    PRIMARY KEY (page_id, version)
)
""")
conn.commit()

rcs = SimpleRCS(None)  # in-memory BytesIO-backed instance
lines = [f"line {i}" for i in range(LINES)]
content = "\n".join(lines) + "\n"
now = "2026-08-04T00:00:00"

ver = rcs.commit(content, author="wkpark", log="init")
rcs._load_head(force=True)  # commit() leaves head_info stale; refresh for the loop below
blob = rcs.stream.getvalue()
conn.execute(
    "INSERT INTO pages (page_name, created_at, updated_at, author, version, current_text, srcs_blob) "
    "VALUES (?,?,?,?,?,?,?)",
    ("TestPage", now, now, "wkpark", ver, content, blob),
)
conn.execute(
    "INSERT INTO revisions (page_id, version, block_bytes, author, log, date) VALUES (1,?,?,?,?,?)",
    (ver, blob, "wkpark", "init", now),
)
conn.commit()

a_times = []  # design A: single-column full-blob UPDATE
b_times = []  # design B proxy: append-only tail INSERT (no rewrite of prior rows)
blob_sizes = []
tail_sizes = []

for i in range(N):
    lines[i % LINES] = f"line {i % LINES} rev{i}"
    content = "\n".join(lines) + "\n"

    head_start_before = rcs.head_info["start"]  # byte offset that commit() will seek-back to
    ver = rcs.commit(content, author="wkpark", log=f"rev{i}")
    rcs._load_head(force=True)  # refresh so next iteration's head_start_before is accurate
    full_blob = rcs.stream.getvalue()
    tail_bytes = full_blob[head_start_before:]  # what actually changed on disk this commit

    blob_sizes.append(len(full_blob))
    tail_sizes.append(len(tail_bytes))

    # --- Design A: rewrite the whole srcs_blob column ---
    t0 = time.perf_counter()
    conn.execute(
        "UPDATE pages SET updated_at=?, author=?, version=?, current_text=?, srcs_blob=? WHERE id=1",
        (now, "wkpark", ver, content, full_blob),
    )
    conn.commit()
    a_times.append(time.perf_counter() - t0)

    # --- Design B proxy: only the changed tail as a normalized row op ---
    t0 = time.perf_counter()
    conn.execute(
        "UPDATE revisions SET block_bytes=? WHERE page_id=1 AND version=(SELECT MAX(version) FROM revisions WHERE page_id=1)",  # noqa: E501
        (tail_bytes[: len(tail_bytes) // 2],),  # placeholder: former-HEAD-as-delta half (approx)
    )
    conn.execute(
        "INSERT INTO revisions (page_id, version, block_bytes, author, log, date) VALUES (1,?,?,?,?,?)",
        (ver, tail_bytes[len(tail_bytes) // 2 :], "wkpark", f"rev{i}", now),  # noqa: E203
    )
    conn.commit()
    b_times.append(time.perf_counter() - t0)

    if (i + 1) % 200 == 0:
        print(
            f"rev {i + 1:4d}: total_blob={blob_sizes[-1]:7d}B  tail={tail_sizes[-1]:5d}B  "
            f"A(update whole blob)={a_times[-1] * 1000:7.3f}ms  B(row op)={b_times[-1] * 1000:7.3f}ms"
        )

conn.close()


def stats(label, data, unit_scale=1000):
    n = len(data)
    first = data[: n // 10]
    last = data[-n // 10 :]
    avg_first = sum(first) / len(first) * unit_scale
    avg_last = sum(last) / len(last) * unit_scale
    print(f"{label}: first 10% avg={avg_first:.3f}ms  last 10% avg={avg_last:.3f}ms  ratio={avg_last / avg_first:.2f}x")


print()
print(f"Final blob size: {blob_sizes[-1]} bytes ({blob_sizes[-1] / 1024:.1f} KB) after {N} commits")
print(f"Avg tail size per commit: {sum(tail_sizes) / len(tail_sizes):.1f} bytes")
stats("Design A (whole-blob UPDATE)", a_times)
stats("Design B (row-op proxy)", b_times)
print()
print(f"Total time A: {sum(a_times) * 1000:.1f}ms   Total time B: {sum(b_times) * 1000:.1f}ms")

os.remove(DB_PATH)
