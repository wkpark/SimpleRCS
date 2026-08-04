#!/usr/bin/env python3
# ruff: noqa: T201
"""Empirical test at multi-page scale: SimpleRCS delta-chain storage
(design A: one growing blob column per page, holding the full reverse-delta
chain) vs naive full-snapshot-per-revision storage (design C: one row per
revision holding the complete page text), across many independent pages.

Approximates a "1000 pages x ~200 revisions" personal-wiki scale by directly
measuring a smaller page count (pages are independent, so cost/storage scale
linearly with page count) and extrapolating.

Measures logical column bytes (SUM(LENGTH(...))), not on-disk file size —
both tables share one file here, and their writes interleave every
iteration, which distorts file-size numbers. For an on-disk-size comparison,
see wiki_isolation_bench.py, which runs each design in its own file.
"""
import os
import sqlite3
import time

from simple_rcs.simple_rcs import SimpleRCS

# Written next to the repo (not system /tmp, which may be a small/shared
# partition) and removed again at the end of the run — see .gitignore.
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".bench_tmp")
DB_PATH = os.path.join(BENCH_DIR, "wiki_multipage_bench.sqlite3")

PAGES = 50
REVS_PER_PAGE = 200
LINES = 50
TARGET_PAGES = 1000  # what we're extrapolating to

os.makedirs(BENCH_DIR, exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute("""
CREATE TABLE pages (            -- design A: our way (SimpleRCS delta-chain, one blob per page)
    id INTEGER PRIMARY KEY,
    page_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    author TEXT,
    version TEXT,
    current_text TEXT,          -- HEAD cache/replica
    srcs_blob BLOB NOT NULL     -- full delta chain + HEAD
)
""")
conn.execute("""
CREATE TABLE page_revisions (   -- design C: naive full snapshot per revision
    page_id INTEGER NOT NULL,
    version TEXT NOT NULL,
    full_text TEXT NOT NULL,
    author TEXT,
    log TEXT,
    date TEXT,
    PRIMARY KEY (page_id, version)
)
""")
conn.commit()

now = "2026-08-04T00:00:00"
a_times = []
c_times = []

t_start = time.perf_counter()

for page_id in range(1, PAGES + 1):
    rcs = SimpleRCS(None)
    lines = [f"{page_id} line {i}" for i in range(LINES)]
    content = "\n".join(lines) + "\n"

    ver = rcs.commit(content, author="wkpark", log="init")
    rcs._load_head(force=True)
    blob = rcs.stream.getvalue()

    conn.execute(
        "INSERT INTO pages (id, page_name, created_at, updated_at, author, version, current_text, srcs_blob) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (page_id, f"Page{page_id}", now, now, "wkpark", ver, content, blob),
    )
    conn.execute(
        "INSERT INTO page_revisions (page_id, version, full_text, author, log, date) VALUES (?,?,?,?,?,?)",
        (page_id, ver, content, "wkpark", "init", now),
    )
    conn.commit()

    for i in range(REVS_PER_PAGE):
        lines[i % LINES] = f"{page_id} line {i % LINES} rev{i}"
        content = "\n".join(lines) + "\n"
        ver = rcs.commit(content, author="wkpark", log=f"rev{i}")
        rcs._load_head(force=True)
        blob = rcs.stream.getvalue()

        t0 = time.perf_counter()
        conn.execute(
            "UPDATE pages SET updated_at=?, author=?, version=?, current_text=?, srcs_blob=? WHERE id=?",
            (now, "wkpark", ver, content, blob, page_id),
        )
        conn.commit()
        a_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        conn.execute(
            "INSERT INTO page_revisions (page_id, version, full_text, author, log, date) VALUES (?,?,?,?,?,?)",
            (page_id, ver, content, "wkpark", f"rev{i}", now),
        )
        conn.commit()
        c_times.append(time.perf_counter() - t0)

wall = time.perf_counter() - t_start

a_bytes = conn.execute("SELECT SUM(LENGTH(srcs_blob)) FROM pages").fetchone()[0]
c_bytes = conn.execute("SELECT SUM(LENGTH(full_text)) FROM page_revisions").fetchone()[0]
conn.close()


def stats(label, data):
    n = len(data)
    avg = sum(data) / n * 1000
    total = sum(data) * 1000
    print(f"{label}: avg={avg:.3f}ms  total={total:.1f}ms  n={n}")


scale = TARGET_PAGES / PAGES

print(f"{PAGES} pages x {REVS_PER_PAGE} revisions = {PAGES * REVS_PER_PAGE} commits, wall={wall:.1f}s")
stats("Design A (our way: delta-chain blob, UPDATE)", a_times)
stats("Design C (naive full-snapshot, INSERT)", c_times)
print()
print(f"Design A stored bytes (delta chain + HEAD, all pages): {a_bytes:>10d}  ({a_bytes / 1024 / 1024:.2f} MB)")
print(f"Design C stored bytes (all full-text revisions):       {c_bytes:>10d}  ({c_bytes / 1024 / 1024:.2f} MB)")
print(f"Storage ratio C/A: {c_bytes / a_bytes:.2f}x")
print()
print(f"Extrapolated to {TARGET_PAGES} pages x {REVS_PER_PAGE} revisions (x{scale:.0f}):")
print(f"  Design A storage: ~{a_bytes * scale / 1024 / 1024:.1f} MB")
print(f"  Design C storage: ~{c_bytes * scale / 1024 / 1024:.1f} MB")
print(f"  Design A total commit time: ~{sum(a_times) * scale:.1f}s")
print(f"  Design C total commit time: ~{sum(c_times) * scale:.1f}s")

os.remove(DB_PATH)
