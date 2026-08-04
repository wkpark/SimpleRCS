#!/usr/bin/env python3
# ruff: noqa: T201
"""Design A (SimpleRCS delta-chain blob, one column per page) vs design C
(naive full-text-per-revision, one row per revision), each measured in its
OWN SQLite file so neither design's numbers are distorted by the other
table's writes sharing the same file.
"""
import os
import sqlite3
import time

from simple_rcs.simple_rcs import SimpleRCS

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".bench_tmp")
PAGES = 50
REVS_PER_PAGE = 200
LINES = 50


def run_design_a():
    path = os.path.join(BENCH_DIR, "a_only.sqlite3")
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE pages (id INTEGER PRIMARY KEY, srcs_blob BLOB NOT NULL)")
    conn.commit()

    for page_id in range(1, PAGES + 1):
        rcs = SimpleRCS(None)
        lines = [f"{page_id} line {i}" for i in range(LINES)]
        content = "\n".join(lines) + "\n"
        rcs.commit(content, author="a", log="init")
        rcs._load_head(force=True)
        conn.execute("INSERT INTO pages (id, srcs_blob) VALUES (?, ?)", (page_id, rcs.stream.getvalue()))
        conn.commit()

        for i in range(REVS_PER_PAGE):
            lines[i % LINES] = f"{page_id} line {i % LINES} rev{i}"
            content = "\n".join(lines) + "\n"
            rcs.commit(content, author="a", log=f"rev{i}")
            rcs._load_head(force=True)
            conn.execute("UPDATE pages SET srcs_blob=? WHERE id=?", (rcs.stream.getvalue(), page_id))
            conn.commit()

    logical = conn.execute("SELECT SUM(LENGTH(srcs_blob)) FROM pages").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    conn.close()
    size = os.path.getsize(path)
    os.remove(path)
    return {"logical": logical, "file": size, "page_count": page_count, "freelist": freelist}


def run_design_c():
    path = os.path.join(BENCH_DIR, "c_only.sqlite3")
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE page_revisions (
            page_id INTEGER, version TEXT, full_text TEXT NOT NULL,
            PRIMARY KEY (page_id, version)
        )
    """)
    conn.commit()

    for page_id in range(1, PAGES + 1):
        lines = [f"{page_id} line {i}" for i in range(LINES)]
        content = "\n".join(lines) + "\n"
        conn.execute(
            "INSERT INTO page_revisions (page_id, version, full_text) VALUES (?,?,?)", (page_id, "init", content)
        )
        conn.commit()

        for i in range(REVS_PER_PAGE):
            lines[i % LINES] = f"{page_id} line {i % LINES} rev{i}"
            content = "\n".join(lines) + "\n"
            conn.execute(
                "INSERT INTO page_revisions (page_id, version, full_text) VALUES (?,?,?)",
                (page_id, f"rev{i}", content),
            )
            conn.commit()

    logical = conn.execute("SELECT SUM(LENGTH(full_text)) FROM page_revisions").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    conn.close()
    size = os.path.getsize(path)
    os.remove(path)
    return {"logical": logical, "file": size, "page_count": page_count, "freelist": freelist}


os.makedirs(BENCH_DIR, exist_ok=True)

print(f"{PAGES} pages x {REVS_PER_PAGE} revisions, each design in its own .sqlite3 file")

t0 = time.perf_counter()
a = run_design_a()
print(
    f"[Design A only] logical={a['logical'] / 1024 / 1024:.2f}MB  file={a['file'] / 1024 / 1024:.2f}MB  "
    f"ratio={a['file'] / a['logical']:.2f}x  page_count={a['page_count']} freelist={a['freelist']}  "
    f"({time.perf_counter() - t0:.1f}s)"
)

t0 = time.perf_counter()
c = run_design_c()
print(
    f"[Design C only] logical={c['logical'] / 1024 / 1024:.2f}MB  file={c['file'] / 1024 / 1024:.2f}MB  "
    f"ratio={c['file'] / c['logical']:.2f}x  page_count={c['page_count']} freelist={c['freelist']}  "
    f"({time.perf_counter() - t0:.1f}s)"
)
