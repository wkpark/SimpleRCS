#!/usr/bin/env python3
"""
bench_diff.py — Compare time and memory across all diff matchers.

Usage:
    uv run tools/bench_diff.py                        # synthetic 100KB, 1% diff
    uv run tools/bench_diff.py --size 500             # 500KB synthetic
    uv run tools/bench_diff.py --diff-ratio 0.05      # 5% lines changed
    uv run tools/bench_diff.py --file-a a.py --file-b b.py   # real files
    uv run tools/bench_diff.py --runs 5               # average over 5 runs
    uv run tools/bench_diff.py --skip myers_stream    # skip slow algorithms
"""

import argparse
import importlib
import io
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def generate_lines(size_kb: int, diff_ratio: float) -> tuple[list[bytes], list[bytes]]:
    """Generate two lists of byte-lines for comparison."""
    line = b"This is a test line with some padding to make it longer. Lorem ipsum dolor sit amet.\n"
    num_lines = max(1, (size_kb * 1024) // len(line))
    diff_count = max(1, int(num_lines * diff_ratio))
    diff_indices = set(range(num_lines // 2, num_lines // 2 + diff_count))

    lines_a = [f"{i:08d}: ".encode() + line for i in range(num_lines)]
    lines_b = [
        f"{i:08d}: MODIFIED — this line was changed to show a difference.\n".encode()
        if i in diff_indices
        else f"{i:08d}: ".encode() + line
        for i in range(num_lines)
    ]
    return lines_a, lines_b


def load_file_lines(path: str) -> list[bytes]:
    with open(path, "rb") as f:
        return f.readlines()


def to_stream(lines: list[bytes]) -> io.BytesIO:
    return io.BytesIO(b"".join(lines))


def stripped(lines: list[bytes]) -> list[bytes]:
    """Strip CR/LF for matchers that hash without newlines."""
    return [ln.rstrip(b"\r\n") for ln in lines]


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

@dataclass
class BenchCase:
    name: str
    label: str          # short name for the table
    variant: str        # "py" or "cython"
    fn: Any             # callable(lines_a, lines_b) -> list[opcodes]


def _try_import(module: str, attr: str) -> Any | None:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr)
    except (ImportError, AttributeError):
        return None


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []

    # 1. difflib stdlib (baseline)
    from difflib import SequenceMatcher as _StdSM
    def _difflib(la, lb):
        m = _StdSM(None, la, lb, autojunk=False)
        return list(m.get_opcodes())
    cases.append(BenchCase("difflib", "difflib (stdlib)", "py", _difflib))

    # 2. StreamSequenceMatcher (greedy, line mode)
    from simple_rcs.pydifflib import StreamSequenceMatcher as _SSM
    def _stream_greedy(la, lb):
        m = _SSM(to_stream(la), to_stream(lb), chunk_size=None)
        return list(m.get_opcodes())
    cases.append(BenchCase("stream_greedy", "StreamSequenceMatcher (greedy)", "py", _stream_greedy))

    # 3. StreamTextSequenceMatcher (difflib port, stream)
    from simple_rcs.pydifflib import StreamTextSequenceMatcher as _STSM
    def _stream_text(la, lb):
        m = _STSM(to_stream(la), to_stream(lb), chunk_size=None)
        return list(m.get_opcodes())
    cases.append(BenchCase("stream_text", "StreamTextSequenceMatcher (difflib port)", "py", _stream_text))

    # 4. myersdiff.MyersSequenceMatcher (linked-list, pure Python)
    from simple_rcs.myersdiff import MyersSequenceMatcher as _MyersPy
    def _myers_py(la, lb):
        m = _MyersPy(a=stripped(la), b=stripped(lb))
        return list(m.get_opcodes())
    cases.append(BenchCase("myers_py", "MyersSequenceMatcher (linked-list)", "py", _myers_py))

    # 5. myersdiff.MyersStreamSequenceMatcher (Myers stream, pure Python)
    from simple_rcs.myersdiff import MyersStreamSequenceMatcher as _MyersStream
    def _myers_stream(la, lb):
        m = _MyersStream(to_stream(la), to_stream(lb), chunk_size=None)
        return list(m.get_opcodes())
    cases.append(BenchCase("myers_stream", "MyersStreamSequenceMatcher", "py", _myers_stream))

    # 6. myersdiff_ses.MyersSequenceMatcher (SES/libmba, pure Python)
    from simple_rcs.myersdiff_ses import MyersSequenceMatcher as _SESPy
    def _ses_py(la, lb):
        m = _SESPy(a=stripped(la), b=stripped(lb))
        return list(m.get_opcodes())
    cases.append(BenchCase("ses_py", "MyersSequenceMatcher SES (libmba)", "py", _ses_py))

    # 7. myersdiff_dmp.MyersSequenceMatcher (DMP, pure Python)
    from simple_rcs.myersdiff_dmp import MyersSequenceMatcher as _DMPPy
    def _dmp_py(la, lb):
        m = _DMPPy(a=stripped(la), b=stripped(lb))
        return list(m.get_opcodes())
    cases.append(BenchCase("dmp_py", "MyersSequenceMatcher DMP", "py", _dmp_py))

    # 8. _myersdiff_ses (Cython)
    _SESCy = _try_import("simple_rcs._myersdiff_ses", "MyersSequenceMatcher")
    if _SESCy:
        def _ses_cy(la, lb):
            m = _SESCy(a=stripped(la), b=stripped(lb))
            return list(m.get_opcodes())
        cases.append(BenchCase("ses_cython", "MyersSequenceMatcher SES (Cython)", "cython", _ses_cy))
    else:
        print("  [warn] _myersdiff_ses Cython module not found, skipping.")

    # 9. _myersdiff_dmp (Cython)
    _DMPCy = _try_import("simple_rcs._myersdiff_dmp", "MyersSequenceMatcher")
    if _DMPCy:
        def _dmp_cy(la, lb):
            m = _DMPCy(a=stripped(la), b=stripped(lb))
            return list(m.get_opcodes())
        cases.append(BenchCase("dmp_cython", "MyersSequenceMatcher DMP (Cython)", "cython", _dmp_cy))
    else:
        print("  [warn] _myersdiff_dmp Cython module not found, skipping.")

    return cases


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class Result:
    case: BenchCase
    time_ms: float
    peak_kb: float
    num_opcodes: int
    skipped: bool = False
    error: str = ""


def measure(case: BenchCase, lines_a: list[bytes], lines_b: list[bytes], runs: int) -> Result:
    times = []
    peak_kb = 0.0
    num_opcodes = 0

    for run in range(runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            opcodes = case.fn(lines_a, lines_b)
        except Exception as e:
            tracemalloc.stop()
            return Result(case, 0, 0, 0, skipped=True, error=str(e))
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(t1 - t0)
        peak_kb = max(peak_kb, peak / 1024)
        if run == 0:
            num_opcodes = len(opcodes)

    best_ms = min(times) * 1000
    return Result(case, best_ms, peak_kb, num_opcodes)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_VARIANT_BADGE = {"py": "py   ", "cython": "cython"}
_COL_W = 46


def print_results(results: list[Result], baseline_ms: float) -> None:
    sep = "-" * 90
    header = f"{'Algorithm':<{_COL_W}} {'Variant':<8} {'Time (ms)':>10} {'Peak (KB)':>10} {'Speedup':>8} {'Opcodes':>8}"
    print(sep)
    print(header)
    print(sep)

    for r in results:
        if r.skipped:
            flag = f"SKIP ({r.error[:30]})"
            print(f"  {r.case.label:<{_COL_W}} {flag}")
            continue

        speedup = baseline_ms / r.time_ms if r.time_ms > 0 else float("inf")
        variant = _VARIANT_BADGE.get(r.case.variant, r.case.variant)
        speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "1.00x (baseline)"
        print(
            f"  {r.case.label:<{_COL_W}} {variant:<8} {r.time_ms:>10.1f} {r.peak_kb:>10.1f} {speedup_str:>8} {r.num_opcodes:>8}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all diff matchers (time + memory).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--size", type=int, default=100, metavar="KB",
                        help="Synthetic test file size in KiB (default: 100)")
    parser.add_argument("--diff-ratio", type=float, default=0.01, metavar="RATIO",
                        help="Fraction of lines that differ (default: 0.01 = 1%%)")
    parser.add_argument("--file-a", metavar="PATH", help="Use real file A instead of synthetic data")
    parser.add_argument("--file-b", metavar="PATH", help="Use real file B instead of synthetic data")
    parser.add_argument("--runs", type=int, default=3, metavar="N",
                        help="Number of runs per algorithm; best time is reported (default: 3)")
    parser.add_argument("--skip", nargs="+", metavar="NAME",
                        help="Algorithm names to skip (e.g. myers_py myers_stream)")
    args = parser.parse_args()

    # --- Data ---
    if args.file_a and args.file_b:
        print(f"Loading files: {args.file_a!r}, {args.file_b!r}")
        lines_a = load_file_lines(args.file_a)
        lines_b = load_file_lines(args.file_b)
    else:
        print(f"Generating synthetic data: {args.size} KB, diff-ratio={args.diff_ratio:.1%}")
        lines_a, lines_b = generate_lines(args.size, args.diff_ratio)

    print(f"Lines: {len(lines_a):,} / {len(lines_b):,}  |  runs per algorithm: {args.runs}\n")

    # --- Cases ---
    skip_set = set(args.skip or [])
    cases = [c for c in build_cases() if c.name not in skip_set]

    # --- Run ---
    results: list[Result] = []
    baseline_ms = None

    for case in cases:
        sys.stdout.write(f"  running {case.label!r} ... ")
        sys.stdout.flush()
        r = measure(case, lines_a, lines_b, args.runs)
        results.append(r)
        if r.skipped:
            print(f"SKIPPED ({r.error[:40]})")
        else:
            print(f"{r.time_ms:.1f} ms")
        if case.name == "difflib" and not r.skipped:
            baseline_ms = r.time_ms

    # Sort by time (skipped at bottom)
    results.sort(key=lambda r: (r.skipped, r.time_ms))

    print()
    print_results(results, baseline_ms or 1.0)


if __name__ == "__main__":
    main()
