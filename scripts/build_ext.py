#!/usr/bin/env python3
# ruff: noqa: T201
"""Build the Cython matchers in place, for the benchmark tools.

`simple_rcs/_myersdiff_{ses,dmp}.pyx` are benchmark-only: nothing under
`simple_rcs/` imports them, so they are deliberately left out of the published
distribution, which is a pure-Python wheel. Build them locally when you want
`tools/bench_diff.py` or `tools/compare_memory_usage.py` to include them:

    uv run scripts/build_ext.py

`packages=[]` is not decoration: without it setuptools falls back to
flat-layout auto-discovery and refuses to build, having found the scratch
directories sitting in the repo root (wiki/, trash/, patches/, ...).
"""
import os
import sys

from Cython.Build import cythonize
from setuptools import setup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["simple_rcs/_myersdiff_ses.pyx", "simple_rcs/_myersdiff_dmp.pyx"]

os.chdir(ROOT)
sys.argv = [sys.argv[0], "build_ext", "--inplace"]
setup(name="simple_rcs_ext", packages=[], ext_modules=cythonize(SOURCES, language_level="3"))
print("\nBuilt in place. tools/bench_diff.py will now include the Cython matchers.")
