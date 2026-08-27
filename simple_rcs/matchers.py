"""Diff backend selection.

The Cython Myers matchers are the preferred backend and are used whenever they
were compiled. They are optional by design (see setup.py), so this module
resolves a backend once at import time and falls back when one is unavailable:

    SIMPLE_RCS_MATCHER (if set) -> Cython Myers -> StreamSequenceMatcher

`StreamSequenceMatcher` is the fallback, not the pure-Python Myers twins: those
are reference implementations from the first cut of the library and hit an
O(ND) worst case on dissimilar inputs. They stay selectable by name for
debugging and benchmarking, never by fallback.

Backend names match the ids `tools/bench_diff.py` prints, so a benchmark run
can be turned into a setting directly:

    SIMPLE_RCS_MATCHER=ses_cython uv run tools/srcs_commit.py FILE -m msg

Switching backends changes which of several equally valid opcode sequences a
delta is built from, so the stored delta bytes differ between backends. Block
hashes cover the logical full content rather than the stored representation, so
histories stay verifiable across a backend change.
"""

import logging
import os
from collections.abc import Callable
from typing import BinaryIO, Protocol

from .pydifflib import StreamSequenceMatcher

logger = logging.getLogger(__name__)

ENV_VAR = "SIMPLE_RCS_MATCHER"
DEFAULT_ORDER = ("dmp_cython", "ses_cython", "stream")

# name -> (module, attribute). "stream" is handled separately: it already
# speaks the stream interface and needs no adapter.
_MYERS_BACKENDS = {
    "dmp_cython": ("simple_rcs._myersdiff_dmp", "MyersSequenceMatcher"),
    "ses_cython": ("simple_rcs._myersdiff_ses", "MyersSequenceMatcher"),
    "dmp_py": ("simple_rcs.myersdiff_dmp", "MyersSequenceMatcher"),
    "ses_py": ("simple_rcs.myersdiff_ses", "MyersSequenceMatcher"),
}
KNOWN_BACKENDS = ("stream", *_MYERS_BACKENDS)


class Matcher(Protocol):
    """What simple_rcs.py's text-delta path needs from a backend."""

    def get_opcodes(self) -> list[tuple[str, int, int, int, int]]: ...

    def get_lines_from_stream(self, stream_type: str, start_index: int, end_index: int) -> list[bytes]: ...


class _MyersStreamAdapter:
    """Gives a line-oriented Myers matcher the stream interface.

    The Myers backends take two lists of lines and expose only get_opcodes();
    StreamSequenceMatcher takes streams and also serves the line ranges the
    caller needs to emit. This holds the lines so both halves are available.

    Newlines are stripped for comparison (the Myers matchers hash whole
    elements, so a line and the same line with a trailing newline would not
    match) but kept in what get_lines_from_stream returns, matching
    StreamSequenceMatcher's keepends=True behaviour.
    """

    def __init__(self, matcher_cls, a_stream: BinaryIO, b_stream: BinaryIO) -> None:
        a_stream.seek(0)
        b_stream.seek(0)
        self._a_lines = a_stream.read().splitlines(keepends=True)
        self._b_lines = b_stream.read().splitlines(keepends=True)
        self._matcher = matcher_cls(
            a=[line.rstrip(b"\r\n") for line in self._a_lines],
            b=[line.rstrip(b"\r\n") for line in self._b_lines],
        )

    def get_opcodes(self) -> list[tuple[str, int, int, int, int]]:
        return list(self._matcher.get_opcodes())

    def get_lines_from_stream(self, stream_type: str, start_index: int, end_index: int) -> list[bytes]:
        lines = self._a_lines if stream_type == "a" else self._b_lines
        if start_index < 0 or end_index < start_index:
            return []
        return lines[start_index:end_index]


def _load(name: str) -> Callable[[BinaryIO, BinaryIO], Matcher] | None:
    """Return a factory for `name`, or None if that backend is unavailable."""
    if name == "stream":
        return lambda a, b: StreamSequenceMatcher(a, b, chunk_size=None)

    target = _MYERS_BACKENDS.get(name)
    if target is None:
        return None

    module_name, attr = target
    try:
        module = __import__(module_name, fromlist=[attr])
        matcher_cls = getattr(module, attr)
    except (ImportError, AttributeError):
        # Not compiled, built for another ABI, or built for another platform.
        return None

    def factory(a: BinaryIO, b: BinaryIO) -> Matcher:
        return _MyersStreamAdapter(matcher_cls, a, b)

    return factory


def _resolve() -> tuple[str, Callable[[BinaryIO, BinaryIO], Matcher]]:
    requested = os.environ.get(ENV_VAR)
    if requested:
        if requested not in KNOWN_BACKENDS:
            logger.warning(
                f"{ENV_VAR}={requested!r} is not a known backend "
                f"(known: {', '.join(KNOWN_BACKENDS)}); using the default order"
            )
        else:
            factory = _load(requested)
            if factory is not None:
                return requested, factory
            logger.warning(f"{ENV_VAR}={requested!r} is unavailable (not compiled?); using the default order")

    for name in DEFAULT_ORDER:
        factory = _load(name)
        if factory is not None:
            if name != DEFAULT_ORDER[0]:
                logger.debug("matcher backend %s unavailable; using %s", DEFAULT_ORDER[0], name)
            return name, factory

    # "stream" is pure Python and always importable, so this is unreachable
    # unless pydifflib itself is broken.
    raise ImportError("No diff backend available, including StreamSequenceMatcher")


ACTIVE_BACKEND, _factory = _resolve()


def new_matcher(a_stream: BinaryIO, b_stream: BinaryIO) -> Matcher:
    """Build a matcher over two line-mode streams using the active backend."""
    return _factory(a_stream, b_stream)


def available_backends() -> list[str]:
    """Backend names that can actually be constructed in this interpreter."""
    return [name for name in KNOWN_BACKENDS if _load(name) is not None]
