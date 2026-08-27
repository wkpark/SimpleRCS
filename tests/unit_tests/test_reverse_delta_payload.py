"""Reverse-delta payload round trips that used to come back wrong without an error."""

import importlib
import io

import pytest

from simple_rcs import matchers
from simple_rcs.simple_rcs import SimpleRCS, SimpleRCSCorruptionError


def _two_versions(old: str, new: str) -> SimpleRCS:
    rcs = SimpleRCS(io.BytesIO())
    rcs.commit(old, author="u", log="old")
    rcs.commit(new, author="u", log="new")
    return rcs


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("a\n\nb\n", "a\nb\n"),  # a paragraph break removed mid-document
        ("a\n\n", "a\nb\n"),  # trailing blank line replaced
        ("\n\n", "y\n\n"),  # blank lines only
        ("\n", "x\n"),
        ("a\n\n\nb\n", "a\nb\n"),  # two blank lines in one payload
    ],
)
def test_blank_lines_at_the_end_of_a_payload_survive_checkout(old, new):
    """ "\\n".join(payload) with a trailing "" was byte-identical to no payload, and
    the apply side silently ran short. The blank line vanished from the old version."""
    rcs = _two_versions(old, new)

    assert rcs.checkout("1.0") == old
    assert rcs.checkout("1.1") == new


def test_a_delta_written_by_the_old_encoder_is_repaired_not_refused():
    """The legacy encoding of the case above: 'a1 1' with its blank payload line
    gone. That encoder could only ever lose one line, always the last, and only
    when it was blank -- so the line is recoverable, and refusing to read it
    would make every existing history with a removed paragraph break
    permanently unreadable."""
    rcs = _two_versions("a\n\nb\n", "a\nb\n")
    raw = rcs.stream.getvalue()
    assert b"delta @a1 1\n\n@;" in raw

    legacy = SimpleRCS(io.BytesIO(raw.replace(b"delta @a1 1\n\n@;", b"delta @a1 1\n@;")))

    assert legacy.checkout("1.0") == "a\n\nb\n"


def test_a_delta_written_by_the_old_encoder_is_repaired_on_a_v2_stream():
    """The hash chain is an independent check on the repair: it was computed by
    the old version over the original full text, so it only verifies if the line
    came back exactly."""
    rcs = SimpleRCS()  # v2 header -> verify() actually runs
    assert rcs._version == 2
    rcs.commit("a\n\nb\n", author="u", log="old")
    rcs.commit("a\nb\n", author="u", log="new")
    raw = rcs.get_bytes()

    legacy = raw.replace(b"delta @a1 1\n\n@;", b"delta @a1 1\n@;")
    assert legacy != raw

    assert SimpleRCS(io.BytesIO(legacy)).checkout("1.0") == "a\n\nb\n"
    assert SimpleRCS(io.BytesIO(legacy)).verify() is True


def test_a_delta_short_by_more_than_one_line_is_corruption():
    """No encoder this project has shipped drops two payload lines, so there is
    nothing to reconstruct and the repair above must not stretch to cover it."""
    rcs = _two_versions("a\n\nb\n", "a\nb\n")
    raw = rcs.stream.getvalue()

    damaged = SimpleRCS(io.BytesIO(raw.replace(b"delta @a1 1\n\n@;", b"delta @a1 3\n\n@;")))

    with pytest.raises(SimpleRCSCorruptionError, match="payload"):
        damaged.checkout("1.0")


def test_adjacent_inserts_at_one_anchor_apply_in_emission_order():
    """A backend may emit two 'a' commands at the same line instead of one merged
    insert (the Cython dmp matcher does for 'b' -> 'a b a c'). Each inserts at the
    same index, so applying them in emission order reversed the payloads."""
    rcs = SimpleRCS(io.BytesIO())

    assert rcs._apply_reverse_delta("b\n", "a0 1\na\na1 1\na\na1 1\nc\n") == "a\nb\na\nc\n"


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    importlib.reload(matchers)


@pytest.mark.parametrize("backend", matchers.available_backends())
def test_repeated_lines_round_trip_on_every_backend(monkeypatch, backend):
    """Content, not delta bytes: each backend may encode this differently."""
    monkeypatch.setenv(matchers.ENV_VAR, backend)
    importlib.reload(matchers)
    old, new = "a\nb\na\nc\n", "b\n"

    rcs = _two_versions(old, new)

    assert rcs.checkout("1.0") == old
