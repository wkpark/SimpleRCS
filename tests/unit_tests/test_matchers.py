import importlib
import io
import sys

import pytest

from simple_rcs import matchers
from simple_rcs.simple_rcs import SimpleRCS

REVISIONS = [
    "alpha\nbravo\ncharlie\ndelta\n",
    "alpha\nbravo MODIFIED\ncharlie\ndelta\n",
    "alpha\ncharlie\ndelta\necho\n",
    "alpha\ncharlie\nfoxtrot\ngolf\ndelta\necho\n",
    "totally\ndifferent\ncontent\nhere\n",
]


def _reload_with(monkeypatch, value):
    """Re-resolve the backend under a given SIMPLE_RCS_MATCHER value."""
    if value is None:
        monkeypatch.delenv(matchers.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(matchers.ENV_VAR, value)
    return importlib.reload(matchers)


@pytest.fixture(autouse=True)
def _restore_module():
    """Leave the module resolved as the ambient environment would have it.

    reload() re-executes in the *same* module namespace, so rebinding _factory
    there is enough: simple_rcs.py holds a reference to new_matcher, whose
    globals are that namespace. Reloading simple_rcs.py itself would replace
    SimpleRCSCorruptionError and break `pytest.raises` identity in other test
    modules, so don't.
    """
    yield
    importlib.reload(matchers)


def test_stream_backend_is_always_available():
    assert "stream" in matchers.available_backends()


def test_default_prefers_cython_when_compiled(monkeypatch):
    mod = _reload_with(monkeypatch, None)
    available = mod.available_backends()
    expected = next(name for name in mod.DEFAULT_ORDER if name in available)
    assert mod.ACTIVE_BACKEND == expected


def test_env_var_selects_a_backend(monkeypatch):
    mod = _reload_with(monkeypatch, "stream")
    assert mod.ACTIVE_BACKEND == "stream"


def test_unknown_backend_falls_back_to_the_default_order(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        mod = _reload_with(monkeypatch, "no_such_backend")
    assert mod.ACTIVE_BACKEND in mod.DEFAULT_ORDER
    assert "not a known backend" in caplog.text


def test_uncompiled_backend_falls_back_to_the_default_order(monkeypatch, caplog):
    """A named-but-unbuildable backend degrades instead of raising.

    Blocking through sys.modules rather than patching the registry: reload()
    rebuilds the module's own tables, so only interpreter-level state survives
    into the re-resolved module.
    """
    monkeypatch.setitem(sys.modules, "simple_rcs._myersdiff_dmp", None)
    with caplog.at_level("WARNING"):
        mod = _reload_with(monkeypatch, "dmp_cython")
    assert mod.ACTIVE_BACKEND != "dmp_cython"
    assert "unavailable" in caplog.text


def test_adapter_line_ranges_match_the_stream_matcher():
    """get_lines_from_stream must keep newlines, as StreamSequenceMatcher does."""
    text = b"one\ntwo\nthree\nfour\n"
    stream_matcher = matchers._load("stream")(io.BytesIO(text), io.BytesIO(text))
    assert stream_matcher.get_lines_from_stream("b", 1, 3) == [b"two\n", b"three\n"]

    for name in matchers.available_backends():
        if name == "stream":
            continue
        adapted = matchers._load(name)(io.BytesIO(text), io.BytesIO(text))
        assert adapted.get_lines_from_stream("b", 1, 3) == [b"two\n", b"three\n"], name
        assert adapted.get_lines_from_stream("b", 2, 1) == [], name


@pytest.mark.parametrize("backend", matchers.available_backends())
def test_history_round_trips_on_every_backend(monkeypatch, backend, tmp_path):
    """Every backend must reproduce identical content for identical input.

    Backends legitimately emit different (equally valid) opcode sequences, so
    the stored delta bytes may differ. What must not differ is the content that
    comes back out, or whether the hash chain verifies.
    """
    mod = _reload_with(monkeypatch, backend)
    assert mod.ACTIVE_BACKEND == backend

    path = tmp_path / f"{backend}.srcs"
    rcs = SimpleRCS(str(path))
    versions = [rcs.commit(text, author="tester", log=f"rev{i}") for i, text in enumerate(REVISIONS)]

    assert versions == ["1.0", "1.1", "1.2", "1.3", "1.4"]
    for ver, expected in zip(versions, REVISIONS, strict=True):
        assert rcs.checkout(ver) == expected, f"{backend} lost content at {ver}"
    assert rcs.verify() is True, f"{backend} broke the hash chain"


def test_backends_agree_with_the_stream_backend(monkeypatch, tmp_path):
    """Cross-check every backend against the previous production engine."""
    outputs = {}
    for backend in matchers.available_backends():
        _reload_with(monkeypatch, backend)
        rcs = SimpleRCS(str(tmp_path / f"agree_{backend}.srcs"))
        for i, text in enumerate(REVISIONS):
            rcs.commit(text, author="tester", log=f"rev{i}")
        outputs[backend] = [rcs.checkout(f"1.{i}") for i in range(len(REVISIONS))]

    baseline = outputs["stream"]
    for backend, got in outputs.items():
        assert got == baseline, f"{backend} disagrees with the stream backend"


def test_in_memory_commit_uses_the_active_backend():
    """Sanity check that the wiring is live, not just importable."""
    rcs = SimpleRCS(None)
    rcs.commit("a\nb\n", author="t", log="v1")
    rcs.commit("a\nc\n", author="t", log="v2")
    assert rcs.checkout("1.0") == "a\nb\n"
    assert rcs.checkout("1.1") == "a\nc\n"
