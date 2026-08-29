"""Refusing a commit that would change nothing.

The library stays permissive -- re-signing, or correcting an author or a log
message, is a real reason to store the same bytes again -- so the decision is a
question `matches_head` answers and `srcs_commit.py` acts on, the way git
splits `commit-tree` from `git commit`.

What is worth pinning is the trailing newline. Text is stored as if it ended in
one whether or not it did, so the obvious `content == checkout()` calls every
such file changed on every commit. That failure is silent: the tool would look
like it worked and would refuse nothing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from simple_rcs.simple_rcs import SimpleRCS

TOOL = Path(__file__).parents[2] / "tools" / "srcs_commit.py"


def _rcs(*revisions: str | bytes) -> SimpleRCS:
    rcs = SimpleRCS(None)
    for i, content in enumerate(revisions):
        rcs.commit(content, author="tester", log=f"v{i}", encoding="raw")
    return rcs


# --------------------------------------------------------------------------
# matches_head
# --------------------------------------------------------------------------

def test_identical_text_matches():
    assert _rcs("a\nb\nc\n").matches_head("a\nb\nc\n") is True


def test_changed_text_does_not():
    assert _rcs("a\nb\nc\n").matches_head("a\nb\nZ\n") is False


def test_content_without_a_trailing_newline_matches_what_it_was_stored_as():
    """The rule this whole helper exists for. `x\\ny` is stored as `x\\ny\\n`,
    so comparing against the raw file would report a change every time and the
    refusal would never fire for any file that lacks a final newline."""
    rcs = _rcs("x\ny")

    assert rcs.checkout() == "x\ny\n"
    assert rcs.matches_head("x\ny") is True


def test_adding_the_newline_yourself_is_still_a_match():
    assert _rcs("x\ny").matches_head("x\ny\n") is True


def test_an_empty_history_never_matches():
    """A first commit is not a repeat, and there is no HEAD to compare with."""
    assert SimpleRCS(None).matches_head("anything\n") is False


def test_only_the_head_is_compared_not_the_whole_history():
    """Restoring an older revision is a change, not a no-op."""
    rcs = _rcs("one\n", "two\n")

    assert rcs.matches_head("two\n") is True
    assert rcs.matches_head("one\n") is False


def test_identical_bytes_match():
    assert _rcs(b"\x00\x01\x02").matches_head(b"\x00\x01\x02") is True


def test_changed_bytes_do_not():
    assert _rcs(b"\x00\x01\x02").matches_head(b"\x00\x01\x03") is False


def test_bytes_never_match_a_text_head():
    """Committing bytes over text is a type change: it is stored as a full
    snapshot rather than a delta, so it is not a no-op even when the bytes are
    the same."""
    assert _rcs("abc\n").matches_head(b"abc\n") is False


def test_text_never_matches_a_binary_head():
    assert _rcs(b"abc\n").matches_head("abc\n") is False


def test_empty_content_is_stored_as_empty_and_matches():
    """The newline rule skips empty text -- a zero-byte file must not come back
    as one byte -- so the comparison has to skip it in the same place."""
    rcs = _rcs("")

    assert rcs.checkout() == ""
    assert rcs.matches_head("") is True
    assert rcs.matches_head("\n") is False


def test_a_metadata_only_read_does_not_poison_the_answer():
    """log() can leave HEAD cached without its content. Asking then has to
    reload rather than compare against a `text` that is not there."""
    rcs = _rcs("a\nb\n")
    rcs.log()

    assert rcs.matches_head("a\nb\n") is True
    assert rcs.matches_head("a\nZ\n") is False


def test_a_stream_is_rejected_rather_than_answered_wrongly():
    """commit() takes a stream too; comparing one would quietly answer
    "changed", which is the wrong direction for this to fail in."""
    import io

    with pytest.raises(TypeError, match="read a stream first"):
        _rcs("a\n").matches_head(io.BytesIO(b"a\n"))


def test_asking_does_not_disturb_the_next_commit():
    rcs = _rcs("a\n")
    rcs.matches_head("a\n")
    rcs.commit("b\n", author="tester", log="v1")

    assert [entry["ver"] for entry in rcs.log()] == ["1.1", "1.0"]
    assert rcs.checkout("1.0") == "a\n"


def test_the_library_still_allows_it():
    """The refusal is the CLI's policy, not the format's: re-signing or fixing
    a log message has to remain possible."""
    rcs = _rcs("a\n")
    rcs.commit("a\n", author="tester", log="re-signed")

    assert [entry["ver"] for entry in rcs.log()] == ["1.1", "1.0"]
    assert rcs.checkout("1.0") == rcs.checkout("1.1") == "a\n"


# --------------------------------------------------------------------------
# srcs_commit.py
# --------------------------------------------------------------------------

def _commit(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args, "--no-sign"], cwd=cwd,
        capture_output=True, text=True,
    )


def _versions(cwd: Path, name: str) -> list[str]:
    rcs = SimpleRCS(str(cwd / ".srcs" / f"{name}.srcs"))
    return [entry["ver"] for entry in rcs.log()]


def test_the_first_commit_of_a_file_is_never_refused(tmp_path):
    (tmp_path / "f.txt").write_text("a\n")

    assert _commit(tmp_path, "f.txt", "-m", "v1").returncode == 0


def test_an_unchanged_file_is_refused(tmp_path):
    (tmp_path / "f.txt").write_text("a\n")
    _commit(tmp_path, "f.txt", "-m", "v1")

    result = _commit(tmp_path, "f.txt", "-m", "again")

    assert result.returncode == 1, result.stdout
    assert "nothing to commit" in result.stdout
    assert "v1.0" in result.stdout
    assert _versions(tmp_path, "f.txt") == ["1.0"]


def test_a_file_without_a_trailing_newline_is_refused_too(tmp_path):
    """End to end, the case a naive comparison lets through."""
    (tmp_path / "f.txt").write_text("x\ny")
    _commit(tmp_path, "f.txt", "-m", "v1")

    assert _commit(tmp_path, "f.txt", "-m", "again").returncode == 1
    assert _versions(tmp_path, "f.txt") == ["1.0"]


def test_allow_empty_commits_it_anyway(tmp_path):
    """git keeps `--allow-empty` because empty commits have uses; so does this."""
    (tmp_path / "f.txt").write_text("a\n")
    _commit(tmp_path, "f.txt", "-m", "v1")

    result = _commit(tmp_path, "f.txt", "-m", "again", "--allow-empty")

    assert result.returncode == 0, result.stdout
    assert _versions(tmp_path, "f.txt") == ["1.1", "1.0"]


def test_a_changed_file_still_commits(tmp_path):
    (tmp_path / "f.txt").write_text("a\n")
    _commit(tmp_path, "f.txt", "-m", "v1")
    (tmp_path / "f.txt").write_text("b\n")

    assert _commit(tmp_path, "f.txt", "-m", "v2").returncode == 0
    assert _versions(tmp_path, "f.txt") == ["1.1", "1.0"]


def test_an_unchanged_binary_file_is_refused(tmp_path):
    (tmp_path / "f.bin").write_bytes(bytes(range(256)))
    _commit(tmp_path, "f.bin", "-m", "v1")

    assert _commit(tmp_path, "f.bin", "-m", "again").returncode == 1
    assert _versions(tmp_path, "f.bin") == ["1.0"]
