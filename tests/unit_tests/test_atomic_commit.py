"""Tests for installing a commit with a temp file and os.replace().

A file-path-backed SimpleRCS no longer overwrites its own HEAD. It copies
everything before HEAD into a temp file beside it, appends the new tail, and
renames over the original. The rename either happens or does not, so an
interrupted commit costs the commit and never the history -- which matters here
more than usual, since every older version is a reverse delta anchored on HEAD.

These tests pin the two halves of that claim: the result is byte-for-byte what
the in-place rewrite produced (no format change), and a failure anywhere before
the rename leaves the original untouched.
"""

import hashlib
import io
import os
import resource
import signal
import stat
import tempfile
from contextlib import contextmanager

import pytest

from simple_rcs.simple_rcs import SimpleRCS

FIXED_DATE = "2026-01-01T00:00:00"


@contextmanager
def _max_file_size(limit: int):
    """Cap file size for the enclosed block, without dying on SIGXFSZ."""
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if hard != resource.RLIM_INFINITY and limit > hard:
        pytest.skip("RLIMIT_FSIZE hard limit is already below the test's limit")
    previous_handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous_handler)


def _digest(path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def _temp_leftovers(directory) -> list[str]:
    return [name for name in os.listdir(directory) if name.endswith(".tmp")]


def _build(path, contents, in_place=False):
    """Commit each of `contents` in turn, at a fixed date so bytes are comparable."""
    rcs = SimpleRCS(str(path))
    if in_place:
        # Drop the path rather than swapping the method out: this is the exact
        # condition _rewrite_head dispatches on, so the two runs differ in the
        # strategy and nothing else. The stream is still the same real file.
        rcs.file_path = None
    for i, content in enumerate(contents):
        rcs.commit(content, author="tester", log=f"v{i}", date=FIXED_DATE)
    rcs.stream.flush()
    return rcs


VERSIONS = ["one\n", "one\ntwo\n", "one\ntwo\nthree\n", "one\nTWO\nthree\n"]


def test_the_atomic_path_writes_the_same_bytes_as_rewriting_in_place(tmp_path):
    """The point of the change is durability, not a new format."""
    atomic = tmp_path / "atomic.rcs"
    in_place = tmp_path / "in_place.rcs"
    _build(atomic, VERSIONS)
    _build(in_place, VERSIONS, in_place=True)

    assert atomic.read_bytes() == in_place.read_bytes()
    assert _temp_leftovers(tmp_path) == []


def test_history_reads_back_after_an_atomic_commit(tmp_path):
    path = tmp_path / "history.rcs"
    _build(path, VERSIONS)

    rcs = SimpleRCS(str(path))
    assert [block["ver"] for block in rcs.log()] == ["1.3", "1.2", "1.1", "1.0"]
    for i, expected in enumerate(VERSIONS):
        assert rcs.checkout(f"1.{i}") == expected
    assert rcs.verify() is True


def test_a_commit_that_runs_out_of_space_leaves_the_original_untouched(tmp_path):
    """The failure this whole change exists for: no space left, mid-write.

    In place this destroyed the old HEAD and with it every earlier version. Here
    it can only kill the temp file.
    """
    path = tmp_path / "full.rcs"
    rcs = _build(path, VERSIONS[:2])
    before = _digest(path)

    # Enough room for the file itself, not enough for a second copy of it.
    with _max_file_size(before[0] + 16), pytest.raises(OSError):
        rcs.commit("one\ntwo\nthree\n", author="tester", log="v2", date=FIXED_DATE)

    assert _digest(path) == before
    assert _temp_leftovers(tmp_path) == []

    reopened = SimpleRCS(str(path))
    assert reopened.checkout("1.0") == VERSIONS[0]
    assert reopened.checkout("1.1") == VERSIONS[1]


def test_a_failure_while_copying_the_history_leaves_the_original_untouched(tmp_path):
    """Same guarantee, but failing in the prefix copy rather than the tail write."""
    path = tmp_path / "copy_fail.rcs"
    rcs = _build(path, VERSIONS[:3])
    before = _digest(path)

    with _max_file_size(before[0] // 2), pytest.raises(OSError):
        rcs.commit("one\ntwo\nthree\nfour\n", author="tester", log="v3", date=FIXED_DATE)

    assert _digest(path) == before
    assert _temp_leftovers(tmp_path) == []


def test_copy_prefix_includes_writes_still_held_in_the_stream_buffer(tmp_path):
    """_copy_prefix reads the raw descriptor, which cannot see Python's buffer.

    commit() happens to flush before getting here (_load_head seeks to the end),
    so this is on the unit rather than on a commit: drop the flush and the copy
    comes up short of the bytes the caller can already see via tell().
    """
    path = tmp_path / "buffered.rcs"
    rcs = SimpleRCS(str(path))
    rcs.commit("hello\n", author="tester", log="v0", date=FIXED_DATE)

    rcs.stream.seek(0, os.SEEK_END)
    rcs.stream.write(b"\n\nnot yet flushed\n")
    size = rcs.stream.tell()

    fd, tmp_name = tempfile.mkstemp(dir=str(tmp_path))
    try:
        rcs._copy_prefix(fd, size)
        os.close(fd)
        assert os.path.getsize(tmp_name) == size
        assert b"not yet flushed" in open(tmp_name, "rb").read()
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def test_the_instance_still_works_after_its_file_is_replaced(tmp_path):
    """os.replace installs a new inode; the old handle points at the unlinked one.

    Skip the reopen and the next commit writes into a file nobody can see.
    """
    path = tmp_path / "reuse.rcs"
    rcs = SimpleRCS(str(path))
    for i, content in enumerate(VERSIONS):
        rcs.commit(content, author="tester", log=f"v{i}", date=FIXED_DATE)

    # Same instance across all four commits: reads still work...
    assert rcs.checkout("1.0") == VERSIONS[0]
    assert rcs.verify() is True
    # ...and so does what is actually on disk.
    assert SimpleRCS(str(path)).checkout("1.3") == VERSIONS[3]


def test_head_is_reloaded_even_when_the_rewrite_does_not_change_the_size(tmp_path):
    """_load_head caches on file size, so a same-size rewrite must force a reload.

    Ordinary commits change the size, which is why this drives _atomic_rewrite_head
    directly with a payload that is the current HEAD block with one field swapped
    for a different value of the same length.
    """
    path = tmp_path / "same_size.rcs"
    rcs = _build(path, VERSIONS[:2])
    size_before = os.path.getsize(path)

    head = rcs.head_info
    assert head["log"] == "v1"
    replacement = dict(head)
    replacement["log"] = "vX"  # same length, different value
    payload = rcs._format_block(
        replacement,
        current_hash=head.get("hash"),
        prev_hash=head.get("prev_hash"),
        signatures=head.get("signatures"),
        is_delta=False,
    )
    assert len(payload) == size_before - head["start"], "payload must keep the file size identical"

    rcs._atomic_rewrite_head(payload, str(path), head["start"])

    assert os.path.getsize(path) == size_before
    assert rcs.head_info["log"] == "vX"


def test_the_replacement_file_keeps_the_original_permissions(tmp_path):
    """os.replace installs a new inode, and mkstemp's is 0600."""
    path = tmp_path / "modes.rcs"
    rcs = _build(path, VERSIONS[:1])
    os.chmod(path, 0o640)

    rcs.commit(VERSIONS[1], author="tester", log="v1", date=FIXED_DATE)

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640


def test_sign_head_takes_the_atomic_path_too(tmp_path):
    path = tmp_path / "signed.rcs"
    rcs = _build(path, VERSIONS[:2])

    assert rcs.sign_head([lambda message: ("tester", f"sig:{message}")]) is True
    assert _temp_leftovers(tmp_path) == []

    reopened = SimpleRCS(str(path))
    assert reopened.head_info["signatures"]
    assert reopened.checkout("1.0") == VERSIONS[0]


def test_durable_false_skips_the_fsyncs(tmp_path, monkeypatch):
    """The escape hatch for bulk work. It must still produce the same file."""
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))

    path = tmp_path / "fast.rcs"
    rcs = SimpleRCS(str(path), durable=False)
    for i, content in enumerate(VERSIONS[:2]):
        rcs.commit(content, author="tester", log=f"v{i}", date=FIXED_DATE)
    assert calls == []

    durable_path = tmp_path / "slow.rcs"
    durable = SimpleRCS(str(durable_path), durable=True)
    for i, content in enumerate(VERSIONS[:2]):
        durable.commit(content, author="tester", log=f"v{i}", date=FIXED_DATE)
    assert calls, "durable=True must fsync"

    monkeypatch.setattr(os, "fsync", real_fsync)
    assert path.read_bytes() == durable_path.read_bytes()


def test_streams_without_a_path_keep_rewriting_in_place(tmp_path, monkeypatch):
    """There is nothing to rename over, so the old path has to stay reachable."""
    taken = []
    monkeypatch.setattr(
        SimpleRCS,
        "_atomic_rewrite_head",
        lambda self, payload, file_path, start: taken.append(payload),
    )

    memory = SimpleRCS(io.BytesIO())
    memory.commit("first\n", author="tester", log="v0", date=FIXED_DATE)
    memory.commit("second\n", author="tester", log="v1", date=FIXED_DATE)
    assert taken == []
    assert memory.checkout("1.0") == "first\n"

    handle_path = tmp_path / "caller.rcs"
    handle_path.write_bytes(b"")
    with open(handle_path, "rb+") as handle:
        caller_owned = SimpleRCS(handle)
        caller_owned.commit("first\n", author="tester", log="v0", date=FIXED_DATE)
        caller_owned.commit("second\n", author="tester", log="v1", date=FIXED_DATE)
        assert taken == []
        assert caller_owned.checkout("1.0") == "first\n"
