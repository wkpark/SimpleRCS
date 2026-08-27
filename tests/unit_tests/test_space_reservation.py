"""Tests for the pre-write space reservation added to the HEAD rewrite path.

commit() and sign_head() both rewrite HEAD *in place*: they seek back over the
live old HEAD, overwrite it, and truncate. There is no point in that sequence
where the file holds both the old and the new HEAD, so a write that runs out of
space halfway destroys the version it was replacing.

_reserve_space() moves that failure ahead of the first destructive byte via
posix_fallocate(), which either reserves the blocks or raises with the file
untouched. These tests drive the failure with RLIMIT_FSIZE, which
posix_fallocate() reports as EFBIG -- the same early-failure path ENOSPC takes.
"""

import hashlib
import io
import os
import resource
import signal
from contextlib import contextmanager

import pytest

from simple_rcs.simple_rcs import SimpleRCS

pytestmark = pytest.mark.skipif(
    not hasattr(os, "posix_fallocate"),
    reason="posix_fallocate() is POSIX-only; the reservation is a documented no-op elsewhere",
)


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


def test_commit_that_cannot_fit_leaves_the_file_byte_identical(tmp_path):
    path = tmp_path / "reserved.rcs"
    rcs = SimpleRCS(str(path))
    rcs.commit("first version\n", author="tester", log="v1")
    rcs.commit("second version\n", author="tester", log="v2")
    del rcs

    before = _digest(path)
    assert before[0] > 0

    # Room for a few more bytes, but nowhere near a whole new block.
    with _max_file_size(before[0] + 16):
        rcs = SimpleRCS(str(path))
        with pytest.raises(OSError):
            rcs.commit("a third version, much longer than the headroom\n" * 20, author="tester", log="v3")

    assert _digest(path) == before, "a rejected commit must not touch the file"

    # And the history is still readable, not merely intact on disk.
    rcs = SimpleRCS(str(path))
    assert rcs.checkout("1.0") == "first version\n"
    assert rcs.checkout("1.1") == "second version\n"
    assert [entry["ver"] for entry in rcs.log()] == ["1.1", "1.0"]


def test_reservation_is_skipped_for_streams_without_a_descriptor():
    # BytesIO has no fileno(); the reservation must no-op rather than raise,
    # so in-memory use keeps working unchanged.
    rcs = SimpleRCS(io.BytesIO())
    rcs.commit("in memory\n", author="tester", log="v1")
    rcs.commit("still in memory\n", author="tester", log="v2")
    assert rcs.checkout("1.0") == "in memory\n"
    assert [entry["ver"] for entry in rcs.log()] == ["1.1", "1.0"]


def test_reservation_ignores_filesystems_that_cannot_reserve(tmp_path, monkeypatch):
    # EOPNOTSUPP means "this filesystem cannot preallocate", not "the disk is
    # full" -- it must not be allowed to block a commit.
    path = tmp_path / "unsupported.rcs"
    rcs = SimpleRCS(str(path))
    rcs.commit("first version\n", author="tester", log="v1")

    def _refuse(fd, offset, length):
        raise OSError(getattr(os, "EOPNOTSUPP", 95), "Operation not supported")

    monkeypatch.setattr(os, "posix_fallocate", _refuse)
    rcs.commit("second version\n", author="tester", log="v2")

    assert rcs.checkout("1.0") == "first version\n"
    assert rcs.checkout("1.1") == "second version\n"


def test_reservation_covers_the_sign_head_rewrite(tmp_path):
    # sign_head() is the second in-place HEAD rewrite; it must go through the
    # same reservation, otherwise adding a signature can destroy HEAD.
    path = tmp_path / "signed.rcs"
    rcs = SimpleRCS(str(path))
    rcs.commit("first version\n", author="tester", log="v1")
    rcs.commit("second version\n", author="tester", log="v2")
    del rcs

    before = _digest(path)

    with _max_file_size(before[0] + 16):
        rcs = SimpleRCS(str(path))
        with pytest.raises(OSError):
            rcs.sign_head([lambda msg: ("signer", "x" * 4096)])

    assert _digest(path) == before, "a rejected sign_head must not touch the file"
