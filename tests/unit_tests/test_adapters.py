"""
Tests for PsycopgLargeObjectAdapter using a fake lobject that mimics
psycopg2's lobject read/write/seek/tell/truncate/close interface, so these
tests don't require a real PostgreSQL connection.
"""

from simple_rcs.adapters import PsycopgLargeObjectAdapter
from simple_rcs.simple_rcs import SimpleRCS


class FakeLobject:
    """Minimal stand-in for psycopg2's lobject, backed by a bytearray."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pos = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size == -1 or size is None:
            data = bytes(self._buf[self._pos :])
            self._pos = len(self._buf)
        else:
            data = bytes(self._buf[self._pos : self._pos + size])
            self._pos += len(data)
        return data

    def write(self, data: bytes) -> int:
        end = self._pos + len(data)
        if end > len(self._buf):
            self._buf.extend(b"\x00" * (end - len(self._buf)))
        self._buf[self._pos : end] = data
        self._pos = end
        return len(data)

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = len(self._buf) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def truncate(self, size: int) -> None:
        del self._buf[size:]
        if self._pos > size:
            self._pos = size

    def close(self) -> None:
        self.closed = True


def test_adapter_reports_capabilities():
    adapter = PsycopgLargeObjectAdapter(FakeLobject())
    assert adapter.readable() is True
    assert adapter.writable() is True
    assert adapter.seekable() is True


def test_adapter_read_write_seek_roundtrip():
    lobj = FakeLobject()
    adapter = PsycopgLargeObjectAdapter(lobj)

    adapter.write(b"hello world")
    assert adapter.tell() == 11

    adapter.seek(0)
    assert adapter.read() == b"hello world"

    adapter.seek(6)
    assert adapter.read(5) == b"world"


def test_adapter_readinto():
    lobj = FakeLobject()
    adapter = PsycopgLargeObjectAdapter(lobj)
    adapter.write(b"abcdef")
    adapter.seek(0)

    buf = bytearray(4)
    n = adapter.readinto(buf)
    assert n == 4
    assert bytes(buf) == b"abcd"


def test_adapter_truncate():
    lobj = FakeLobject()
    adapter = PsycopgLargeObjectAdapter(lobj)
    adapter.write(b"abcdefgh")
    adapter.truncate(3)
    adapter.seek(0)
    assert adapter.read() == b"abc"


def test_adapter_close_delegates_and_is_idempotent():
    lobj = FakeLobject()
    adapter = PsycopgLargeObjectAdapter(lobj)
    adapter.close()
    assert lobj.closed is True
    adapter.close()  # must not raise or double-close the underlying lobject


def test_simple_rcs_over_adapter_commit_and_checkout():
    lobj = FakeLobject()
    adapter = PsycopgLargeObjectAdapter(lobj)

    rcs = SimpleRCS(adapter)
    rcs.commit("Line 1\nLine 2\n", author="a", log="v1")
    rcs.commit("Line 1\nLine 2 modified\n", author="a", log="v2")

    assert rcs.checkout("1.1") == "Line 1\nLine 2 modified\n"
    assert rcs.checkout("1.0") == "Line 1\nLine 2\n"
    assert rcs.verify() is True

    # Re-open a fresh SimpleRCS over the same backing lobject: data must
    # have actually been persisted through the adapter, not just cached.
    lobj.seek(0)
    reopened = SimpleRCS(PsycopgLargeObjectAdapter(lobj))
    assert reopened.checkout("1.0") == "Line 1\nLine 2\n"
    assert reopened.checkout("1.1") == "Line 1\nLine 2 modified\n"
