"""
Stream adapters for SimpleRCS.

PsycopgLargeObjectAdapter wraps a psycopg2 lobject as a BinaryIO-compatible
stream so it can be passed directly to SimpleRCS() as the backing store.

Usage:
    import psycopg2
    conn = psycopg2.connect(DSN)
    lobj = conn.lobject(oid, mode='rwb')   # 0 to create new
    rcs  = SimpleRCS(PsycopgLargeObjectAdapter(lobj))
    rcs.commit(b"...", log="v1")
    conn.commit()
"""

from __future__ import annotations

import io


class PsycopgLargeObjectAdapter(io.RawIOBase):
    """
    Wraps a psycopg2 lobject as a seekable BinaryIO stream.

    psycopg2's lobject supports read/write/seek/tell/truncate but does not
    inherit from io.IOBase, so SimpleRCS (which expects BinaryIO) cannot use it
    directly.  This adapter bridges the gap with zero extra buffering.

    The adapter does NOT close the underlying database connection or
    transaction — callers must commit/rollback as needed.
    """

    def __init__(self, lobj) -> None:
        self._lobj = lobj

    # --- io.RawIOBase interface ---

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, b: bytearray | memoryview) -> int:
        data = self._lobj.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            return self._lobj.read()
        return self._lobj.read(size)

    def write(self, data: bytes | bytearray) -> int:
        return self._lobj.write(bytes(data))

    def seek(self, pos: int, whence: int = 0) -> int:
        return self._lobj.seek(pos, whence)

    def tell(self) -> int:
        return self._lobj.tell()

    def truncate(self, size: int | None = None) -> int:
        if size is None:
            size = self._lobj.tell()
        self._lobj.truncate(size)
        return size

    def close(self) -> None:
        if not self.closed:
            self._lobj.close()
        super().close()
