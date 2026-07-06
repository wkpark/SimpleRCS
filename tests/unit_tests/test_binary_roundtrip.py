"""
Binary diff/patch roundtrip tests for SimpleRCS.

Covers:
  - commit(bytes) → checkout() byte-for-byte equality
  - Multi-version binary delta chain (v1→v2→v3, checkout all)
  - In-place byte modification (no shift)
  - Insertion in the middle (shifts all subsequent chunk boundaries)
  - Deletion
  - Text→Binary type change (snapshot boundary)
  - Large real file (PDF) — skipped if file is absent
  - base85 encoding variant
"""

import hashlib
import os
import struct

import pytest

from simple_rcs.simple_rcs import SimpleRCS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_PATH = os.path.join(REPO_ROOT, "test_orig.pdf")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_rcs() -> SimpleRCS:
    """In-memory SimpleRCS instance (fast, no disk I/O)."""
    return SimpleRCS(None)


def _make_binary(size: int, seed: int = 0) -> bytes:
    """Deterministic pseudo-random bytes."""
    rng = bytearray()
    v = seed
    while len(rng) < size:
        v = (v * 1664525 + 1013904223) & 0xFFFFFFFF
        rng += struct.pack(">I", v)
    return bytes(rng[:size])


def _modify_inplace(data: bytes, offset: int, length: int = 20) -> bytes:
    """Replace `length` bytes at `offset` with inverted bytes."""
    ba = bytearray(data)
    for i in range(length):
        ba[offset + i] = (~ba[offset + i]) & 0xFF
    return bytes(ba)


def _modify_insert(data: bytes, offset: int, insert: bytes) -> bytes:
    return data[:offset] + insert + data[offset:]


def _modify_delete(data: bytes, offset: int, length: int) -> bytes:
    return data[:offset] + data[offset + length :]


# ---------------------------------------------------------------------------
# Basic roundtrip
# ---------------------------------------------------------------------------


def test_binary_basic_commit_checkout():
    """commit(bytes) → checkout() returns identical bytes."""
    rcs = _make_rcs()
    payload = _make_binary(8192, seed=1)

    rcs.commit(payload, author="tester", log="binary v1")

    result = rcs.checkout("1.0")
    assert isinstance(result, bytes)
    assert result == payload, "checkout must return exact original bytes"


def test_binary_two_version_roundtrip():
    """v1(bytes) → v2(modified bytes) → checkout v1 must match original."""
    rcs = _make_rcs()
    v1_data = _make_binary(16384, seed=2)
    v2_data = _modify_inplace(v1_data, offset=1000, length=32)

    assert v1_data != v2_data

    rcs.commit(v1_data, log="v1")
    rcs.commit(v2_data, log="v2")

    assert rcs.checkout("1.0") == v1_data, "v1 roundtrip failed"
    assert rcs.checkout("1.1") == v2_data, "v2 roundtrip failed"


def test_binary_three_version_chain():
    """Three-version delta chain: checkout every version."""
    rcs = _make_rcs()
    base = _make_binary(32768, seed=3)
    v1 = base
    v2 = _modify_inplace(base, offset=100, length=50)
    v3 = _modify_inplace(base, offset=16000, length=50)

    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2")
    rcs.commit(v3, log="v3")

    assert rcs.checkout("1.0") == v1, "v1 checkout failed"
    assert rcs.checkout("1.1") == v2, "v2 checkout failed"
    assert rcs.checkout("1.2") == v3, "v3 HEAD checkout failed"


# ---------------------------------------------------------------------------
# Structural change scenarios
# ---------------------------------------------------------------------------


def test_binary_inplace_modification():
    """Single-byte modification in the middle — chunk boundaries unchanged."""
    rcs = _make_rcs()
    orig = _make_binary(65536, seed=10)
    modified = bytearray(orig)
    modified[32768] ^= 0xFF  # flip one byte
    modified = bytes(modified)

    rcs.commit(orig, log="original")
    rcs.commit(modified, log="flip one byte")

    assert rcs.checkout("1.0") == orig, "original not restored after single-byte flip"
    assert rcs.checkout("1.1") == modified, "modified HEAD mismatch"


def test_binary_insertion_shifts_chunks():
    """Insert 200 bytes in the middle — all subsequent chunk boundaries shift."""
    rcs = _make_rcs()
    orig = _make_binary(65536, seed=20)
    inserted = _modify_insert(orig, offset=len(orig) // 2, insert=b"\xde\xad\xbe\xef" * 50)

    assert len(inserted) == len(orig) + 200

    rcs.commit(orig, log="original")
    rcs.commit(inserted, log="insert 200 bytes at midpoint")

    assert rcs.checkout("1.0") == orig, "original not restored after insertion"
    assert rcs.checkout("1.1") == inserted, "inserted version HEAD mismatch"


def test_binary_deletion():
    """Delete 512 bytes — subsequent data shifts up."""
    rcs = _make_rcs()
    orig = _make_binary(65536, seed=30)
    deleted = _modify_delete(orig, offset=4096, length=512)

    assert len(deleted) == len(orig) - 512

    rcs.commit(orig, log="original")
    rcs.commit(deleted, log="delete 512 bytes")

    assert rcs.checkout("1.0") == orig, "original not restored after deletion"
    assert rcs.checkout("1.1") == deleted, "deleted version HEAD mismatch"


def test_binary_multiple_scattered_edits():
    """Several small edits at different offsets — stress-tests delta accuracy."""
    rcs = _make_rcs()
    orig = _make_binary(131072, seed=40)  # 128 KB

    v2 = bytearray(orig)
    for offset in [1000, 20000, 60000, 100000]:
        v2[offset : offset + 16] = b"\xab\xcd" * 8
    v2 = bytes(v2)

    rcs.commit(orig, log="v1")
    rcs.commit(v2, log="v2 scattered edits")

    assert rcs.checkout("1.0") == orig, "v1 not restored"
    assert rcs.checkout("1.1") == v2, "v2 mismatch"


# ---------------------------------------------------------------------------
# Encoding variants
# ---------------------------------------------------------------------------


def test_binary_base85_encoding():
    """Verify roundtrip with base85 encoding (more compact than base64)."""
    rcs = _make_rcs()
    orig = _make_binary(16384, seed=50)
    mod = _modify_inplace(orig, offset=8000, length=32)

    rcs.commit(orig, log="v1")
    rcs.commit(mod, log="v2", encoding="base85")

    assert rcs.checkout("1.0") == orig, "base85: v1 roundtrip failed"
    assert rcs.checkout("1.1") == mod, "base85: v2 roundtrip failed"


# ---------------------------------------------------------------------------
# Type change boundary (text → binary)
# ---------------------------------------------------------------------------


def test_text_then_binary_type_change():
    """Switching from text to binary forces a snapshot — both must be retrievable."""
    rcs = _make_rcs()
    text_content = "Hello, world!\nSecond line.\n"
    binary_content = _make_binary(4096, seed=60)

    rcs.commit(text_content, log="text version")
    rcs.commit(binary_content, log="binary version")  # type change → snapshot

    result_text = rcs.checkout("1.0")
    assert isinstance(result_text, str)
    assert result_text == text_content

    result_bin = rcs.checkout("1.1")
    assert isinstance(result_bin, bytes)
    assert result_bin == binary_content


# ---------------------------------------------------------------------------
# Snapshot in binary chain
# ---------------------------------------------------------------------------


def test_binary_snapshot_chain():
    """Snapshot mid-chain: checkout across snapshot boundary must be correct."""
    rcs = _make_rcs()
    v1 = _make_binary(32768, seed=70)
    v2 = _modify_inplace(v1, offset=100, length=32)
    v3 = _modify_inplace(v2, offset=16000, length=32)
    v4 = _modify_inplace(v3, offset=30000, length=32)

    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2")
    rcs.commit(v3, log="v3", snapshot=True)  # v2 saved as full-text snapshot
    rcs.commit(v4, log="v4")

    assert rcs.checkout("1.0") == v1, "v1 across snapshot boundary"
    assert rcs.checkout("1.1") == v2, "v2 (snapshot block) checkout"
    assert rcs.checkout("1.2") == v3, "v3 after snapshot"
    assert rcs.checkout("1.3") == v4, "v4 HEAD"


# ---------------------------------------------------------------------------
# File-based (disk I/O)
# ---------------------------------------------------------------------------


def test_binary_file_based_roundtrip(tmp_path):
    """SimpleRCS backed by a real on-disk file — binary commit/checkout."""
    rcs_path = str(tmp_path / "binary.rcs")
    orig = _make_binary(65536, seed=80)
    mod = _modify_inplace(orig, offset=32000, length=64)

    rcs = SimpleRCS(rcs_path)
    rcs.commit(orig, log="v1")
    rcs.commit(mod, log="v2")

    # Re-open from disk
    rcs2 = SimpleRCS(rcs_path)
    assert rcs2.checkout("1.0") == orig, "disk-based v1 roundtrip failed"
    assert rcs2.checkout("1.1") == mod, "disk-based v2 roundtrip failed"


# ---------------------------------------------------------------------------
# Large real binary file (PDF)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(PDF_PATH), reason=f"test file not found: {PDF_PATH}")
def test_pdf_inplace_modification_roundtrip():
    """Real PDF: in-place byte modification → commit → checkout → byte-for-byte equal."""
    with open(PDF_PATH, "rb") as f:
        orig = f.read()

    size = len(orig)
    # Flip 20 bytes at 1/3 and 2/3 through the file
    mod = bytearray(orig)
    for off in [size // 3, size * 2 // 3]:
        for i in range(10):
            mod[off + i] = (~mod[off + i]) & 0xFF
    mod = bytes(mod)

    assert mod != orig
    assert len(mod) == len(orig)

    rcs = _make_rcs()
    rcs.commit(orig, log="original pdf")
    rcs.commit(mod, log="modified pdf")

    restored_v1 = rcs.checkout("1.0")
    assert isinstance(restored_v1, bytes)
    assert len(restored_v1) == len(orig), "size mismatch after roundtrip"
    assert _sha256(restored_v1) == _sha256(orig), "SHA-256 mismatch: v1 not restored exactly"

    restored_v2 = rcs.checkout("1.1")
    assert _sha256(restored_v2) == _sha256(mod), "SHA-256 mismatch: v2 (HEAD) corrupted"


@pytest.mark.skipif(not os.path.exists(PDF_PATH), reason=f"test file not found: {PDF_PATH}")
def test_pdf_insertion_roundtrip():
    """Real PDF: insert 100 bytes at midpoint → checkout both versions."""
    with open(PDF_PATH, "rb") as f:
        orig = f.read()

    mid = len(orig) // 2
    inserted = orig[:mid] + b"\x00" * 100 + orig[mid:]

    rcs = _make_rcs()
    rcs.commit(orig, log="original pdf")
    rcs.commit(inserted, log="100-byte insert at midpoint")

    assert rcs.checkout("1.0") == orig, "PDF v1 not restored after insertion"
    assert rcs.checkout("1.1") == inserted, "PDF inserted version mismatch"


@pytest.mark.skipif(not os.path.exists(PDF_PATH), reason=f"test file not found: {PDF_PATH}")
def test_pdf_three_version_chain():
    """Real PDF: three-version delta chain — SHA-256 verified at every version."""
    with open(PDF_PATH, "rb") as f:
        orig = f.read()

    size = len(orig)
    v1 = orig
    v2 = _modify_inplace(orig, offset=size // 4, length=50)
    v3 = _modify_insert(v2, offset=size // 2, insert=b"\xff" * 64)

    rcs = _make_rcs()
    rcs.commit(v1, log="pdf v1")
    rcs.commit(v2, log="pdf v2 modified")
    rcs.commit(v3, log="pdf v3 with insert")

    assert _sha256(rcs.checkout("1.0")) == _sha256(v1), "pdf v1 SHA-256 mismatch"
    assert _sha256(rcs.checkout("1.1")) == _sha256(v2), "pdf v2 SHA-256 mismatch"
    assert _sha256(rcs.checkout("1.2")) == _sha256(v3), "pdf v3 SHA-256 mismatch"
