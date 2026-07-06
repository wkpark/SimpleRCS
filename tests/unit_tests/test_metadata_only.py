"""
Tests for the metadata_only parsing path in SimpleRCS.

Covers:
  - log() with binary versions — metadata fields correct, no content decoded
  - log() then checkout() — _head_meta_only cache bypass works
  - _load_head(metadata_only=True) — lazy fields (content_stream_offset etc.) present
  - _get_prev_block(metadata_only=True) — lazy fields present in prev block
  - log(limit=N) / log(reverse=True) — work correctly through metadata_only path
  - base85 binary delta — @@ escaping handled correctly in scan path
  - log() → commit() → checkout() — cache coherence across write
  - @ sign in metadata fields — @@ unescaping in _parse_block_meta_from_stream
  - v2 hash/prev_hash fields — returned correctly through metadata_only path
  - consecutive log() calls — second call reuses meta cache
  - single-version log() — no prev block traversal
  - _fill() EOF guard — block_end > actual stream length terminates cleanly
"""

import io
import struct
import threading

from simple_rcs.simple_rcs import SimpleRCS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rcs() -> SimpleRCS:
    return SimpleRCS(None)


def _make_binary(seed: int, size: int = 16384) -> bytes:
    v = seed
    buf = bytearray()
    while len(buf) < size:
        v = (v * 1664525 + 1013904223) & 0xFFFFFFFF
        buf += struct.pack(">I", v)
    return bytes(buf[:size])


# ---------------------------------------------------------------------------
# log() with binary content
# ---------------------------------------------------------------------------


def test_log_binary_metadata_fields():
    """log() on binary-only history returns correct ver/author/log fields."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(1), author="alice", log="initial binary")
    rcs.commit(_make_binary(2), author="bob", log="updated binary")
    rcs.commit(_make_binary(3), author="carol", log="third binary")

    history = rcs.log()

    assert len(history) == 3
    assert history[0]["ver"] == "1.2"
    assert history[0]["author"] == "carol"
    assert history[0]["log"] == "third binary"
    assert history[1]["ver"] == "1.1"
    assert history[1]["author"] == "bob"
    assert history[1]["log"] == "updated binary"
    assert history[2]["ver"] == "1.0"
    assert history[2]["author"] == "alice"
    assert history[2]["log"] == "initial binary"


def test_log_binary_no_text_in_entries():
    """log() entries must not contain decoded binary content ('text' key absent)."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(10), log="v1")
    rcs.commit(_make_binary(11), log="v2")

    history = rcs.log()
    for entry in history:
        assert "text" not in entry, "log() entries must not carry decoded content"


# ---------------------------------------------------------------------------
# Critical: log() then checkout() — cache bypass
# ---------------------------------------------------------------------------


def test_log_then_checkout_binary():
    """
    log() sets _head_meta_only=True; subsequent checkout() must bypass the
    meta-only cache and re-load full content.
    """
    rcs = _make_rcs()
    v1 = _make_binary(20)
    v2 = _make_binary(21)

    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2")

    # Warm the metadata-only cache
    logs = rcs.log()
    assert len(logs) == 2

    # checkout() must still return correct content despite metadata-only cache
    assert rcs.checkout("1.0") == v1, "v1 content wrong after log() primed meta cache"
    assert rcs.checkout("1.1") == v2, "v2 content wrong after log() primed meta cache"


def test_log_then_checkout_text():
    """Same cache bypass test for text content."""
    rcs = _make_rcs()
    rcs.commit("hello world\n", author="a", log="v1")
    rcs.commit("hello updated\n", author="b", log="v2")

    rcs.log()  # warms meta-only cache

    assert rcs.checkout("1.0") == "hello world\n"
    assert rcs.checkout("1.1") == "hello updated\n"


def test_checkout_then_log_then_checkout():
    """Full → meta → full sequence must not corrupt cached content."""
    rcs = _make_rcs()
    v1 = _make_binary(30)
    v2 = _make_binary(31)
    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2")

    # Full load first
    assert rcs.checkout("1.1") == v2

    # Meta-only via log()
    history = rcs.log()
    assert history[0]["ver"] == "1.1"

    # Full load again — must still be correct
    assert rcs.checkout("1.0") == v1
    assert rcs.checkout("1.1") == v2


# ---------------------------------------------------------------------------
# _load_head(metadata_only=True) — lazy field presence
# ---------------------------------------------------------------------------


def test_load_head_metadata_only_lazy_fields_binary():
    """After _load_head(metadata_only=True), head_info has lazy content fields."""
    rcs = _make_rcs()
    payload = _make_binary(42)
    rcs.commit(payload, log="v1")

    # Force metadata-only re-load
    rcs.head_info = None
    rcs._head_cache_size = -1
    rcs._load_head(metadata_only=True)

    hi = rcs.head_info
    assert hi is not None
    assert hi.get("ver") == "1.0"
    assert hi.get("is_binary") is True
    assert "content_stream_offset" in hi, "lazy offset missing"
    assert "content_length" in hi, "lazy length missing"
    assert "content_encoding" in hi, "lazy encoding missing"
    assert "text" not in hi, "text must not be decoded in metadata_only mode"


def test_load_head_metadata_only_meta_fields_present():
    """ver/date/author/log are populated in metadata_only mode."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(43), author="tester", log="meta test")

    rcs.head_info = None
    rcs._head_cache_size = -1
    rcs._load_head(metadata_only=True)

    hi = rcs.head_info
    assert hi["ver"] == "1.0"
    assert hi["author"] == "tester"
    assert hi["log"] == "meta test"


# ---------------------------------------------------------------------------
# _get_prev_block(metadata_only=True) — lazy fields in prev block
# ---------------------------------------------------------------------------


def test_get_prev_block_metadata_only_lazy_fields():
    """
    _get_prev_block(metadata_only=True) returns meta fields without decoded content.

    The prev block is stored as a binary delta (not a binary snapshot), so it has
    is_binary=True and is_delta=True but no content_stream_offset — that field is
    only set for binary *snapshot* blocks (i.e. the binary key, not delta key).
    """
    rcs = _make_rcs()
    rcs.commit(_make_binary(50), log="v1")
    rcs.commit(_make_binary(51), log="v2")

    rcs._load_head()  # full load to get head_info with start offset
    head_start = rcs.head_info["start"]

    prev = rcs._get_prev_block(head_start, metadata_only=True)
    assert prev is not None
    assert prev.get("ver") == "1.0"
    assert prev.get("is_binary") is True
    assert prev.get("is_delta") is True
    assert "text" not in prev


def test_get_prev_block_metadata_only_vs_full_meta_match():
    """Metadata fields returned by metadata_only=True match those from full parse."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(60), author="orig", log="original")
    rcs.commit(_make_binary(61), log="head")

    rcs._load_head()
    head_start = rcs.head_info["start"]

    full = rcs._get_prev_block(head_start, metadata_only=False)
    meta = rcs._get_prev_block(head_start, metadata_only=True)

    assert full is not None and meta is not None
    for key in ("ver", "author", "log", "is_binary", "is_delta"):
        assert full.get(key) == meta.get(key), f"mismatch on key '{key}'"


# ---------------------------------------------------------------------------
# log(limit=N) and log(reverse=True) through metadata_only path
# ---------------------------------------------------------------------------


def test_log_limit_binary():
    """log(limit=2) returns at most 2 entries even with more history."""
    rcs = _make_rcs()
    for i in range(5):
        rcs.commit(_make_binary(i, size=4096), log=f"v{i}")

    history = rcs.log(limit=2)
    assert len(history) == 2
    assert history[0]["ver"] == "1.4"
    assert history[1]["ver"] == "1.3"


def test_log_reverse_binary():
    """log(reverse=True) returns oldest-first through metadata_only path."""
    rcs = _make_rcs()
    for i in range(3):
        rcs.commit(_make_binary(i + 100, size=4096), log=f"rev{i}")

    history = rcs.log(reverse=True)
    assert len(history) == 3
    assert history[0]["ver"] == "1.0"
    assert history[1]["ver"] == "1.1"
    assert history[2]["ver"] == "1.2"


# ---------------------------------------------------------------------------
# Mixed text + binary history
# ---------------------------------------------------------------------------


def test_log_mixed_text_binary_history():
    """log() traverses correctly across text→binary type change boundary."""
    rcs = _make_rcs()
    rcs.commit("text content\n", author="a", log="text v1")
    rcs.commit(_make_binary(200), author="b", log="binary v2")

    history = rcs.log()
    assert len(history) == 2
    assert history[0]["log"] == "binary v2"
    assert history[1]["log"] == "text v1"

    # checkout must still work
    assert rcs.checkout("1.0") == "text content\n"
    assert rcs.checkout("1.1") == _make_binary(200)


# ---------------------------------------------------------------------------
# base85 binary delta — @@ escaping in scan path
# ---------------------------------------------------------------------------


def test_base85_binary_delta_metadata_only_scan():
    """
    base85 charset contains '@', so stored delta has @@ escaping.
    metadata_only scan (not seek skip) must handle @@ pairs correctly.
    """
    rcs = _make_rcs()
    v1 = _make_binary(1001)
    v2 = _make_binary(1002)

    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2", encoding="base85")  # old HEAD stored as base85 delta

    history = rcs.log()
    assert len(history) == 2
    assert history[0]["ver"] == "1.1"
    assert history[0]["log"] == "v2"
    assert history[1]["ver"] == "1.0"
    assert history[1]["log"] == "v1"

    # After log() primed meta cache, both checkouts must still be correct
    assert rcs.checkout("1.0") == v1, "v1 content wrong after base85 log() meta scan"
    assert rcs.checkout("1.1") == v2, "v2 content wrong after base85 log() meta scan"


def test_base85_head_and_delta_metadata_only():
    """HEAD stored as base85 binary (seek skip), delta as base85 (scan with @@)."""
    rcs = _make_rcs()
    v1 = _make_binary(1010)
    v2 = _make_binary(1011)
    v3 = _make_binary(1012)

    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2", encoding="base85")
    rcs.commit(v3, log="v3", encoding="base85")  # HEAD = base85 binary

    history = rcs.log()
    assert len(history) == 3
    assert [h["ver"] for h in history] == ["1.2", "1.1", "1.0"]

    assert rcs.checkout("1.0") == v1
    assert rcs.checkout("1.1") == v2
    assert rcs.checkout("1.2") == v3


# ---------------------------------------------------------------------------
# commit() after log() — cache coherence across write
# ---------------------------------------------------------------------------


def test_log_then_commit_then_checkout():
    """
    log() primes meta-only cache; commit() must bypass it to load full HEAD
    content before delta computation. All three versions retrievable after.
    """
    rcs = _make_rcs()
    v1 = _make_binary(2001)
    v2 = _make_binary(2002)
    v3 = _make_binary(2003)

    rcs.commit(v1, log="v1")
    rcs.commit(v2, log="v2")

    # Prime meta-only cache
    assert len(rcs.log()) == 2
    assert rcs._head_meta_only is True

    # commit() must re-load full HEAD (not use meta cache) to compute delta correctly
    rcs.commit(v3, log="v3")

    assert rcs.checkout("1.0") == v1, "v1 wrong after log→commit sequence"
    assert rcs.checkout("1.1") == v2, "v2 wrong after log→commit sequence"
    assert rcs.checkout("1.2") == v3, "v3 wrong after log→commit sequence"


# ---------------------------------------------------------------------------
# @ sign in metadata fields — @@ unescaping
# ---------------------------------------------------------------------------


def test_log_at_sign_in_metadata_fields():
    """'@' in author/log stored as '@@' and correctly unescaped in metadata_only parse."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(3001), author="user@domain.com", log="fix: report@office")
    rcs.commit(_make_binary(3002), author="dev@co.io", log="update: data@v2")

    history = rcs.log()
    assert history[0]["author"] == "dev@co.io"
    assert history[0]["log"] == "update: data@v2"
    assert history[1]["author"] == "user@domain.com"
    assert history[1]["log"] == "fix: report@office"

    # Must also survive the cache-bypass checkout
    assert rcs.checkout("1.0") == _make_binary(3001)


def test_log_at_sign_via_load_head_metadata_only():
    """_load_head(metadata_only=True) correctly unescapes @@ in all meta fields."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(3010), author="a@b", log="msg@here")

    rcs.head_info = None
    rcs._head_cache_size = -1
    rcs._load_head(metadata_only=True)

    hi = rcs.head_info
    assert hi["author"] == "a@b"
    assert hi["log"] == "msg@here"


# ---------------------------------------------------------------------------
# v2 hash / prev_hash fields in log() entries
# ---------------------------------------------------------------------------


def test_log_v2_hash_fields_binary():
    """hash and prev_hash fields are returned in log() entries via metadata_only path."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(4001), log="v1")
    rcs.commit(_make_binary(4002), log="v2")

    history = rcs.log()

    # HEAD (v2) has both hash and a prev_hash linking to v1
    assert history[0]["hash"] is not None and history[0]["hash"] != ""
    assert history[0]["prev_hash"] is not None and history[0]["prev_hash"] != ""

    # First commit (v1) has a hash but no prev_hash
    assert history[1]["hash"] is not None and history[1]["hash"] != ""
    assert history[1]["prev_hash"] is None

    # hash values are distinct
    assert history[0]["hash"] != history[1]["hash"]

    # prev_hash of v2 matches hash of v1
    assert history[0]["prev_hash"] == history[1]["hash"]


# ---------------------------------------------------------------------------
# Consecutive log() calls — cache reuse
# ---------------------------------------------------------------------------


def test_consecutive_log_calls_idempotent():
    """Second log() call returns identical results without re-scanning."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(5001), log="v1")
    rcs.commit(_make_binary(5002), log="v2")

    h1 = rcs.log()
    cache_size_after_first = rcs._head_cache_size

    h2 = rcs.log()

    assert h1 == h2
    # Cache size unchanged — no re-scan happened
    assert rcs._head_cache_size == cache_size_after_first
    assert rcs._head_meta_only is True


# ---------------------------------------------------------------------------
# Single-version log() — no prev block traversal
# ---------------------------------------------------------------------------


def test_log_single_binary_version():
    """log() on a single binary commit returns one entry with no prev traversal."""
    rcs = _make_rcs()
    rcs.commit(_make_binary(6001), author="solo", log="only commit")

    history = rcs.log()
    assert len(history) == 1
    assert history[0]["ver"] == "1.0"
    assert history[0]["author"] == "solo"
    assert history[0]["log"] == "only commit"


def test_log_empty_rcs():
    """log() on an empty (never committed) RCS returns an empty list."""
    rcs = _make_rcs()
    assert rcs.log() == []


# ---------------------------------------------------------------------------
# _fill() EOF guard — block_end > actual stream length
# ---------------------------------------------------------------------------


def test_fill_eof_guard_no_hang():
    """
    _parse_block_meta_from_stream must exit cleanly when block_end exceeds the
    actual stream length (simulating the premature-EOF scenario the guard defends).

    Without the 'if not chunk: stream_cursor = block_end' guard, _fill() would
    loop forever because stream_cursor never reaches block_end.
    """
    rcs = _make_rcs()
    rcs.commit(_make_binary(7001), log="v1")

    rcs.stream.seek(0)
    full = rcs.stream.read()

    # Replace stream with a fresh BytesIO so reads at EOF return b""
    rcs.stream = io.BytesIO(full)

    result_holder = {}
    done = threading.Event()

    def _run():
        # Pass block_end well beyond actual file length — forces _fill() to
        # attempt reads past EOF (returns b"") while stream_cursor < block_end.
        r = rcs._parse_block_meta_from_stream(0, len(full) + 65536)
        result_holder["result"] = r
        done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert done.wait(timeout=5.0), "_parse_block_meta_from_stream hung — _fill() EOF guard missing or broken"
    # Result may be complete or partial; must not be an infinite loop
    assert isinstance(result_holder.get("result"), dict)
