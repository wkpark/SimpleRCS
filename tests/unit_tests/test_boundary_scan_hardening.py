"""Regression tests for issues an independent code-review pass found in the
false-positive HEAD-scan fix (see test_head_scan_false_positive.py) itself.

1. _parse_block_meta_from_stream (the metadata_only path used by log()/
   blame()) was stricter than _parse_block_content_no_regex for meta fields
   (ver/date/author/log/...) after that fix, but not yet for binary/delta/
   text fields -- those still tolerated a missing terminating ';' and could
   "succeed" with corrupted data where the content parser correctly failed.

2. commit()'s first-commit append didn't ensure the new block satisfies the
   boundary invariant _is_block_boundary now requires when scanning for it
   (immediately preceded by a blank line, or at self._data_start). If the
   stream had non-block trailing bytes despite head_info being None, the
   freshly committed block became permanently unfindable to a later scan.

3. The retry loop added by the false-positive fix re-assembled the full
   block-to-EOF byte range on every rejected candidate, making a scan
   super-linear (trending toward O(n^2)) on content with many "blank line +
   marker" collisions -- exactly the kind of content that fix targets (e.g.
   changelog/wiki prose with several "version @N" mentions).
"""

import io
import time

from simple_rcs.simple_rcs import SimpleRCS


def test_meta_parser_matches_content_parser_on_malformed_text_field():
    # A "text" field missing its terminating ';', immediately followed by a
    # well-formed-looking "date" field. The content parser has always
    # stopped cold here (no 'text', no 'date' in its output); the metadata
    # parser used to slide past the missing ';' and read "bogus-trailing-
    # date" as a real date, disagreeing with the content parser on the same
    # bytes.
    rcs = SimpleRCS()
    block = b"ver @1.0@;\ntext @hello world@date @bogus-trailing-date@;\n\n"

    content_parsed = rcs._parse_block_content_no_regex(block)
    assert content_parsed == {"ver": "1.0", "is_delta": False, "is_binary": False}

    rcs.stream = io.BytesIO(block)
    meta_parsed = rcs._parse_block_meta_from_stream(0, len(block))
    assert meta_parsed == content_parsed


def test_meta_parser_matches_content_parser_on_malformed_delta_field():
    rcs = SimpleRCS()
    block = b"ver @1.1@;\ndelta @d1 1\\na1 1\\nnew@date @bogus-trailing-date@;\n\n"

    content_parsed = rcs._parse_block_content_no_regex(block)

    rcs.stream = io.BytesIO(block)
    meta_parsed = rcs._parse_block_meta_from_stream(0, len(block))

    assert "date" not in content_parsed
    assert meta_parsed == content_parsed


def test_meta_parser_matches_content_parser_on_malformed_binary_field():
    rcs = SimpleRCS()
    block = b"ver @1.0@;\nbinary @4;base64,aGVsbG8=date @bogus-trailing-date@;\n\n"

    content_parsed = rcs._parse_block_content_no_regex(block)

    rcs.stream = io.BytesIO(block)
    meta_parsed = rcs._parse_block_meta_from_stream(0, len(block))

    assert "date" not in content_parsed
    assert "content_stream_offset" not in content_parsed
    assert meta_parsed == content_parsed


def test_commit_after_non_block_trailing_bytes_stays_findable():
    rcs = SimpleRCS()
    # Simulate a stream with non-block trailing bytes despite head_info
    # being None -- e.g. external corruption, a truncated prior write, or a
    # caller-supplied stream not entirely written by SimpleRCS.
    rcs.stream.seek(0, 2)
    rcs.stream.write(b"some stray garbage bytes with no block framing")
    rcs._load_head(force=True)
    assert rcs.head_info is None

    rcs.commit("hello", author="alice")

    reloaded = SimpleRCS(rcs.stream.getvalue())
    assert reloaded.checkout() == "hello\n"
    assert len(reloaded.log()) == 1
    assert reloaded.log()[0]["author"] == "alice"


def test_load_head_scan_stays_roughly_linear_with_many_collisions():
    def build(paragraphs: int) -> str:
        return "\n\n".join(f"version @9.9 paragraph {i} " + "x" * 200 for i in range(paragraphs)) + "\n"

    times = []
    for n in (200, 800, 3200):  # 4x content size each step
        rcs = SimpleRCS()
        rcs.commit(build(n), author="a", log="v1")
        raw = rcs.stream.getvalue()

        t0 = time.perf_counter()
        reloaded = SimpleRCS(raw)
        times.append(time.perf_counter() - t0)

        assert reloaded.checkout() == build(n)

    # Linear scanning costs ~4x per 4x-size step; the pre-fix quadratic
    # behavior measured close to 13-16x. A generous 8x still clearly tells
    # them apart without being sensitive to machine noise.
    assert times[1] / times[0] < 8
    assert times[2] / times[1] < 8


def test_field_value_ending_in_marker_prefix_is_not_a_block_start():
    """A value ending in "ver " borrows the closing '@' to form a fake marker.

    Distinct from the escaped-'@' collisions the other tests cover: here the
    '@' completing "ver @" is the delimiter _format_block writes, not escaped
    content, so it is not part of a '@@' pair. Any check that only looks for
    doubled '@' would let this through. log/author take the caller's string
    verbatim, unlike text, which commit() always terminates with a newline.
    """
    rcs = SimpleRCS()
    rcs.commit("real content here\n", author="t", log="Please check ver ")

    raw = rcs.stream.getvalue()
    fake = raw.find(b"ver @", raw.find(b"log @"))
    assert fake != -1, "precondition: the log field must produce a fake marker"
    assert raw[fake + 5 : fake + 6] == b";", "precondition: not a '@@' pair"

    reloaded = SimpleRCS(raw)
    assert reloaded.checkout() == "real content here\n"
    assert reloaded.log()[0]["log"] == "Please check ver "


def _damage_head_boundary(rcs: SimpleRCS) -> bytes:
    """Drop one of the two newlines framing HEAD, breaking the boundary invariant.

    Stands in for a stream SimpleRCS didn't write end to end: external damage,
    a truncated write, or a file an older build appended to without the
    separator commit() now writes.
    """
    rcs._load_head(force=True)  # commit() leaves head_info stale
    raw = rcs.stream.getvalue()
    head_start = rcs.head_info["start"]
    assert raw[head_start - 2 : head_start] == b"\n\n"
    return raw[: head_start - 1] + raw[head_start:]


def test_unlocatable_head_does_not_merge_into_an_earlier_block():
    """Falling back to an earlier block must not absorb the later one's fields.

    The scan is handed a range ending at EOF, so a candidate resolved by
    fallback used to parse straight through its own terminator and let the
    following block's ver/hash overwrite it -- yielding a head_info whose
    'ver' named one version while 'start' pointed at another. A commit() on
    top of that then rewrote from the earlier offset, dropping the version in
    between.
    """
    rcs = SimpleRCS()
    rcs.commit("first\n", author="t", log="a")
    rcs.commit("second\n", author="t", log="b")
    damaged = _damage_head_boundary(rcs)

    reloaded = SimpleRCS(damaged)
    if reloaded.head_info is not None:
        # Whatever it settles on, the identity and the offset must agree.
        start = reloaded.head_info["start"]
        assert damaged[start:].startswith(f"ver @{reloaded.head_info['ver']}@;".encode())

    # log() reads through the metadata-only parser, checkout() through the
    # content one; the two must not disagree about what HEAD is.
    assert (reloaded.log()[0]["ver"] if reloaded.log() else None) == (
        reloaded.head_info["ver"] if reloaded.head_info else None
    )


def test_delta_block_is_never_accepted_as_head():
    """HEAD is always full text, so a delta block means the real HEAD is missing.

    Accepting one would return the raw, unapplied delta script as content --
    there is nothing after it to apply it against.
    """
    rcs = SimpleRCS()
    rcs.commit("first\n", author="t", log="a")
    rcs.commit("second\n", author="t", log="b")
    damaged = _damage_head_boundary(rcs)

    reloaded = SimpleRCS(damaged)
    assert not (reloaded.head_info or {}).get("is_delta"), "a demoted block was accepted as HEAD"
    # The 1.0 block holds "d1 1\na1 1\nfirst" -- a delta script, not content.
    assert "d1 1" not in reloaded.checkout(), "checkout() returned an unapplied delta script"


def test_parsers_stop_at_the_next_block_start():
    """All three parsers stop at the following block rather than merging it in."""
    rcs = SimpleRCS()
    rcs.commit("first\n", author="t", log="a")
    rcs.commit("second\n", author="t", log="b")
    rcs._load_head(force=True)
    raw = rcs.stream.getvalue()

    # Deliberately hand each parser a range spanning *both* blocks.
    two_blocks = raw[rcs._data_start :]
    assert two_blocks.count(b"ver @") == 2, "precondition: the range covers two blocks"

    for parse in (rcs._parse_block_content_no_regex, rcs._parse_block_content):
        parsed = parse(two_blocks)
        assert parsed["ver"] == "1.0", f"{parse.__name__} merged the next block"

    meta = rcs._parse_block_meta_from_stream(rcs._data_start, len(raw))
    assert meta["ver"] == "1.0", "_parse_block_meta_from_stream merged the next block"
