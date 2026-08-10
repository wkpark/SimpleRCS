"""Regression tests for the HEAD/history backward-scan false-positive bug.

`_load_head()` and `_get_prev_block()` locate block boundaries by scanning raw
bytes backwards for the literal marker "ver @"/"version @". `codec.escape`
only doubles '@' -> '@@'; it does not protect that marker string from
occurring *inside* an escaped field value. Ordinary content like

    "Please ask whoever @admin is on call today."

contains "...ver @..." (from "whoever" + the space before its escaped '@')
once escaped, which used to make the backward scan mistake a byte offset
*inside* the HEAD block's own text field for the start of a (partial, later)
block — silently returning truncated/empty content or a metadata dict with
None fields, instead of raising or scanning past it.

Found auditing a downstream consumer (simplercs-wiki) against the installed
package: verified directly, not speculative. See `_is_block_boundary`'s
docstring for the fix (a real block start must be immediately preceded by
the blank line `_format_block` always writes, or be the very first block).

That boundary check alone isn't sufficient, though: `codec.escape` doesn't
escape newlines, so ordinary content containing a literal blank line
immediately followed by "ver @"/"version @" (e.g. changelog/wiki prose like
"...\n\nversion @2.0 release notes...") still passes it, and used to make
the scanner stop looking (either accepting garbage or giving up on the
whole buffer) instead of continuing left to the real block. `_scan_for_block`
now keeps retrying further left whenever a structurally-valid-looking
candidate still fails to parse, not just when it fails the boundary check.
See `test_head_checkout_survives_blank_line_and_marker_collision` et al.
"""

from simple_rcs.simple_rcs import SimpleRCS

# Ends up containing the literal byte sequence "ver @" once escaped (@ -> @@):
# "...whoever @@admin..." -> rfind(b"ver @") matches inside "whoever ' '@@".
COLLIDING_TEXT = "Please ask whoever @admin is on call today.\n"

# Ends up containing the literal byte sequence "version @" once escaped:
# "...changelog: version @@2..." -> rfind(b"version @") matches at "version ' '@@".
COLLIDING_TEXT_VERSION_MARKER = "changelog: version @2 fixes the bug.\n"

# Contains a *real* blank line (newlines aren't escaped) immediately followed
# by "version @" -- passes _is_block_boundary's blank-line check on its own,
# so this only round-trips correctly because of the retry-on-parse-failure
# loop in _scan_for_block, not the boundary check alone.
COLLIDING_LOG_MESSAGE = "Initial import.\n\nversion @2.0 release notes...\n"


def _reload(rcs: SimpleRCS) -> SimpleRCS:
    """Round-trips through raw bytes to force a fresh backward scan, the
    same as opening a file that was written by a previous process."""
    raw = rcs.stream.getvalue()
    return SimpleRCS(raw)


def test_head_checkout_survives_colliding_content_in_first_commit():
    rcs = SimpleRCS()
    rcs.commit(COLLIDING_TEXT, author="alice", log="v1")

    reloaded = _reload(rcs)
    assert reloaded.checkout() == COLLIDING_TEXT
    assert reloaded.log(limit=1)[0]["author"] == "alice"


def test_head_checkout_survives_colliding_content_in_later_commit():
    rcs = SimpleRCS()
    rcs.commit("Line 1\n", author="alice", log="v1")
    rcs.commit(COLLIDING_TEXT, author="bob", log="v2")

    reloaded = _reload(rcs)
    assert reloaded.checkout() == COLLIDING_TEXT
    meta = reloaded.log(limit=1)[0]
    assert meta["author"] == "bob"
    assert meta["ver"] == "1.1"


def test_head_checkout_survives_version_marker_collision():
    rcs = SimpleRCS()
    rcs.commit(COLLIDING_TEXT_VERSION_MARKER, author="carol", log="v1")

    reloaded = _reload(rcs)
    assert reloaded.checkout() == COLLIDING_TEXT_VERSION_MARKER
    assert reloaded.log(limit=1)[0]["author"] == "carol"


def test_historical_block_survives_colliding_content_in_earlier_commit():
    rcs = SimpleRCS()
    rcs.commit(COLLIDING_TEXT, author="alice", log="v1")
    rcs.commit("Line 1\nLine 2\n", author="bob", log="v2")
    rcs.commit("Line 1\nLine 2\nLine 3\n", author="carol", log="v3")

    reloaded = _reload(rcs)
    # V1 is a historical (non-HEAD) block, reached via _get_prev_block's own
    # backward scan from V3 -> V2 -> V1 -- exercises the same false-positive
    # marker there, not just in _load_head.
    assert reloaded.checkout("1.0") == COLLIDING_TEXT
    history = reloaded.log()
    assert [h["ver"] for h in history] == ["1.2", "1.1", "1.0"]
    assert history[-1]["author"] == "alice"


def test_blame_survives_colliding_content_in_earlier_commit():
    rcs = SimpleRCS()
    rcs.commit(COLLIDING_TEXT, author="alice", log="v1")
    rcs.commit(COLLIDING_TEXT + "One more line.\n", author="bob", log="v2")

    reloaded = _reload(rcs)
    blame = reloaded.blame()
    assert [b["line"] for b in blame] == [*COLLIDING_TEXT.splitlines(), "One more line."]
    assert blame[-1]["ver"] == "1.1"


def test_head_checkout_survives_blank_line_and_marker_collision():
    # The colliding text lives in the *log* field (not the page content),
    # and passes _is_block_boundary's blank-line check on its own -- only
    # the retry-on-parse-failure loop in _scan_for_block saves this one.
    rcs = SimpleRCS()
    rcs.commit("hello world\n", author="alice", log=COLLIDING_LOG_MESSAGE)

    reloaded = _reload(rcs)
    assert reloaded.checkout() == "hello world\n"
    meta = reloaded.log(limit=1)[0]
    assert meta["ver"] == "1.0"
    assert meta["author"] == "alice"
    assert meta["log"] == COLLIDING_LOG_MESSAGE


def test_log_has_no_phantom_entry_from_blank_line_and_marker_collision():
    # Regression for _parse_block_meta_from_stream's old leniency: a
    # candidate that failed to parse because its terminating ';' was
    # missing used to be stored anyway (as e.g. {'ver': '', 'author': None,
    # ...}), inserting a fabricated row into log()/blame() instead of being
    # discarded like the non-metadata parser already discarded it.
    rcs = SimpleRCS()
    rcs.commit("Line 1\n", author="alice", log="v1")
    rcs.commit("Line 1\nLine 2\n", author="bob", log=COLLIDING_LOG_MESSAGE)

    reloaded = _reload(rcs)
    history = reloaded.log()
    assert [h["ver"] for h in history] == ["1.1", "1.0"]
    assert all(h["ver"] not in ("", None) and h["author"] is not None for h in history)


def test_historical_block_survives_blank_line_and_marker_collision():
    rcs = SimpleRCS()
    rcs.commit(COLLIDING_LOG_MESSAGE, author="alice", log="v1")
    rcs.commit("Line 1\nLine 2\n", author="bob", log="v2")

    reloaded = _reload(rcs)
    assert reloaded.checkout("1.0") == COLLIDING_LOG_MESSAGE
    history = reloaded.log()
    assert [h["ver"] for h in history] == ["1.1", "1.0"]
    assert history[-1]["author"] == "alice"
