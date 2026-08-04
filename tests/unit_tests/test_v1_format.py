"""
Regression tests for legacy v1-format files (no v2 header, no 'delta' keyword —
every non-HEAD block stores its content under 'text', disambiguated only by
position: last block = HEAD = full text, everything else = reverse delta).

Guards the _get_prev_block() fix: v1 blocks were previously always parsed with
is_delta=False (correct for v2's 'text' keyword, wrong for v1's overloaded
'text' keyword), which made checkout()/blame()/verify() return the raw,
unapplied delta script for every non-HEAD version instead of reconstructed
content.
"""

import pytest

from simple_rcs.simple_rcs import SimpleRCS, SimpleRCSCorruptionError

# A genuine v1-format sample (matches the original _format_block() from
# patches/0001-feat-initial-SimpleRCS-version-control-system.patch: no header
# line, keys = ["ver", "date", "author", "log", "text"] unconditionally).
V1_CONTENT = """ver @1.0@;
date @2025-11-29T15:32:18.351821@;
author @user1@;
log @Init@;
text @d2 2
a3 1
Line 2@;

ver @1.1@;
date @2025-11-29T15:32:18.352728@;
author @user1@;
log @Mod@;
text @d3 2
a4 1
Line 3@;

ver @1.2@;
date @2025-11-29T15:32:18.353336@;
author @user1@;
log @Add@;
text @Line 1
Line 2 Modified
Line 3
Line 4@;

"""


@pytest.fixture
def rcs():
    instance = SimpleRCS(V1_CONTENT)
    assert instance._version == 1, "fixture content must be detected as v1"
    return instance


def test_v1_detected(rcs):
    assert rcs._version == 1


def test_v1_checkout_head(rcs):
    assert rcs.checkout("1.2") == "Line 1\nLine 2 Modified\nLine 3\nLine 4"
    assert rcs.checkout() == "Line 1\nLine 2 Modified\nLine 3\nLine 4"


def test_v1_checkout_historical_versions(rcs):
    # Before the fix these returned the raw, unapplied delta scripts
    # ("d2 2\na3 1\nLine 2", etc.) instead of reconstructed content.
    assert rcs.checkout("1.0") == "Line 1\nLine 2\n"
    assert rcs.checkout("1.1") == "Line 1\nLine 2 Modified\nLine 3\n"


def test_v1_log(rcs):
    history = rcs.log()
    assert [h["ver"] for h in history] == ["1.2", "1.1", "1.0"]
    assert history[-1]["author"] == "user1"
    assert history[-1]["log"] == "Init"


def test_v1_blame(rcs):
    blame = rcs.blame()
    lines = [b["line"] for b in blame]
    assert lines == ["Line 1", "Line 2 Modified", "Line 3", "Line 4"]
    # Line 1 has never changed since 1.0
    assert blame[0]["ver"] == "1.0"
    # Line 2 was modified in 1.1
    assert blame[1]["ver"] == "1.1"


def test_v1_verify_always_true(rcs):
    # v1 has no hash-chain integrity feature; verify() is a no-op success.
    assert rcs.verify() is True


def test_v1_corrupted_delta_raises_on_checkout():
    corrupted = V1_CONTENT.replace("d2 2\na3 1", "dX 2\na3 1", 1)
    corrupted_rcs = SimpleRCS(corrupted)
    with pytest.raises(SimpleRCSCorruptionError):
        corrupted_rcs.checkout("1.0")
