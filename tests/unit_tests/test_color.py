"""Diff colouring.

Two layers. The first pins where each escape goes, because that is what a
reader notices when it is wrong -- a reset on the far side of the newline
paints the terminal to the right margin, and a sign that loses its colour makes
an added line look like context. The second feeds real ``git diff`` output
through the colouriser and demands the bytes match what git itself would have
printed, which is the only check that stays honest as the rules get edited.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from simple_rcs import color
from simple_rcs.color import RESET, Palette, colorize_unified_diff, want_color

GIT = shutil.which("git")

BOLD = "\033[1m"
CYAN = "\033[36m"
RED = "\033[31m"
GREEN = "\033[32m"
BG_RED = "\033[41m"


#: Any header will do; the colouriser only reads it as "a hunk starts here".
HUNK = "@@ -1,9 +1,9 @@\n"


def paint(*lines: str) -> list[str]:
    """Colour lines from the top of a diff, before any hunk has started."""
    return list(colorize_unified_diff(lines))


def body(*lines: str) -> list[str]:
    """Colour lines from inside a hunk, with the header dropped again."""
    return list(colorize_unified_diff([HUNK, *lines]))[1:]


# --------------------------------------------------------------------------
# escape placement
# --------------------------------------------------------------------------

def test_the_reset_comes_before_the_newline():
    """After it, the terminal paints its background out to the right margin."""
    assert body("-gone\n") == [f"{RED}-gone{RESET}\n"]


def test_a_line_without_a_newline_keeps_not_having_one():
    """difflib yields the last line of a file that ends without one as-is."""
    assert body("-gone") == [f"{RED}-gone{RESET}"]


def test_a_carriage_return_is_part_of_the_terminator():
    assert body("-gone\r\n") == [f"{RED}-gone{RESET}\r\n"]


def test_context_lines_are_reset_even_though_they_are_not_coloured():
    """The slot is empty, but the reset still terminates any colour the
    *content* carried -- a diff of a file that itself contains escapes."""
    assert body(" kept\n") == [f" kept{RESET}\n"]


def test_a_blank_context_line_is_still_reset():
    """There is nothing to colour, but the run was opened by the sign, so it
    is closed -- git resets whenever it wrote a colour, empty or not."""
    assert body(" \n") == [f" {RESET}\n"]


def test_an_added_line_colours_its_sign_separately_from_its_content():
    """One run, and a whitespace error in the middle would strip the sign of
    its colour along with everything after it."""
    assert body("+new\n") == [f"{GREEN}+{RESET}{GREEN}new{RESET}\n"]


def test_an_empty_added_line_is_just_the_sign():
    assert body("+\n") == [f"{GREEN}+{RESET}\n"]


def test_trailing_whitespace_on_an_added_line_is_highlighted():
    assert body("+new   \n") == [f"{GREEN}+{RESET}{GREEN}new{RESET}{BG_RED}   {RESET}\n"]


def test_trailing_whitespace_on_a_removed_line_is_not():
    """git's default ws-error-highlight is `new` only: the line is going away."""
    assert body("-gone   \n") == [f"{RED}-gone   {RESET}\n"]


def test_a_line_that_is_only_whitespace_is_all_error():
    assert body("+  \n") == [f"{GREEN}+{RESET}{BG_RED}  {RESET}\n"]


def test_a_tab_indent_is_neither_coloured_nor_flagged():
    assert body("+\tnew\n") == [f"{GREEN}+{RESET}\t{GREEN}new{RESET}\n"]


def test_spaces_before_a_tab_in_the_indent_are_flagged():
    """space-before-tab: the spaces are the error, the tab that follows is not."""
    assert body("+  \tnew\n") == [f"{GREEN}+{RESET}{BG_RED}  {RESET}\t{GREEN}new{RESET}\n"]


def test_a_space_after_a_tab_in_the_indent_is_fine():
    assert body("+\t new\n") == [f"{GREEN}+{RESET}\t{GREEN} new{RESET}\n"]


def test_spaces_before_a_tab_outside_the_indent_are_not_flagged():
    """The rule is about indentation; a tab used as a separator is not an error."""
    assert body("+zz \t x\n") == [f"{GREEN}+{RESET}{GREEN}zz \t x{RESET}\n"]


def test_only_c_locale_blanks_count_as_trailing_whitespace():
    """`str.isspace` also accepts U+00A0 and friends; git's `isspace` does not,
    and flagging a character git leaves alone is worse than flagging none."""
    assert body("+new\u00a0\n") == [f"{GREEN}+{RESET}{GREEN}new\u00a0{RESET}\n"]


def test_an_empty_whitespace_slot_turns_the_check_off():
    """git's own way of disabling it: an empty `color.diff.whitespace`. With
    nothing to interrupt the content, the line stops being split at the sign --
    verified against `git -c color.diff.whitespace= diff --color=always`."""
    quiet = Palette(whitespace="")
    assert list(colorize_unified_diff([HUNK, "+new   \n"], quiet))[1] == (
        f"{GREEN}+new   {RESET}\n"
    )


# --------------------------------------------------------------------------
# the hunk header
# --------------------------------------------------------------------------

def test_the_hunk_header_is_three_runs():
    """frag for the range, context for the blank, func for the name -- so that
    a reader can dim the name without dimming the range."""
    assert paint("@@ -2,3 +2,3 @@ def f():\n") == [
        f"{CYAN}@@ -2,3 +2,3 @@{RESET} {RESET}def f():{RESET}\n"
    ]


def test_a_hunk_header_without_a_name_stops_after_the_range():
    assert paint("@@ -1,4 +1,6 @@\n") == [f"{CYAN}@@ -1,4 +1,6 @@{RESET}\n"]


@pytest.mark.parametrize("line", ["@@ nope\n", "@@ @@\n"])
def test_something_that_only_starts_like_a_hunk_header_is_left_as_context(line):
    """A header needs both a closing `@@` and ten bytes of range. The second
    case has the `@@` and not the length, which is the half of the rule a
    colouriser is likely to drop."""
    assert paint(line) == [f"{line.rstrip(chr(10))}{RESET}\n"]


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def test_file_headers_are_bold():
    assert paint("--- a/x.py\n", "+++ b/x.py\n") == [
        f"{BOLD}--- a/x.py{RESET}\n", f"{BOLD}+++ b/x.py{RESET}\n"
    ]


def test_a_removed_line_of_dashes_is_not_mistaken_for_a_file_header():
    """Deleting a line that reads `---` yields `----`, which a prefix test
    would colour bold. Inside a hunk only the sign is consulted."""
    assert paint("@@ -1,2 +1,1 @@\n", "----\n") == [
        f"{CYAN}@@ -1,2 +1,1 @@{RESET}\n", f"{RED}----{RESET}\n"
    ]


def test_an_added_line_of_pluses_is_not_mistaken_for_a_file_header():
    assert paint("@@ -1,1 +1,2 @@\n", "++++\n")[1] == f"{GREEN}+{RESET}{GREEN}+++{RESET}\n"


def test_the_next_file_header_ends_the_hunk():
    out = paint("@@ -1,1 +1,1 @@\n", " kept\n", "diff --git a/y b/y\n", "--- a/y\n")
    assert out[2] == f"{BOLD}diff --git a/y b/y{RESET}\n"
    assert out[3] == f"{BOLD}--- a/y{RESET}\n"


def test_the_no_newline_marker_is_context():
    assert paint("@@ -1,1 +1,1 @@\n", "\\ No newline at end of file\n")[1] == (
        f"\\ No newline at end of file{RESET}\n"
    )


# --------------------------------------------------------------------------
# want_color
# --------------------------------------------------------------------------

class _Stream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_always_and_never_ignore_the_stream(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert want_color("always", _Stream(False)) is True
    assert want_color("never", _Stream(True)) is False


def test_auto_follows_the_stream(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    assert want_color("auto", _Stream(True)) is True
    assert want_color("auto", _Stream(False)) is False


@pytest.mark.parametrize(("name", "value"), [("NO_COLOR", "1"), ("TERM", "dumb")])
def test_auto_gives_way_to_the_environment(monkeypatch, name, value):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv(name, value)
    assert want_color("auto", _Stream(True)) is False


def test_a_stream_that_cannot_answer_is_not_a_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    assert want_color("auto", object()) is False


def test_an_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="auto, always or never"):
        want_color("sometimes", _Stream(True))


# --------------------------------------------------------------------------
# against real git
# --------------------------------------------------------------------------

BEFORE = """\
def render(self, node):
    head = node.head
    body = node.body
    return head + body


def walk(self, tree):
\tfor node in tree:
\t\tyield node
"""

#: The trailing blanks are the whole point of the fixture, so they are spliced
#: in rather than written at the end of a source line, where the linter (rightly)
#: strips them.
AFTER = """\
def render(self, node):
    head = node.head
    body = node.body
    return head + body + "\\n"


def walk(self, tree):
\tfor node in tree:
  \t\tyield node
""".replace("node.body\n", "node.body   \n")


@pytest.mark.skipif(GIT is None, reason="git is not installed")
@pytest.mark.parametrize("context", [0, 1, 3, 5])
def test_colouring_a_plain_git_diff_reproduces_a_coloured_one(tmp_path, context):
    """The only check that does not encode my reading of diff.c back into the
    assertion. git renders both forms of the same diff; the colouriser has to
    turn the first into the second, byte for byte.
    """
    def run(*args: str) -> str:
        result = subprocess.run(
            [GIT, *args], cwd=tmp_path, capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TERM": "xterm"},
        )
        return result.stdout

    run("init", "-q", ".")
    Path(tmp_path / "render.py").write_text(BEFORE)
    run("add", "render.py")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    Path(tmp_path / "render.py").write_text(AFTER)

    width = f"-U{context}"
    plain = run("-c", "color.ui=never", "diff", width, "render.py")
    coloured = run("-c", "color.ui=always", "diff", width, "render.py")

    assert plain, "the fixture produced no diff"
    assert "".join(colorize_unified_diff(plain.splitlines(keepends=True))) == coloured


def test_a_blank_line_added_at_eof_is_a_known_divergence():
    """git paints the whole line, sign included, when the added blank line is
    the last in the file (`blank-at-eof`). Deciding that needs to know where
    the new file ends, which a stream of diff lines does not carry, so the line
    is coloured as an ordinary added one. Pinned rather than hidden."""
    assert body("+\t\n") == [f"{GREEN}+{RESET}{BG_RED}\t{RESET}\n"]
    # what git would print here is f"{BG_RED}+\t{RESET}\n"


def test_the_default_palette_is_gits():
    """Pinned so a change to a colour is a deliberate edit, not a drift."""
    assert color.DEFAULT_PALETTE == Palette(
        context="", meta=BOLD, frag=CYAN, old=RED, new=GREEN, func="", whitespace=BG_RED,
    )
