"""Colour a unified diff the way ``git diff`` does.

A unified diff is read by eye far more often than by ``patch``, and colour is
what makes the shape of a hunk visible at a glance. git's scheme is the one
every reader already knows, so this module reproduces it rather than inventing
another: red removals, green additions, cyan hunk headers, bold file headers,
and a red background on whitespace a reviewer would not want committed.

Reproducing it means reproducing where the escapes *go*, not just which colour
each line gets, and that placement is less obvious than it looks:

* The reset comes **before** the line terminator, never after. A reset after
  the newline paints the terminal's own background to the end of the line,
  which is what makes a naive colouriser look wrong on a dark theme.
* An added line is emitted as two runs -- the ``+`` on its own, then the
  content -- so that the whitespace check can interrupt the second one without
  the sign losing its colour.
* The hunk header is three runs: the ``@@ ... @@`` in *frag*, the blanks after
  it in *context*, and the function name in *func*.
* A line is reset even when its colour is the empty string. Context lines and
  function names therefore end in an escape that appears to do nothing, and
  that is deliberate -- it terminates any colour the *content* of the line may
  have carried.

Every rule here was read off ``diff.c`` (``emit_line_0``, ``emit_hunk_header``)
and ``ws.c`` (``ws_check_emit_1``) and is pinned against real ``git diff``
output by ``tests/unit_tests/test_color.py``.

Not implemented, deliberately: colour-moved detection, the ``blank-at-eof``
whitespace rule (it needs to know where the new file ends, which a line stream
does not), and the ``color.diff.*`` style-string parser. Callers that want
different colours pass a different :class:`Palette`.
"""

import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

#: ``\033[m``, git's ``GIT_COLOR_RESET``. Not ``\033[0m``; the two are
#: equivalent to a terminal, but staying byte-identical to git keeps the
#: differential test meaningful.
RESET = "\033[m"

#: What C's ``isspace`` matches in the C locale. ``str.isspace`` is wider (it
#: accepts U+00A0 and friends), which would flag whitespace git leaves alone.
_BLANKS = " \t\v\f\r"


@dataclass(frozen=True)
class Palette:
    """One escape sequence per ``color.diff.*`` slot.

    The defaults are git's own (``diff_colors`` in ``diff.c``). An empty string
    means "no colour", exactly as git's ``GIT_COLOR_NORMAL`` does -- the run is
    still reset, so the slot stays a place to hang a colour on later.

    Setting ``whitespace`` to ``""`` turns the whitespace check off outright,
    which is what git does when ``color.diff.whitespace`` is empty.
    """

    context: str = ""                 # GIT_COLOR_NORMAL
    meta: str = "\033[1m"             # GIT_COLOR_BOLD
    frag: str = "\033[36m"            # GIT_COLOR_CYAN
    old: str = "\033[31m"             # GIT_COLOR_RED
    new: str = "\033[32m"             # GIT_COLOR_GREEN
    func: str = ""                    # GIT_COLOR_NORMAL
    whitespace: str = "\033[41m"      # GIT_COLOR_BG_RED


DEFAULT_PALETTE = Palette()

#: A palette that emits nothing. Colouring with it is a no-op *in appearance*
#: only -- the resets are still written -- so callers that want untouched
#: output should skip the colouriser instead of passing this.
NO_COLOR_PALETTE = Palette(meta="", frag="", old="", new="", whitespace="")


def want_color(mode: str = "auto", stream=None) -> bool:
    """Whether to colour, given ``--color=<mode>`` and where the output goes.

    ``auto`` means "a terminal is reading this": not a pipe, not a file, not a
    terminal that cannot render escapes. ``NO_COLOR`` is honoured on top of
    that (git does not honour it, but every tool written since does, and a
    reader who exports it means it).
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode != "auto":
        raise ValueError(f"colour mode must be auto, always or never: {mode!r}")
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if stream is None:
        stream = sys.stdout
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        # A closed or non-file stream is not a terminal.
        return False


def _split_newline(line: str) -> tuple[str, str]:
    """Peel off ``\\n`` and then a ``\\r`` before it, git's order in ``emit_line_0``.

    The carriage return is peeled whether or not a newline followed it, which
    is what ``emit_line_0`` does -- the two checks are sequential, not nested.
    """
    text, term = line, ""
    if text.endswith("\n"):
        text, term = text[:-1], "\n"
    if text.endswith("\r"):
        text, term = text[:-1], "\r" + term
    return text, term


def _emit(color: str, body: str) -> str:
    """``emit_line_0`` for the shape this module needs.

    Every caller in ``diff.c`` that reaches us passes the colour as
    ``set_sign`` and ``NULL`` as ``set``, so the escape precedes the sign
    rather than sitting between the sign and the content -- ``body`` here is
    the line with its sign still on it. An empty colour still opens a run that
    has to be closed, which is why the reset does not depend on it.
    """
    text, term = _split_newline(body)
    if not text:
        return term
    return f"{color}{text}{RESET}{term}"


def _emit_added(body: str, palette: Palette) -> str:
    """``ws_check_emit_1`` (ws.c): the content of a ``+`` line, split so that
    whitespace a reviewer would reject shows up as a red block.

    Two of git's default rules are line-local and implemented here:
    ``blank-at-eol`` (trailing whitespace) and ``space-before-tab`` (spaces in
    the indent that a tab then follows). ``blank-at-eof`` is not: deciding it
    needs the end of the new file, which is not in a stream of diff lines.
    """
    ws = palette.whitespace
    text, term = (body[:-1], "\n") if body.endswith("\n") else (body, "")

    # WS_CR_AT_EOL is off by default, so a CR at the end is trailing
    # whitespace like any other -- git paints it too, and so do we.
    trailing = len(text)
    for i in range(len(text) - 1, -1, -1):
        if text[i] not in _BLANKS:
            break
        trailing = i

    out: list[str] = []
    written = index = 0
    while index < trailing:
        char = text[index]
        if char == " ":
            index += 1
            continue
        if char != "\t":
            break
        if written < index:
            # space-before-tab: the spaces are the error, the tab is not.
            out.append(f"{ws}{text[written:index]}{RESET}{char}")
        else:
            out.append(text[written : index + 1])
        written = index = index + 1

    if trailing - written > 0:
        out.append(f"{palette.new}{text[written:trailing]}{RESET}")
    if trailing != len(text):
        out.append(f"{ws}{text[trailing:]}{RESET}")
    return "".join(out) + term


def _emit_hunk_header(line: str, palette: Palette) -> str:
    """``emit_hunk_header`` (diff.c): frag, then the blanks, then the funcname."""
    body, term = _split_newline(line)
    end = body.find("@@", 2)
    if len(body) < 10 or end < 0:
        # A range needs ten bytes and a closing `@@`; without them git gives up
        # and prints the line as an ordinary context marker rather than
        # splitting it somewhere meaningless.
        return _emit(palette.context, line)
    end += 2

    out = [f"{palette.frag}{body[:end]}{RESET}"]
    blanks = end
    while blanks < len(body) and body[blanks] in " \t":
        blanks += 1
    if blanks != end:
        out.append(f"{palette.context}{body[end:blanks]}{RESET}")
    if blanks < len(body):
        out.append(f"{palette.func}{body[blanks:]}{RESET}")
    return "".join(out) + term


def colorize_unified_diff(lines: Iterable[str], palette: Palette = DEFAULT_PALETTE) -> Iterator[str]:
    """Colour a unified diff, one line in and one line out.

    Line terminators are preserved, including their absence on a last line that
    has none, so this can be dropped between any producer and its writer.

    Classification is stateful on purpose. Deleting a line that itself reads
    ``---`` produces ``----``, which a prefix test would mistake for a file
    header; inside a hunk only the first character is consulted, and anything
    that is not a diff sign ends the hunk.
    """
    in_hunk = False
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            yield _emit_hunk_header(line, palette)
            continue

        if in_hunk:
            sign = line[:1]
            if sign == "+":
                if not palette.whitespace:
                    # An empty slot turns the check off, and with it the reason
                    # to split the line at all (git: `if (!*ws) ws = NULL`).
                    yield _emit(palette.new, line)
                    continue
                # The sign is its own run so the whitespace check can break up
                # the content without it.
                yield _emit(palette.new, "+") + _emit_added(line[1:], palette)
                continue
            if sign == "-":
                yield _emit(palette.old, line)
                continue
            if sign in (" ", "\\", "\n", "\r", ""):
                yield _emit(palette.context, line)
                continue
            # A new file's header: this hunk is over.
            in_hunk = False

        yield _emit(palette.meta, line)
