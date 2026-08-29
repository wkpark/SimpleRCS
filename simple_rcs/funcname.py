"""Name the enclosing function on a hunk header, the way ``git diff`` does.

A hunk header carries a line range, which says where a change is but not what
it is part of. git appends the enclosing declaration -- ``@@ -4,3 +4,3 @@ def
render(self):`` -- and finds it with a rule simple enough to be worth stating
plainly: walk *upwards* from the line above the hunk and take the first line
that looks like a declaration.

Three details of that rule are easy to get wrong, and all three are reproduced
here because a header that names the wrong function is worse than one that
names none:

* The search runs over the **pre-image** only. Rename a function and edit its
  body in one commit, and the header shows the old name. That is not a defect;
  the header describes where the removed lines came from.
* The search stops at the point where the *previous* hunk began looking, so no
  span of the file is scanned twice.
* When the search finds nothing, the previous hunk's name is kept rather than
  cleared. Finding no new declaration means the hunk is still inside the one
  already named.

Without a driver the test is purely positional -- a line whose first character
begins an identifier. It cannot see an indented Python ``def``; that needs a
driver. The drivers here are written for this project and are deliberately
short: a hunk label is a hint, and a wrong hint costs a reader more than a
missing one.
"""

import re
from collections.abc import Callable, Sequence

#: git truncates the name it puts on the header to a fixed-size buffer.
MAX_LABEL = 80

#: Per-language declaration patterns, tried in order. A pattern may be prefixed
#: with '!' to mean "if this matches, the line is *not* a declaration" -- it
#: rejects immediately without trying the ones after it. Where a pattern has a
#: capturing group, the group is what gets shown, which is how leading
#: indentation is dropped from the label. What follows the declaration head
#: -- an opening brace, a return type, a trailing comment -- is kept, as git
#: does: the label is the line, minus indentation, not a parsed-out name.
DRIVERS: dict[str, tuple[str, ...]] = {
    "python": (
        r"^[ \t]*((?:async[ \t]+)?(?:def|class)[ \t].*)$",
    ),
    "c": (
        # Preprocessor lines, labels and block ends open at column 0 too.
        r"!^#",
        r"!^[}\])]",
        r"!^[A-Za-z_][A-Za-z0-9_]*:[ \t]*$",
        r"^([A-Za-z_$].*)$",
    ),
    "java": (
        # A call or a control-flow head can look exactly like a signature.
        r"!^[ \t]*(?:catch|do|for|if|new|return|switch|throw|while)\b",
        r"!^[ \t]*(?:import|package)[ \t]",
        # At least one leading token, and nothing that ends the statement:
        # without both, `doSomething();` reads as a signature.
        r"^[ \t]*((?:[\w$<>\[\],.?&]+[ \t]+)+[\w$]+[ \t]*\([^;]*)$",
        r"^[ \t]*((?:public|private|protected|abstract|final|static|[ \t])*"
        r"(?:class|interface|enum|record)[ \t].*)$",
    ),
    "javascript": (
        # `if (x) {` and `render(props) {` are the same shape to a regex.
        r"!^[ \t]*(?:if|for|while|switch|catch|do|else|return|try|with)\b",
        r"^[ \t]*((?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?"
        r"(?:function\b|class\b|const\b|let\b|var\b).*)$",
        r"^[ \t]*([\w$.\[\]'\"]+[ \t]*[:=][ \t]*(?:async[ \t]+)?"
        r"(?:function\b|\(.*\)[ \t]*=>).*)$",
        r"^[ \t]*((?:get|set|static|async)?[ \t]*[\w$]+[ \t]*\(.*\)[ \t]*\{.*)$",
    ),
    "go": (
        r"^((?:func|type|package)[ \t].*)$",
    ),
    "rust": (
        r"^[ \t]*((?:pub[ \t]*(?:\([^)]*\))?[ \t]*)?(?:const[ \t]+|async[ \t]+|unsafe[ \t]+|extern[ \t]+\S+[ \t]+)*"
        r"(?:fn|struct|enum|trait|impl|mod|union|macro_rules!)[ \t!].*)$",
    ),
    "ruby": (
        r"^[ \t]*((?:def|class|module)[ \t].*)$",
    ),
    "shell": (
        r"^[ \t]*([\w.-]+[ \t]*\([ \t]*\).*)$",
        r"^[ \t]*(function[ \t]+[\w.-]+.*)$",
    ),
    "php": (
        r"^[ \t]*((?:(?:abstract|final|public|private|protected|static)[ \t]+)*"
        r"(?:function|class|interface|trait|enum)[ \t].*)$",
    ),
    "perl": (
        r"^[ \t]*((?:sub|package)[ \t].*)$",
    ),
    "lua": (
        r"^[ \t]*((?:local[ \t]+)?function[ \t].*)$",
    ),
    "css": (
        r"^([^\s@].*\{[ \t]*)$",
        r"^(@(?:media|supports|keyframes|font-face)\b.*)$",
    ),
    "markdown": (
        r"^(#{1,6}[ \t].*)$",
    ),
    "toml": (
        r"^(\[.*\])[ \t]*$",
    ),
}

#: Aliases, including the names git uses for the same languages, so a driver
#: remembered from ``.gitattributes`` resolves here too.
DRIVERS["cpp"] = DRIVERS["c"]
DRIVERS["typescript"] = DRIVERS["javascript"]
DRIVERS["bash"] = DRIVERS["shell"]
DRIVERS["golang"] = DRIVERS["go"]

#: Extension -> driver, for ``auto``. A name with no entry falls back to the
#: positional rule, which is what git does for an unregistered type.
EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyx": "python", ".pxd": "python",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp",
    ".java": "java",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby", ".rake": "ruby",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".php": "php",
    ".pl": "perl", ".pm": "perl",
    ".lua": "lua",
    ".css": "css", ".scss": "css", ".less": "css",
    ".md": "markdown", ".markdown": "markdown",
    ".toml": "toml",
}


def _clip(text: str) -> str:
    """Truncate to the label budget, then drop trailing space.

    The order matters: cutting a line at MAX_LABEL can leave whitespace at the
    new end, and a label with a ragged tail reads as though something was lost
    rather than trimmed.
    """
    return text[:MAX_LABEL].rstrip()


def positional_match(line: str) -> str | None:
    """The driverless rule: a declaration is a line that starts an identifier.

    No language is parsed and no keyword is known. It works because in most
    file formats written without indentation at the top level -- C, Go, shell
    -- a line beginning in column 0 with an identifier character is a
    declaration far more often than not.
    """
    stripped = line.rstrip("\r\n")
    if stripped and (stripped[0].isalpha() or stripped[0] in "_$"):
        return _clip(stripped)
    return None


class RegexMatcher:
    """A driver: ordered patterns, some of which reject rather than accept."""

    def __init__(self, patterns: Sequence[str], *, negation: bool = True) -> None:
        """``negation=False`` takes every pattern literally.

        The '!' prefix is this module's own notation for writing drivers. A
        pattern that came from the caller has no such convention -- a leading
        '!' there is just a character to match, as in CSS's ``!important``.
        """
        if not patterns:
            raise ValueError("a driver needs at least one pattern")
        if negation and patterns[-1].startswith("!"):
            # Every pattern being a rejection would accept nothing at all, and
            # the mistake is silent -- no header would ever carry a name.
            raise ValueError("the last pattern must not be a negation")
        self._regexes = []
        for pattern in patterns:
            negated = negation and pattern.startswith("!")
            self._regexes.append((negated, re.compile(pattern[1:] if negated else pattern)))

    def __call__(self, line: str) -> str | None:
        stripped = line.rstrip("\r\n")
        for negated, regex in self._regexes:
            match = regex.search(stripped)
            if not match:
                continue
            if negated:
                return None
            group = 1 if match.lastindex and match.group(1) is not None else 0
            return _clip(match.group(group))
        return None


def matcher_for(driver: str | None = None, filename: str | None = None,
                pattern: str | None = None) -> Callable[[str], str | None]:
    """Pick the line test: an explicit pattern, a named driver, or the default.

    ``driver`` may be ``"auto"``, in which case ``filename``'s extension
    decides. An extension nobody registered is not an error -- it falls back to
    the positional rule, so the feature degrades to "sometimes right" rather
    than to a crash.
    """
    if pattern is not None:
        return RegexMatcher([pattern], negation=False)
    if driver in (None, "auto"):
        if filename is None:
            return positional_match
        suffix = filename[filename.rfind("."):].lower() if "." in filename else ""
        driver = EXTENSIONS.get(suffix)
        if driver is None:
            return positional_match
    if driver == "default":
        return positional_match
    if driver not in DRIVERS:
        raise ValueError(f"unknown funcname driver {driver!r}")
    return RegexMatcher(DRIVERS[driver])


class HunkAnnotator:
    """Labels for successive hunks of one file, in the order they are emitted.

    Stateful on purpose: both the lower bound of the search and the carried
    label depend on the previous hunk. Feed hunks in ascending order, and use a
    fresh instance per file.
    """

    def __init__(self, pre_image: Sequence[str],
                 matcher: Callable[[str], str | None] | None = None) -> None:
        self._lines = pre_image
        self._match = matcher or positional_match
        self._searched_from = -1
        self._label = ""

    def label(self, hunk_start: int) -> str:
        """The name for a hunk starting at ``hunk_start`` (0-based, pre-image).

        Returns "" only until the first declaration is found; after that some
        label is always returned, since an unsuccessful search keeps the last
        one.
        """
        start = hunk_start - 1
        limit = self._searched_from
        step = -1 if start > limit else 1
        index = start
        while index != limit and 0 <= index < len(self._lines):
            found = self._match(self._lines[index])
            if found is not None:
                self._label = found
                break
            index += step
        self._searched_from = start
        return self._label


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def annotate_unified_diff(lines, annotator: HunkAnnotator):
    """Add labels to the hunk headers of an already-formatted unified diff.

    For diffs this module did not lay out itself -- ``difflib.unified_diff``'s,
    say -- the start line has to be read back out of the header. A hunk with
    zero lines on a side reports the position *before* it rather than its own
    first line, which is the one place the arithmetic differs.

    One file's diff, in the order it was emitted: the annotator's pre-image and
    its carried state belong to a single file. Whatever the caller used to end
    its lines is preserved, including ``lineterm=""``.
    """
    for line in lines:
        match = _HUNK_HEADER.match(line)
        if match is None:
            yield line
            continue
        start = int(match.group(1))
        count = 1 if match.group(2) is None else int(match.group(2))
        label = annotator.label(start if count == 0 else start - 1)
        if not label:
            yield line
            continue
        rest = line[match.end():]
        terminator = rest[len(rest.rstrip("\r\n")):]
        yield f"{line[:match.end()]} {label}{terminator}"
