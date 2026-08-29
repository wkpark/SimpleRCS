"""Hunk header labels: the search rule, the drivers, and agreement with git.

The rule is small but has three edges that a casual implementation gets wrong
-- pre-image only, carry-over when nothing is found, and the label being the
declaration *above* the hunk rather than inside it. Those are pinned here.

The last test is the one that matters most: it runs real git over the same
content and requires the labels to match. A driver is a guess about a language,
and the only way to know the guess is good is to compare it against the
implementation everyone else is reading.
"""

import difflib
import re
import shutil
import subprocess

import pytest

from simple_rcs import funcname

GIT = shutil.which("git")

PRE_IMAGE = """\
top_level_one
    indented
    body
top_level_two
    more
    body
    tail
""".splitlines(keepends=True)


def _annotate(pre_image, starts, matcher=None):
    """Labels for hunks starting at the given 0-based pre-image lines."""
    annotator = funcname.HunkAnnotator(pre_image, matcher)
    return [annotator.label(s) for s in starts]


# --- the search rule ---------------------------------------------------------

def test_the_label_is_the_declaration_above_the_hunk():
    assert _annotate(PRE_IMAGE, [1]) == ["top_level_one"]
    assert _annotate(PRE_IMAGE, [5]) == ["top_level_two"]


def test_a_hunk_starting_at_the_first_line_has_nothing_above_it():
    """Not a fallback to the first declaration: there is genuinely no context."""
    assert _annotate(PRE_IMAGE, [0]) == [""]


def test_a_declaration_is_not_found_by_looking_downwards():
    """Line 3 declares top_level_two, but a hunk starting at 3 is *at* it."""
    assert _annotate(PRE_IMAGE, [3]) == ["top_level_one"]


def test_a_hunk_with_no_declaration_above_it_keeps_the_previous_label():
    """The second hunk is still inside top_level_two, so it says so.

    Searching finds nothing between the two hunks; clearing the label there
    would leave the reader with less information than carrying it forward.
    """
    assert _annotate(PRE_IMAGE, [5, 6]) == ["top_level_two", "top_level_two"]


def test_hunks_before_the_first_declaration_stay_unlabelled():
    pre_image = ["    x\n", "    y\n", "decl\n", "    z\n"]
    assert _annotate(pre_image, [1, 3]) == ["", "decl"]


# --- the positional rule -----------------------------------------------------

@pytest.mark.parametrize("line", ["name\n", "_name\n", "$name\n", "int main(void)\n"])
def test_a_line_starting_an_identifier_is_a_declaration(line):
    assert funcname.positional_match(line) == line.strip()


@pytest.mark.parametrize("line", ["    indented\n", "\tindented\n", "}\n", "#define X\n", "\n"])
def test_anything_else_is_not(line):
    assert funcname.positional_match(line) is None


def test_the_label_is_truncated_before_its_trailing_space_is_dropped():
    """Cutting at the budget can land in the middle of a run of spaces, and a
    label ending in whitespace reads as truncation damage rather than a trim."""
    head = "g" * (funcname.MAX_LABEL - 2)
    assert funcname.positional_match(head + "    tail\n") == head

    over = "g" * (funcname.MAX_LABEL + 40)
    assert funcname.positional_match(over + "\n") == "g" * funcname.MAX_LABEL

    under = "g" * 10
    assert funcname.positional_match(under + "   \n") == under


def test_carriage_returns_are_not_part_of_the_label():
    assert funcname.positional_match("decl\r\n") == "decl"


# --- drivers -----------------------------------------------------------------

def test_a_driver_sees_declarations_the_positional_rule_cannot():
    line = "    def render(self):\n"
    assert funcname.positional_match(line) is None
    assert funcname.matcher_for("python")(line) == "def render(self):"


def test_a_capturing_group_is_what_gets_shown():
    matcher = funcname.RegexMatcher([r"^\s*(def .*)$"])
    assert matcher("        def go():\n") == "def go():"


def test_without_a_group_the_whole_match_is_shown():
    """Including the indentation it matched -- which is why every driver here
    wraps the part worth showing in a group."""
    matcher = funcname.RegexMatcher([r"^\s*def .*$"])
    assert matcher("  def go():\n") == "  def go():"


CALL_SHAPED = [
    # A regex sees no difference between these and a signature; the drivers
    # have to, or every statement in a body becomes a hunk label.
    ("java", "        doSomething();"),
    ("java", "        list.add(item);"),
    ("java", "        int x = compute(a, b);"),
    # Needs the "at least one leading token" half of the rule on its own.
    ("java", "        compute(a, b)"),
    # Needs the "no statement terminator" half on its own. git excludes bare
    # signatures like this for the same reason, so the label matches git's.
    ("java", "    void doThing();"),
    ("javascript", "  if (x) {"),
    ("javascript", "  for (const a of b) {"),
    ("javascript", "  while (ok) {"),
    ("javascript", "  switch (v) {"),
    ("typescript", "  if (x) {"),
]

DECLARATIONS = [
    ("java", "    public int add(int a, int b) {", "public int add(int a, int b) {"),
    ("java", "    public static <T> List<T> of(T x) {", "public static <T> List<T> of(T x) {"),
    ("java", "public class Foo {", "public class Foo {"),
    ("javascript", "  render(props) {", "render(props) {"),
    ("javascript", "  async load(id) {", "async load(id) {"),
    ("javascript", "export function go() {", "export function go() {"),
    ("javascript", "  const handler = (e) => {", "const handler = (e) => {"),
]


@pytest.mark.parametrize(("driver", "line"), CALL_SHAPED)
def test_a_call_or_a_control_flow_head_is_not_a_declaration(driver, line):
    assert funcname.matcher_for(driver)(line) is None


@pytest.mark.parametrize(("driver", "line", "expected"), DECLARATIONS)
def test_the_declarations_those_rejections_sit_next_to_still_match(driver, line, expected):
    """The rejections are narrow on purpose -- they must not take the
    signatures with them."""
    assert funcname.matcher_for(driver)(line) == expected


def test_prose_files_get_the_positional_rule_rather_than_a_guess():
    """A reStructuredText title is defined by the underline on the *next*
    line, which a per-line rule cannot see. A driver that matched anything
    unindented labelled paragraphs and bullets, which is worse than the
    fallback -- so there is no driver for it."""
    assert "rst" not in funcname.DRIVERS
    assert funcname.matcher_for("auto", "guide.rst") is funcname.positional_match
    assert funcname.positional_match("- a bullet item\n") is None
    assert funcname.positional_match("====\n") is None


def test_a_negated_pattern_rejects_without_trying_the_rest():
    matcher = funcname.RegexMatcher([r"!^if\b", r"^(\w+.*)$"])
    assert matcher("if (x) {\n") is None
    assert matcher("main(void) {\n") == "main(void) {"


def test_a_driver_of_only_negations_would_accept_nothing():
    """Silent if allowed: no header would ever carry a name."""
    with pytest.raises(ValueError, match="must not be a negation"):
        funcname.RegexMatcher([r"^(\w+)$", r"!^x"])
    with pytest.raises(ValueError, match="at least one pattern"):
        funcname.RegexMatcher([])


@pytest.mark.parametrize(("name", "driver"), [
    ("m.py", "python"), ("m.PY", "python"), ("m.c", "c"), ("m.rs", "rust"),
    ("m.go", "go"), ("m.sh", "shell"), ("m.md", "markdown"),
])
def test_auto_resolves_by_extension(name, driver):
    assert funcname.matcher_for("auto", name)("\n") is None  # smoke: it builds
    assert funcname.EXTENSIONS[name[name.rfind("."):].lower()] == driver


def test_an_unregistered_extension_degrades_to_the_positional_rule():
    """Better a sometimes-right label than a crash on an unknown file type."""
    assert funcname.matcher_for("auto", "notes.xyz") is funcname.positional_match
    assert funcname.matcher_for("auto", "Makefile") is funcname.positional_match
    assert funcname.matcher_for("auto", None) is funcname.positional_match


def test_a_caller_supplied_pattern_takes_a_leading_bang_literally():
    """'!' is this module's notation for writing drivers, not the user's: in
    CSS it is just the first character of `!important`."""
    # The pattern has to *start* with '!' for the convention to bite.
    matcher = funcname.matcher_for(None, None, r"!important.*")
    assert matcher("!important stuff\n") == "!important stuff"


def test_an_explicit_pattern_wins_over_the_driver():
    matcher = funcname.matcher_for("python", "m.py", r"^(SECTION .*)$")
    assert matcher("    def go(self):\n") is None
    assert matcher("SECTION two\n") == "SECTION two"


def test_a_misspelled_driver_is_an_error_rather_than_a_silent_fallback():
    with pytest.raises(ValueError, match="unknown funcname driver"):
        funcname.matcher_for("pythn", "m.py")


def test_the_names_git_uses_resolve_to_the_same_drivers():
    assert funcname.DRIVERS["bash"] == funcname.DRIVERS["shell"]
    assert funcname.DRIVERS["golang"] == funcname.DRIVERS["go"]
    assert funcname.DRIVERS["cpp"] == funcname.DRIVERS["c"]


# --- annotating an already-formatted diff ------------------------------------

def test_headers_of_a_difflib_diff_are_labelled():
    new = PRE_IMAGE[:5] + ["    changed\n"] + PRE_IMAGE[6:]
    lines = difflib.unified_diff(PRE_IMAGE, new, "a", "b", n=0)
    annotator = funcname.HunkAnnotator(PRE_IMAGE)
    headers = [ln.rstrip("\n") for ln in funcname.annotate_unified_diff(lines, annotator)
               if ln.startswith("@@")]
    assert headers == ["@@ -6 +6 @@ top_level_two"]


def test_a_zero_length_range_names_the_line_before_it():
    """difflib reports a pure insertion at the position *before* it, so reading
    the start line back out of the header needs that case handled."""
    new = PRE_IMAGE[:4] + ["    inserted\n"] + PRE_IMAGE[4:]
    lines = difflib.unified_diff(PRE_IMAGE, new, "a", "b", n=0)
    annotator = funcname.HunkAnnotator(PRE_IMAGE)
    headers = [ln.rstrip("\n") for ln in funcname.annotate_unified_diff(lines, annotator)
               if ln.startswith("@@")]
    assert headers == ["@@ -4,0 +5 @@ top_level_two"]


def test_the_line_terminator_of_the_input_is_preserved():
    """difflib's lineterm="" mode yields headers with no newline; adding one to
    just the labelled line would leave the diff inconsistent with itself."""
    new = PRE_IMAGE[:5] + ["    changed\n"] + PRE_IMAGE[6:]
    bare = [ln.rstrip("\n") for ln in difflib.unified_diff(PRE_IMAGE, new, "a", "b", n=0)]
    annotator = funcname.HunkAnnotator(PRE_IMAGE)
    out = [ln for ln in funcname.annotate_unified_diff(iter(bare), annotator)
           if ln.startswith("@@")]
    assert out == ["@@ -6 +6 @@ top_level_two"]


def test_lines_that_are_not_headers_pass_through_untouched():
    body = ["--- a\n", "+++ b\n", "-@@ not a header\n", " context\n"]
    annotator = funcname.HunkAnnotator(PRE_IMAGE)
    assert list(funcname.annotate_unified_diff(iter(body), annotator)) == body


# --- agreement with git ------------------------------------------------------

SAMPLES = {
    "python": ("m.py", "python", """\
class Widget:
    def render(self):
        a = 1
        b = 2
        c = 3
        d = 4
        return a

    async def resize(self, n):
        w = 1
        h = 2
        return w
"""),
    "shell": ("m.sh", "bash", """\
usage() {
    echo one
    echo two
    echo three
}

function verify () {
    test -n "$1"
    test -d "$2"
    echo done
}
"""),
    "markdown": ("m.md", "markdown", """\
# Title

intro one
intro two

## Section

body one
body two
body three
"""),
    "ruby": ("m.rb", "ruby", """\
class Widget
  def render
    a = 1
    b = 2
    c = 3
  end

  def resize
    w = 1
    h = 2
  end
end
"""),
    "rust": ("m.rs", "rust", """\
pub struct Widget {
    a: u32,
    b: u32,
}

pub fn render(w: &Widget) -> u32 {
    let x = 1;
    let y = 2;
    x + y
}
"""),
    "perl": ("m.pl", "perl", """\
package Widget;

sub render {
    my $a = 1;
    my $b = 2;
    return $a;
}

sub resize {
    my $w = 1;
    return $w;
}
"""),
}

_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)$")


def _our_headers(pre_image, post_image, driver, name, context):
    annotator = funcname.HunkAnnotator(pre_image, funcname.matcher_for(driver, name))
    lines = difflib.unified_diff(pre_image, post_image, "a", "b", n=context)
    return {ln[:ln.index(" @@")]: _HEADER.match(ln.rstrip("\n")).group(1)
            for ln in funcname.annotate_unified_diff(lines, annotator)
            if ln.startswith("@@")}


def _git_headers(tmp_path, name, git_driver, pre_image, post_image, context):
    def run(*args):
        return subprocess.run([GIT, *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True)

    run("init", "-q", ".")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "tester")
    (tmp_path / ".gitattributes").write_text(f"* diff={git_driver}\n")
    (tmp_path / name).write_text("".join(pre_image))
    run("add", "-A")
    run("commit", "-qm", "base")
    (tmp_path / name).write_text("".join(post_image))
    out = run("diff", f"-U{context}", "--", name).stdout
    headers = {}
    for line in out.split("\n"):
        match = _HEADER.match(line)
        if match:
            headers[line[:line.index(" @@")]] = match.group(1)
    return headers


def _normalise(label):
    """Drop a trailing block brace before comparing against git.

    git's own drivers do not agree with each other -- or with themselves across
    versions -- about whether ``usage() {`` keeps its brace, and the local git
    is whichever one is installed. Pinning that one character would make this
    test report the git version rather than the search rule.
    """
    return label.rstrip().removesuffix("{").rstrip()


@pytest.mark.skipif(GIT is None, reason="needs git to compare against")
@pytest.mark.parametrize("driver", sorted(SAMPLES))
@pytest.mark.parametrize("context", [0, 1, 3])
def test_labels_agree_with_git(tmp_path, driver, context):
    """Same content, same context width, same labels as `git diff` produces.

    Only hunks whose ranges match are compared: difflib and git's xdiff may
    group changes differently, and a grouping difference is not a labelling
    difference.
    """
    name, git_driver, text = SAMPLES[driver]
    pre_image = text.splitlines(keepends=True)
    post_image = list(pre_image)
    for i in (2, len(pre_image) - 2):
        post_image[i] = "        CHANGED\n"

    ours = _our_headers(pre_image, post_image, driver, name, context)
    theirs = _git_headers(tmp_path, name, git_driver, pre_image, post_image, context)

    shared = set(ours) & set(theirs)
    assert shared, f"no comparable hunks: ours={ours} git={theirs}"
    assert ({k: _normalise(ours[k]) for k in shared}
            == {k: _normalise(theirs[k]) for k in shared})
