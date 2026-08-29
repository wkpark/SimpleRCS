"""srcs_diff, end to end.

test_gitpatch and test_funcname cover the formats and the rules themselves.
What is only reachable through the CLI is the wiring: which branch --binary
takes, the bytes conversion when one side is text and the other is not, the
path written into the patch, the exit status, and which flags switch hunk
labelling on. Those are exercised here by running the script, since nothing
imports from tools/.
"""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from simple_rcs.simple_rcs import SimpleRCS

TOOL = Path(__file__).parents[2] / "tools" / "srcs_diff.py"
GIT = shutil.which("git")

V1 = bytes(range(256)) * 2
V2 = V1[:300] + b"a replaced middle" + V1[340:] + b"and a tail"


def _history(tmp_path: Path, revisions, name: str = "data.bin") -> Path:
    """A .srcs where srcs_diff will look for it: <cwd>/.srcs/<name>.srcs."""
    srcs_dir = tmp_path / ".srcs"
    srcs_dir.mkdir(exist_ok=True)
    rcs = SimpleRCS(str(srcs_dir / f"{name}.srcs"))
    for i, content in enumerate(revisions):
        rcs.commit(content, author="tester", log=f"v{i}", encoding="raw")
    del rcs
    (tmp_path / name).write_bytes(revisions[-1] if isinstance(revisions[-1], bytes) else revisions[-1].encode())
    return tmp_path / name


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args], cwd=cwd, capture_output=True, text=True
    )


def test_binary_flag_writes_a_patch_and_exits_zero(tmp_path):
    """diff(1) would exit 1 for "they differ", which reads as failure when the
    patch is being redirected to a file."""
    _history(tmp_path, [V1, V2])

    result = _run(tmp_path, "data.bin", "--binary", "-r", "1.0:1.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("diff --git a/data.bin b/data.bin\n")
    assert "GIT binary patch" in result.stdout


def test_without_the_flag_the_exit_status_keeps_diff_semantics(tmp_path):
    _history(tmp_path, [V1, V2])

    result = _run(tmp_path, "data.bin", "-r", "1.0:1.1")

    assert result.returncode == 1
    assert "Binary files" in result.stdout
    assert "GIT binary patch" not in result.stdout


def test_the_patch_names_the_path_as_it_was_given(tmp_path):
    """git resolves a/<path> relative to where it runs, so the patch has to
    carry the path the caller used. The file sits in a subdirectory here on
    purpose: with a bare name, "as given" and "basename" are the same string
    and the branch would not be under test."""
    nested = tmp_path / "sub"
    nested.mkdir()
    _history(nested, [V1, V2])

    given = _run(tmp_path, "sub/data.bin", "--binary", "-r", "1.0:1.1", "--srcs-dir", "sub/.srcs")
    assert given.stdout.startswith("diff --git a/sub/data.bin b/sub/data.bin\n"), given.stderr

    # An absolute path cannot appear in a patch, so the bare name stands in.
    absolute = _run(
        tmp_path, str(nested / "data.bin"), "--binary", "-r", "1.0:1.1", "--srcs-dir", "sub/.srcs"
    )
    assert absolute.stdout.startswith("diff --git a/data.bin b/data.bin\n"), absolute.stderr


def test_text_revisions_still_get_a_unified_diff(tmp_path):
    """As `git diff --binary` behaves: the flag adds binary patches, it does not
    turn text diffs into them."""
    _history(tmp_path, ["one\ntwo\n", "one\ntwo modified\n"], name="notes.txt")

    result = _run(tmp_path, "notes.txt", "--binary", "-r", "1.0:1.1")

    assert "GIT binary patch" not in result.stdout
    assert "+two modified" in result.stdout


def test_a_history_that_changed_type_still_produces_a_patch(tmp_path):
    """One side text, the other bytes -- the branch converts before encoding."""
    _history(tmp_path, ["still text\n", V2])

    result = _run(tmp_path, "data.bin", "--binary", "-r", "1.0:1.1")

    assert result.returncode == 0, result.stderr
    assert "GIT binary patch" in result.stdout


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_git_applies_the_patch_the_cli_produced(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run([GIT, *args], cwd=repo, capture_output=True, text=True, check=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")

    target = _history(repo, [V1, V2])
    target.write_bytes(V1)  # the checked-in state is the patch's preimage
    git("add", "data.bin")
    git("commit", "-q", "-m", "base")

    result = _run(repo, "data.bin", "--binary", "-r", "1.0:1.1")
    assert result.returncode == 0, result.stderr
    patch = repo / "out.patch"
    patch.write_text(result.stdout)

    git("apply", "--check", "out.patch")
    git("apply", "out.patch")
    assert target.read_bytes() == V2

    git("apply", "-R", "out.patch")
    assert target.read_bytes() == V1


SOURCE_V1 = """\
class Widget:
    def render(self):
        a = 1
        b = 2
        c = 3
        d = 4
        return a

    def resize(self):
        w = 1
        return w
"""
SOURCE_V2 = SOURCE_V1.replace("c = 3", "c = 99").replace("w = 1", "w = 2")

#: The Cython matchers only exist once the extensions are built, so they join
#: the sweep when present rather than being permanently excluded from it.
ENGINES = ["difflib", "pydifflib", "myers"] + [
    name for name in ("ses", "dmp")
    if importlib.util.find_spec(f"simple_rcs._myersdiff_{name}") is not None
]


def _headers(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("@@")]


@pytest.mark.parametrize("engine", ENGINES)
def test_hunk_headers_are_unlabelled_by_default(tmp_path, engine):
    """The flag is opt-in: adding the feature must not rewrite the output of
    every existing caller."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-U0", "--engine", engine)

    assert _headers(result.stdout) == ["@@ -5 +5 @@", "@@ -10 +10 @@"], result.stderr


@pytest.mark.parametrize("engine", ENGINES)
def test_p_labels_every_hunk_on_every_engine(tmp_path, engine):
    """The label comes from the emit layer, not the diff algorithm, so the
    engine must not change it."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-U0", "-p", "--engine", engine)

    assert _headers(result.stdout) == [
        "@@ -5 +5 @@ def render(self):",
        "@@ -10 +10 @@ def resize(self):",
    ], result.stderr


def test_naming_a_driver_is_enough_to_turn_labelling_on(tmp_path):
    """Passing --funcname-driver and getting no labels would be a trap."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-U0", "--funcname-driver", "default")

    # The positional rule cannot see an indented def, so both hunks fall back
    # to the class -- which is exactly how it differs from the python driver.
    assert _headers(result.stdout) == [
        "@@ -5 +5 @@ class Widget:",
        "@@ -10 +10 @@ class Widget:",
    ], result.stderr


def test_a_custom_regex_replaces_the_driver(tmp_path):
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-U0", "-F", r"^(class .*)$")

    assert _headers(result.stdout) == [
        "@@ -5 +5 @@ class Widget:",
        "@@ -10 +10 @@ class Widget:",
    ], result.stderr


def test_a_broken_regex_is_reported_rather_than_traced(tmp_path):
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-F", "(unclosed")

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "Error:" in result.stderr


def test_context_width_is_settable(tmp_path):
    """-U is what makes labelling worth having: at the default width the two
    changes here merge into one hunk and only the first label is ever shown."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    wide = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-U2", "-p")

    assert _headers(wide.stdout) == ["@@ -3,9 +3,9 @@ def render(self):"], wide.stderr


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_the_labels_match_what_git_prints_for_the_same_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run([GIT, *args], cwd=repo, capture_output=True, text=True, check=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / ".gitattributes").write_text("*.py diff=python\n")

    target = _history(repo, [SOURCE_V1, SOURCE_V2], name="m.py")
    target.write_text(SOURCE_V1)
    git("add", "m.py", ".gitattributes")
    git("commit", "-q", "-m", "base")
    target.write_text(SOURCE_V2)

    theirs = _headers(git("diff", "-U0", "--", "m.py").stdout)
    ours = _headers(_run(repo, "m.py", "-r", "1.0:1.1", "-U0", "-p").stdout)

    assert ours == theirs


def test_a_negative_context_width_is_rejected(tmp_path):
    """It produced `@@ -6,-1 +6,-1 @@` -- which no patch tool accepts -- and
    still exited 0, so a redirected diff looked like it had succeeded."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "--unified=-1")

    assert result.returncode == 2
    assert "must not be negative" in result.stderr
    assert "@@" not in result.stdout


def test_asking_for_the_default_driver_by_name_still_labels(tmp_path):
    """--funcname-driver says it implies -p; naming its own default must not be
    the one spelling that silently does nothing."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-U0", "--funcname-driver", "auto")

    assert _headers(result.stdout) == [
        "@@ -5 +5 @@ def render(self):",
        "@@ -10 +10 @@ def resize(self):",
    ], result.stderr


# --------------------------------------------------------------------------
# colour
#
# test_color pins the escapes themselves against real git. What is only
# reachable here is the switch: that piping never colours, that every engine
# goes through the colouriser, and that --binary stays machine-readable.
# --------------------------------------------------------------------------

ESC = "\033["


def test_a_piped_diff_is_not_coloured(tmp_path):
    """subprocess gives the CLI a pipe, which is what --color=auto is for."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1")

    assert ESC not in result.stdout


@pytest.mark.parametrize("engine", ENGINES)
def test_color_always_colours_every_engine(tmp_path, engine):
    """The colouriser wraps one shared writer, so an engine that bypassed it
    would be the only plain one -- easy to add and easy to miss."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "--color=always", "--engine", engine)

    assert result.returncode in (0, 1), result.stderr
    assert "\033[36m@@" in result.stdout
    assert "\033[31m-" in result.stdout
    assert "\033[32m+" in result.stdout


def test_color_with_no_value_means_always(tmp_path):
    """git spells it that way, and a bare --color that did nothing over a pipe
    would look broken."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    assert ESC in _run(tmp_path, "m.py", "-r", "1.0:1.1", "--color").stdout


@pytest.mark.parametrize("flag", ["--no-color", "--color=never"])
def test_colour_can_be_turned_off_explicitly(tmp_path, flag):
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    assert ESC not in _run(tmp_path, "m.py", "-r", "1.0:1.1", flag).stdout


def test_an_unknown_colour_mode_is_rejected(tmp_path):
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    result = _run(tmp_path, "m.py", "-r", "1.0:1.1", "--color=sometimes")

    assert result.returncode == 2
    assert ESC not in result.stdout


def test_a_binary_patch_is_never_coloured(tmp_path):
    """It exists to be redirected into `git apply`; escapes would corrupt it."""
    _history(tmp_path, [V1, V2])

    result = _run(tmp_path, "data.bin", "--binary", "-r", "1.0:1.1", "--color=always")

    assert result.returncode == 0, result.stderr
    assert ESC not in result.stdout


def test_turning_colour_on_does_not_change_the_diff_itself(tmp_path):
    """Stripping the escapes back out must give the uncoloured output, or the
    colouriser is rewriting content rather than wrapping it."""
    _history(tmp_path, [SOURCE_V1, SOURCE_V2], name="m.py")

    plain = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-p").stdout
    painted = _run(tmp_path, "m.py", "-r", "1.0:1.1", "-p", "--color=always").stdout

    assert re.sub(r"\033\[[0-9;]*m", "", painted) == plain
