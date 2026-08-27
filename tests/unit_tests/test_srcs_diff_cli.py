"""srcs_diff --binary, end to end.

test_gitpatch covers the patch format itself. What is only reachable through
the CLI is the wiring: which branch --binary takes, the bytes conversion when
one side is text and the other is not, the path written into the patch, and the
exit status. Those are exercised here by running the script, since nothing
imports from tools/.
"""

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
