"""checkout() applies text deltas to one line list; it must give what applying them one by one gives."""

import random

import pytest

from simple_rcs.simple_rcs import SimpleRCS


def _step_by_step(rcs: SimpleRCS, ver: str) -> str | bytes:
    """checkout() as it was: every delta applied through _apply_reverse_delta on the full text."""
    rcs._load_head(force=True)
    block = rcs.head_info
    content = block["text"]
    while block["ver"] != ver:
        block = rcs._get_prev_block(block["start"])
        content = rcs._apply_reverse_delta(content, block["text"]) if block.get("is_delta", True) else block["text"]
    return content


def test_every_version_of_a_long_history_checks_out():
    rng = random.Random(20261031)  # noqa: S311 -- reproducible test data
    lines = [f"line {i} @ 한글\n" for i in range(60)]
    rcs = SimpleRCS()
    bodies = []
    for ver in range(300):
        for _ in range(rng.randint(1, 3)):
            at = rng.randrange(len(lines))
            op = rng.randrange(3)
            if op == 0:
                lines[at] = f"edit {ver} @@ {rng.random()}\n"
            elif op == 1:
                lines[at:at] = ["\n", f"insert {ver}\n"]
            elif len(lines) > 10:
                del lines[at : at + rng.randint(1, 3)]
        bodies.append("".join(lines))
        rcs.commit(bodies[-1], author="u", log=f"v{ver}", snapshot=ver > 0 and ver % 70 == 0)

    for ver, body in enumerate(bodies):
        assert rcs.checkout(f"1.{ver}") == body
    assert rcs.verify()


@pytest.mark.parametrize(
    "versions",
    [
        ["a\r\nb\r\nc\r\n", "a\r\nB\r\nc\r\n", "a\r\nB\r\nc\r\nd\r\n"],  # CRLF
        ["a\rb\nc\n", "a\rB\nc\n", "x\na\rB\nc\n"],  # lone CR inside the text
        ["a b\nc\n", "a b\nC\n", "a\nb\nC\n"],  # a line break only str.splitlines knows
        ["a\x0cb\nc\n", "b\nc\n", "a\x0cb\nc\nd\n"],  # form feed
        ["plain\nlines\n", "plain\r\nlines\n", "plain\nlines\nend"],  # break styles mixed across versions
    ],
)
def test_checkout_matches_step_by_step_application(versions):
    rcs = SimpleRCS()
    for i, text in enumerate(versions):
        rcs.commit(text, author="u", log=f"v{i}")

    for i in range(len(versions)):
        assert rcs.checkout(f"1.{i}") == _step_by_step(rcs, f"1.{i}")


def test_lone_cr_lines_keep_step_by_step_semantics():
    """A line ending in a lone CR, followed by an inserted blank line, rejoins into one CRLF line when the
    full text is split again. A line list kept across deltas would still hold two lines, and the next
    delta's line numbers would land on different lines."""
    raw = (
        "ver @1.0@;\ndate @2026-01-01T00:00:00@;\nauthor @u@;\nlog @a@;\ntext @d2 1\n@;\n\n"
        "ver @1.1@;\ndate @2026-01-01T00:00:01@;\nauthor @u@;\nlog @b@;\ntext @a1 1\n\n@;\n\n"
        "ver @1.2@;\ndate @2026-01-01T00:00:02@;\nauthor @u@;\nlog @c@;\ntext @a\rb\n@;\n\n"
    )
    rcs = SimpleRCS(raw)

    assert rcs.checkout("1.1") == _step_by_step(rcs, "1.1") == "a\r\nb\n"
    assert rcs.checkout("1.0") == _step_by_step(rcs, "1.0") == "a\r\n"


def test_text_history_under_a_binary_head_checks_out():
    rcs = SimpleRCS()
    rcs.commit("one\ntwo\n", author="u", log="text")
    rcs.commit("one\nthree\n", author="u", log="text")
    rcs.commit(b"\x00binary\xff", author="u", log="binary")

    for ver in ("1.0", "1.1", "1.2"):
        assert rcs.checkout(ver) == _step_by_step(rcs, ver)
    assert rcs.checkout("1.2") == b"\x00binary\xff"
