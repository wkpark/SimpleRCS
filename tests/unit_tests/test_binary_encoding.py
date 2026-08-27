"""Binary payload storage: escaping is the container's rule, not an encoding.

Every value in a block doubles its '@'. That is what makes '@' parity a
decidable way to find field delimiters, so it has to hold for binary payloads
too -- they used to be the one exception, stored verbatim. The `<length>;<tag>,`
header now says only how the bytes were encoded (raw / base64 / base85) and
counts what is actually on disk, post-escape.

`raw` is the RCS-style option that escaping makes usable: keep the bytes, double
the '@'. ~0.4% overhead against base64's 33%, and it decodes faster.
"""

import io
import re

import pytest

from simple_rcs import codec
from simple_rcs.simple_rcs import SimpleRCS

ENCODINGS = ["raw", "base64", "base85"]

# Payloads shaped like the framing itself: markers, delimiters, block
# separators, and length headers. Only `raw` stores these bytes verbatim, so
# only `raw` can put them into the stream where a scan will meet them.
ADVERSARIAL = [
    b"",
    b"@",
    b"@" * 100,
    b"@;",
    b"\n\nver @@;",
    b"\n\nversion @1.0@;\ndate @x@;\n",
    b"binary @\n\nver @@;\xf4i\xb4",
    b"12;raw,99;base64,@@99;base64,",
    bytes(range(256)) * 4,
]


def _commit_all(rcs: SimpleRCS, blobs, encoding: str) -> None:
    for i, blob in enumerate(blobs):
        rcs.commit(blob, author="u@h", log=f"m{i} @; ver @x", encoding=encoding)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_every_version_round_trips_through_adversarial_payloads(encoding):
    rcs = SimpleRCS(io.BytesIO())
    _commit_all(rcs, ADVERSARIAL, encoding)
    raw = rcs.get_bytes()

    reopened = SimpleRCS(io.BytesIO(raw))
    assert [reopened.checkout(f"1.{i}") for i in range(len(ADVERSARIAL))] == ADVERSARIAL
    assert [e["ver"] for e in SimpleRCS(io.BytesIO(raw)).log()] == [f"1.{i}" for i in range(len(ADVERSARIAL))][::-1]


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_the_hash_chain_verifies_over_binary_blocks(encoding):
    rcs = SimpleRCS()  # v2: writes the header, so verify() actually runs
    assert rcs._version == 2
    _commit_all(rcs, ADVERSARIAL, encoding)

    raw = rcs.get_bytes()
    assert SimpleRCS(io.BytesIO(raw)).verify() is True
    assert SimpleRCS(io.BytesIO(raw)).checkout("1.0") == ADVERSARIAL[0]


def _odd_parity_hits(block: bytes) -> int:
    """'@;' preceded by an odd run of '@' -- 2k escaped plus one delimiter."""
    return sum(1 for m in re.finditer(rb"@+;", block) if (m.end() - m.start() - 1) % 2 == 1)


# A v1 first commit of binary content writes exactly these five fields.
# Counting them with a "^ver @"-style regex does not work here, and that is the
# point: the escaped payload below itself contains a line starting "ver @@",
# which a marker search takes for a field and parity does not.
_FIELDS_IN_A_FIRST_BINARY_BLOCK = 5  # ver, date, author, log, binary


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_at_parity_is_exact_for_every_encoding(encoding):
    """The property the escaping exists for. base85 used to break it: its
    alphabet has both '@' and ';', so an unescaped payload could spell a
    terminator. The leading group below encodes to base85 "000@;" precisely to
    reproduce that."""
    payload = b"\x00\x00\x19\xd9" + b"\xff@;\n\nver @content that looks like framing@;"
    rcs = SimpleRCS(io.BytesIO())
    rcs.commit(payload, author="u", log="a@;b / literal @@; pair / ver @fake", encoding=encoding)

    raw = rcs.get_bytes()
    assert _odd_parity_hits(raw[raw.find(b"ver @") :]) == _FIELDS_IN_A_FIRST_BINARY_BLOCK


def test_raw_stores_less_than_base64():
    payload = bytes(range(256)) * 40  # 10 KiB, '@' at the natural 1/256 rate
    sizes = {}
    for encoding in ("raw", "base64"):
        rcs = SimpleRCS(io.BytesIO())
        rcs.commit(payload, author="u", log="v1", encoding=encoding)
        sizes[encoding] = len(rcs.get_bytes())

    assert sizes["raw"] < len(payload) * 1.02, "escaping should cost ~0.4%, not a constant factor"
    assert sizes["base64"] > len(payload) * 1.3


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_the_header_length_counts_the_bytes_actually_stored(encoding):
    """What makes the forward seek-skip exact for every encoding: the count is
    taken after escaping, not before."""
    payload = bytes(range(256)) * 2
    stored = codec.encode_binary(payload, encoding)

    assert codec.binary_tag(stored) == encoding
    assert codec.decode_binary(stored) == payload
    length, _, rest = stored.partition(b",")
    assert int(length.split(b";")[0]) == len(rest)


def test_an_unknown_encoding_tag_is_rejected():
    with pytest.raises(ValueError, match="Unknown binary encoding"):
        codec.decode_binary(b"4;base99,AAAA")


def test_a_payload_containing_an_encoding_tag_is_not_misdispatched():
    """decode_binary used to search the whole value for ';base64,'. A raw
    payload can contain that itself."""
    payload = b"xx;base64,QUJD and ;base85, too"
    assert codec.decode_binary(codec.encode_binary(payload, "raw")) == payload


def test_binary_tag_ignores_an_encoding_tag_inside_a_text_delta():
    """A text delta's payload is user content. Matching ';base64,' anywhere in
    it took an ordinary text delta for a binary one."""
    text_delta = "a1 1\nsome prose mentioning 12;base64,AAA in passing\n"

    assert codec.binary_tag(text_delta) is None
    assert codec.binary_tag("12;base64,AAA") == "base64"


def test_get_bytes_is_lossless_where_get_content_is_not():
    rcs = SimpleRCS(io.BytesIO())
    rcs.commit(b"\xff\xfe\x00 binary that is not text\n", author="u", log="v1", encoding="raw")

    assert rcs.get_bytes() == rcs.stream.getvalue()
    # get_content() decodes with errors="replace", so it cannot round-trip this.
    assert rcs.get_content().encode(rcs.encoding) != rcs.get_bytes()


def test_base64_streams_are_byte_identical_to_the_unescaped_format():
    """base64's alphabet has no '@', so escaping is a no-op and its stored form
    did not change -- the reason existing base64 histories need no migration."""
    rcs = SimpleRCS(io.BytesIO())
    rcs.commit(bytes(range(256)) * 2, author="u", log="v1", encoding="base64")

    raw = rcs.get_bytes()
    start = raw.index(b"binary @") + len(b"binary @")
    header, _, payload = raw[start : raw.index(b"@;", start)].partition(b",")

    assert b"@" not in payload, "nothing to escape in a base64 payload"
    assert int(header.split(b";")[0]) == len(payload)


def _marker_run_length(raw: bytes, marker_start: int) -> int:
    """Length of the '@' run that the marker at `marker_start` opens."""
    at = raw.index(b"@", marker_start)
    end = at
    while end < len(raw) and raw[end : end + 1] == b"@":
        end += 1
    return end - at


def test_a_marker_inside_an_escaped_payload_is_not_taken_for_a_block():
    """`raw` stores bytes verbatim, so a payload can contain the block marker.
    Escaping makes the two cases tell themselves apart: a delimiter is a single
    '@', a doubled one inside a value is a run of two or more.

    The payload has to be HEAD -- an older block is folded into a bsdiff delta,
    which no longer holds the marker-shaped bytes.
    """
    poison = b"binary @\n\nver @@;\xf4i\xb4 payload"
    rcs = SimpleRCS(io.BytesIO())
    rcs.commit(b"first\n", author="u", log="v1", encoding="raw")
    rcs.commit(b"second\n", author="u", log="v2", encoding="raw")
    rcs.commit(poison, author="u", log="v3", encoding="raw")

    raw = rcs.get_bytes()
    runs = [_marker_run_length(raw, m.start() + 2) for m in re.finditer(rb"\n\n(ver|version) @", raw)]
    # Blocks 1.1 and 1.2 open with a blank line; 1.0 sits at _data_start.
    assert runs.count(1) == 2, "one delimiter per real block after the first"
    assert any(run > 1 for run in runs), "the payload must really contain a marker-shaped run"

    reopened = SimpleRCS(io.BytesIO(raw))
    assert reopened.head_info["ver"] == "1.2"
    assert reopened.checkout("1.2") == poison
    assert [e["ver"] for e in SimpleRCS(io.BytesIO(raw)).log()] == ["1.2", "1.1", "1.0"]
    assert SimpleRCS(io.BytesIO(raw)).checkout("1.0") == b"first\n"
