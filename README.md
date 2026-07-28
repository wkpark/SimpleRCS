# SimpleRCS

**SimpleRCS** is a custom-designed, stream-based version control system for
ultra-lightweight management of a single document. It borrows the reverse-delta
storage idea from classic RCS and rebuilds it around modern performance
considerations — stream-centric I/O, O(1) HEAD access, backward-scanning
history traversal — for cases where Git's object model is overkill, or where
direct file-system/database blob integration is preferred.

It operates on a file-like object (`BinaryIO`): an on-disk path, an in-memory
`BytesIO`, or an externally-owned handle all work the same way.

## Key Features

- **Reverse delta storage.** The latest version (HEAD) is always stored as
  full text at the end of the stream; every older version is a reverse delta
  (instructions to turn version *N+1* back into version *N*). HEAD reads are
  O(1); historical reads are O(k), where k is the distance from HEAD.
- **Append-dominant writes.** Committing overwrites only the previous HEAD
  block (full text → delta) and appends the new HEAD. No full-history
  rewrite, ever.
- **Backward-scanning, stream-centric I/O.** History is located by scanning
  from the end of the stream, without loading the whole file into memory —
  scalable to long histories.
- **Text and binary content, uniformly.** Text uses an RCS-style (`diff -n`)
  line delta; binary content uses a BSDIFF40-compatible patch format
  (interoperable with the native `bsdiff`/`bspatch` tools). `commit()` accepts
  `str`, `bytes`, or `BinaryIO` directly.
- **v2 hash chain & signatures.** Each block carries a hash of its logical
  content plus the previous block's hash, making tampering with any past
  version detectable. Optional GPG-based multi-signer signing/verification is
  built in (`simple_rcs_gpg`).
- **Intermediate snapshots.** `commit(..., snapshot=True)` stores the previous
  HEAD as full text instead of a delta, breaking the delta chain at that point
  for faster retrieval of that region of history.
- **`log()` metadata-only path.** History listing parses block metadata
  directly off the stream without decoding delta/binary payloads.

## Architecture

```
[V1 Delta] [V2 Delta] ... [Vn-1 Delta] [Vn Full Text]
                                         ^^^^^^^^^^^^^
                                         HEAD (always full text)
```

Committing a new version:

1. Read the current HEAD (full text).
2. Compute a reverse delta (New → Old HEAD).
3. Overwrite the on-disk HEAD block with that delta.
4. Append the new content as the new HEAD block (full text).

Checking out an old version walks backward from HEAD, applying each block's
reverse delta (or jumping directly to a snapshot's full text) until the
target version is reached.

## Comparison

Updated from the original design comparison (see [PR #2](https://github.com/wkpark/SimpleRCS/pull/2)) to reflect the current implementation:

| Feature | SimpleRCS | RCS (Original) | Git | DB (Simple Snapshot) |
| :--- | :--- | :--- | :--- | :--- |
| Storage unit | Single file (stream) | Single file (`*,v`) | Object DB (`.git`) + working tree | DB row |
| Delta direction | Reverse delta (HEAD = full text) | Reverse delta (HEAD = full text) | Snapshot (delta generated at pack time) | Full copy (no delta) |
| Storage pattern | Append-dominant (tail rewrite + append) | Header update + insert | Content-addressable storage | Insert new row |
| I/O pattern | Efficient tail write; single-pass backward scan (no re-read) for HEAD lookup | Random write (header/body) | Many small files / packfiles | DB transaction |
| Content types | Text (RCS diff) **and** binary (BSDIFF40-compatible), same file format | Text only (binary needs external encoding) | Any (blob-based) | Any (column type) |
| Integrity | v2: per-block hash chain + optional multi-signer GPG signatures | None | Content-addressable (SHA-1/256 of objects) | Depends on DB |
| Intermediate snapshots | Optional, per-commit (`snapshot=True`) | Not supported | N/A (every commit is already a full snapshot object) | N/A |
| HEAD lookup | **O(1)**, single-pass (read from end) | O(1) (read from start) | O(1) (ref → commit → tree) | O(1) (select by ID) |
| History lookup | **O(k)**, backward scan (k = distance from HEAD) | O(N) forward scan | O(log N) DAG traversal | O(1) (select with limit) |
| Diff engine | Greedy hash-matching (`StreamSequenceMatcher`, O(N)); Cython-accelerated Myers SES/DMP available for benchmarking | Standard `diff` | `xdiff` (Myers-based) | N/A |
| Complexity | Low (single class, no object graph) | Medium (custom grammar) | High (complex object model) | Low (SQL) |

**Note on the diff engine:** the production `commit()`/`checkout()` path uses a
fast, greedy O(N) hash-matching algorithm (`StreamSequenceMatcher`) that
favors speed over minimal edit distance. The repository also ships
Cython-accelerated Myers SES/DMP implementations (15–26x faster than their
pure-Python equivalents) that guarantee a shortest edit script; they are
currently exercised through `tools/bench_diff.py` for comparison rather than
wired into the default commit path.

## Usage

```python
from simple_rcs.simple_rcs import SimpleRCS

# In-memory, or pass a file path to persist to disk
rcs = SimpleRCS("my_document.srcs")

v1 = rcs.commit("Hello\nWorld\n", author="alice", log="initial")
v2 = rcs.commit("Hello\nSimpleRCS\n", author="bob", log="tweak line 2")

rcs.checkout(v1)          # -> "Hello\nWorld\n"
rcs.checkout()            # HEAD, i.e. v2

rcs.log()                 # [{'ver': '1.1', 'author': 'bob', ...}, {'ver': '1.0', ...}]
rcs.diff(v1, v2)           # unified diff between two versions
rcs.blame()                 # per-line author/version attribution for HEAD
rcs.verify()               # walks the v2 hash chain, returns True/False

# Binary content works the same way
rcs.commit(open("image.png", "rb").read(), author="alice")
```

Command-line helpers live in `tools/`: `srcs_log.py` (history/signature
listing), `srcs_diff.py` (unified diff between versions or engines),
`srcs_sign_head.py` (GPG-sign the current HEAD), and `bench_diff.py`
(diff-engine benchmark harness).

## Testing

```
uv run pytest tests/unit_tests/ -q
```

61 tests covering text/binary round-trips, hash-chain verification, snapshot
handling, and `log()`'s metadata-only parsing path.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

# SimpleRCS (한글)

**SimpleRCS**는 단일 문서를 초경량으로 관리하기 위해 새로 설계한 스트림 기반
버전 관리 시스템이다. 전통적인 RCS의 역방향 델타 저장 방식을 차용하되, 스트림
중심 I/O, O(1) HEAD 접근, 백워드 스캔 기반 히스토리 순회 등 현대적인 성능
고려사항으로 다시 설계했다. Git의 객체 모델이 과할 때, 혹은 파일시스템/DB
블롭에 직접 통합하고 싶을 때 적합하다.

파일과 유사한 객체(`BinaryIO`) 위에서 동작한다 — 디스크 경로, 인메모리
`BytesIO`, 외부에서 넘겨받은 핸들 모두 동일하게 동작한다.

## 핵심 특징

- **역방향 델타 저장.** 최신 버전(HEAD)은 항상 스트림 끝에 전체 텍스트로
  저장되고, 이전 버전들은 역방향 델타(버전 N+1을 N으로 되돌리는 명령)로
  저장된다. HEAD 읽기는 O(1), 과거 버전 읽기는 O(k)(k = HEAD로부터의 거리).
- **Append 위주 쓰기.** 커밋 시 기존 HEAD 블록만 덮어쓰고(전체 텍스트→델타),
  새 HEAD는 끝에 추가(append)된다. 전체 히스토리를 다시 쓰는 일은 없다.
- **백워드 스캔, 스트림 중심 I/O.** 히스토리는 스트림 끝에서부터 스캔해서
  찾으며, 전체 파일을 메모리에 올리지 않는다 — 긴 히스토리에도 확장 가능.
- **텍스트/바이너리 콘텐츠를 동일한 인터페이스로.** 텍스트는 RCS 스타일
  (`diff -n`) 라인 델타, 바이너리는 BSDIFF40 호환 패치 포맷(네이티브
  `bsdiff`/`bspatch`와 상호운용 가능)을 사용한다. `commit()`은 `str`,
  `bytes`, `BinaryIO`를 그대로 받는다.
- **v2 해시 체인 & 서명.** 각 블록은 논리적 콘텐츠의 해시와 이전 블록의
  해시를 함께 가지므로, 과거 어느 버전이든 변조하면 감지된다. GPG 기반
  다중 서명자 서명/검증도 내장(`simple_rcs_gpg`).
- **중간 스냅샷.** `commit(..., snapshot=True)`로 커밋하면 직전 HEAD를
  델타 대신 전체 텍스트로 저장해, 그 지점에서 델타 체인을 끊고 해당 구간의
  과거 버전 조회를 빠르게 만든다.
- **`log()` 메타데이터 전용 경로.** 히스토리 목록 조회 시 델타/바이너리
  본문을 디코딩하지 않고 스트림에서 메타데이터만 직접 파싱한다.

## 비교 분석

[PR #2](https://github.com/wkpark/SimpleRCS/pull/2)의 최초 비교표를 현재
구현 기준으로 갱신했다:

| 항목 | SimpleRCS | RCS (원본) | Git | DB (단순 스냅샷) |
| :--- | :--- | :--- | :--- | :--- |
| 저장 단위 | 단일 파일(스트림) | 단일 파일(`*,v`) | 객체 DB(`.git`) + 작업 트리 | DB 레코드(Row) |
| 델타 방향 | 역방향 델타(Head=전체 텍스트) | 역방향 델타(Head=전체 텍스트) | 스냅샷(패킹 시 델타 생성) | 전체 복사(델타 없음) |
| 저장 구조 | Append 위주(꼬리 덮어쓰기 + append) | 헤더 갱신 + 삽입 | 콘텐츠 주소 기반 저장소 | 새 행 삽입 |
| I/O 패턴 | 효율적인 꼬리 쓰기, HEAD 조회는 재읽기 없는 단일 패스 백워드 스캔 | 랜덤 쓰기(헤더/본문) | 많은 작은 파일 / packfile | DB 트랜잭션 |
| 콘텐츠 종류 | 텍스트(RCS diff)와 바이너리(BSDIFF40 호환)를 같은 파일 포맷으로 지원 | 텍스트 전용(바이너리는 별도 인코딩 필요) | 모든 타입(blob 기반) | 모든 타입(컬럼 타입) |
| 무결성 | v2: 블록별 해시 체인 + 선택적 다중 서명자 GPG 서명 | 없음 | 콘텐츠 주소 기반(객체 SHA-1/256) | DB에 따라 다름 |
| 중간 스냅샷 | 커밋별 선택 가능(`snapshot=True`) | 미지원 | 해당 없음(모든 커밋이 이미 완전한 스냅샷 객체) | 해당 없음 |
| HEAD 조회 | **O(1)**, 단일 패스(끝에서부터 읽기) | O(1)(처음부터 읽기) | O(1)(ref → commit → tree) | O(1)(ID로 select) |
| 히스토리 조회 | **O(k)**, 백워드 스캔(k = HEAD로부터의 거리) | O(N) 순방향 스캔 | O(log N) DAG 순회 | O(1)(limit으로 select) |
| Diff 엔진 | 그리디 해시 매칭(`StreamSequenceMatcher`, O(N)); Cython 가속 Myers SES/DMP는 벤치마크용으로 별도 제공 | 표준 `diff` | `xdiff`(Myers 기반) | 해당 없음 |
| 복잡도 | 낮음(단일 클래스, 객체 그래프 없음) | 중간(전용 문법) | 높음(복잡한 객체 모델) | 낮음(SQL) |

**Diff 엔진 관련 참고:** 실제 `commit()`/`checkout()` 경로는 최소 편집
거리보다 속도를 우선하는 빠른 그리디 O(N) 해시 매칭 알고리즘
(`StreamSequenceMatcher`)을 사용한다. 저장소에는 최단 편집 스크립트를
보장하는 Cython 가속 Myers SES/DMP 구현(순수 Python 대비 15~26배 빠름)도
포함되어 있지만, 현재는 기본 커밋 경로가 아니라 `tools/bench_diff.py`를
통한 비교/벤치마크 용도로만 쓰인다.

## 사용법

```python
from simple_rcs.simple_rcs import SimpleRCS

# 인메모리로 쓰거나, 경로를 넘기면 디스크에 저장됨
rcs = SimpleRCS("my_document.srcs")

v1 = rcs.commit("Hello\nWorld\n", author="alice", log="initial")
v2 = rcs.commit("Hello\nSimpleRCS\n", author="bob", log="tweak line 2")

rcs.checkout(v1)          # -> "Hello\nWorld\n"
rcs.checkout()            # HEAD, 즉 v2

rcs.log()                 # [{'ver': '1.1', 'author': 'bob', ...}, {'ver': '1.0', ...}]
rcs.diff(v1, v2)           # 두 버전 간 unified diff
rcs.blame()                 # HEAD의 라인별 저자/버전 귀속
rcs.verify()               # v2 해시 체인 검증, True/False 반환

# 바이너리 콘텐츠도 동일하게 동작
rcs.commit(open("image.png", "rb").read(), author="alice")
```

명령줄 도구는 `tools/`에 있다: `srcs_log.py`(히스토리/서명 목록),
`srcs_diff.py`(버전/엔진 간 unified diff), `srcs_sign_head.py`(현재
HEAD를 GPG로 서명), `bench_diff.py`(diff 엔진 벤치마크).

## 테스트

```
uv run pytest tests/unit_tests/ -q
```

텍스트/바이너리 라운드트립, 해시 체인 검증, 스냅샷 처리, `log()`의 메타데이터
전용 파싱 경로까지 총 61개 테스트로 커버한다.

## 라이선스

Apache License 2.0 — [LICENSE](LICENSE) 참고.
