# 파서 벤치마크: no_regex vs regex, Cython 포팅 타게팅

상태: 조사 완료, 구현 미착수 (Cython 포팅 여부는 미결정)

## 배경

`_parse_block_content_no_regex()`(hot path, `_get_prev_block()`이 항상 호출)를
Cython(.pyx)으로 포팅하면 얼마나 빨라질지에 대한 질문에서 출발. 추측 대신
실측으로 타게팅 지점을 확인.

## 1. 2000버전 시나리오에서 파싱이 차지하는 비중

50줄 파일에 매 커밋마다 한 줄만 바꿔가며 2000회 `commit()` (델타 블록이 작음,
~200~300B). 최악 케이스인 `checkout("1.0")`(2000개 블록 전체를 역순으로 순회)
측정:

| 항목 | 시간 | 비중 |
|---|---|---|
| `checkout("1.0")` 전체 | 48.4ms | 100% |
| 그중 `_parse_block_content_no_regex` 순수 파싱 | 25.2ms | **52%** |
| 나머지 (RCS 델타 적용 등) | 23.2ms | 48% |

파싱이 checkout 시간의 절반을 차지 — 최적화 타게팅 방향 자체는 유효함.

## 2. no_regex vs regex, 블록 크기별 실측

같은 `_parse_block_content_no_regex()` / `_parse_block_content()`(regex, 레퍼런스
구현)를 블록 크기를 늘려가며 직접 호출 (`timeit`, 5000~20000회 반복 평균):

| 블록 크기 | no_regex | regex | regex/no_regex |
|---|---|---|---|
| ~300B (한 줄 델타) | 11.0us | 14.4us | 1.3x (오차범위, regex가 이길 때도 있음) |
| 1.6KB | 13.0us | 144.7us | 11.1x |
| 15KB | 33.4us | 621.8us | 18.6x |
| 92KB | 149.3us | 4161.3us | 27.9x |

**해석**: 작은 델타 블록(수백 바이트)에서는 둘이 사실상 동급 — regex는 C 엔진이라
짧은 문자열에서는 오히려 no_regex보다 빠르기도 함. 하지만 블록이 커질수록
`(?:[^@]|@@)*` alternation이 문자 단위로 재스캔되며 사실상 선형이 아니라 폭발적으로
느려짐. 스냅샷/바이너리/큰 파일 전체 텍스트 블록 기준으로는 no_regex가 10~28배 빠름.

→ no_regex를 hot path로 쓰는 기존 설계 결정은 "큰 블록" 기준으로 명확히 정당함.
regex 파서(`_parse_block_content`)는 hot path가 아니므로 Cython 대상에서 제외.

## 3. Cython 포팅 시 기대 효과 (추정)

2000버전/소형 델타 시나리오에서 no_regex 호출 1건(~12us)의 비용 대부분은 실제
바이트 비교 작업이 아니라 `while` 루프의 파이썬 바이트코드 디스패치 오버헤드
(`content[pos] in b" \t\r\n"` 같은 멤버십 체크를 바이트 단위로 반복). 이런 형태의
루프는 `char*`/포인터 연산으로 옮기는 Cython 포팅에서 통상 5~10배 단축이 현실적인
범위.

- 파싱: 25.2ms → 3~5ms (추정)
- checkout 전체: 48.4ms → 25~28ms, **약 1.7~2배** (상한선; 델타 적용 쪽 48%는
  그대로 남음)

## 결론

- 파싱은 2000버전 checkout 시간의 절반을 차지 — 최적화 가치 있는 타겟.
- regex→no_regex 전환은 이미 끝난 결정이고 큰 블록에서 압도적으로 유효함
  (재검토 불필요).
- Cython 포팅은 소형 델타 체인 기준 checkout 전체를 최대 2배 정도 개선하는
  수준 — 델타 적용 로직(BSDIFF/RCS diff apply)을 손대지 않는 한 그 이상은
  안 나옴. 빌드/배포 복잡도(플랫폼별 컴파일) 대비 이득이 맞는지는 별도 판단
  필요, 아직 미결정.

## 재현 방법

```python
import sys, timeit
sys.path.insert(0, "/share/repo/SimpleRCS")
from simple_rcs.simple_rcs import SimpleRCS

rcs = SimpleRCS()
lines = [f"line {i}" for i in range(50)]
for i in range(2000):
    lines[i % 50] = f"line {i % 50} rev{i}"
    rcs.commit("\n".join(lines) + "\n", author="a", log=f"rev{i}")

import time
t0 = time.perf_counter()
rcs.checkout("1.0")
print(time.perf_counter() - t0)
```

블록 크기별 비교는 `_parse_block_content_no_regex(block_bytes)` /
`_parse_block_content(block_bytes)`를 동일 블록에 대해 `timeit`으로 직접 비교.
