# 핵심 관용구 (Core Idioms)

[← 목차](../README.md) · 다음: [예외 · 컨텍스트 매니저 · 파일 시스템 →](02-errors-context-files.md)

데이터를 짧고 읽기 쉽게 처리하는 파이썬 기본기입니다. 실행 가능한 예제는 [`examples/python_idioms.py`](../../examples/python_idioms.py) 에 있습니다.

---

## 1. 컴프리헨션 (Comprehensions)

리스트·딕셔너리·세트를 한 줄로 만듭니다. 조건문도 넣을 수 있습니다.

```python
squares   = [x * x for x in range(10)]                 # list
evens     = [x for x in range(10) if x % 2 == 0]       # 필터 조건
labels    = ["even" if x % 2 == 0 else "odd" for x in range(5)]  # 값 선택 (if-else 는 앞쪽에)
sq_map    = {x: x * x for x in range(5)}               # dict
uniq_len  = {len(w) for w in ["a", "bb", "cc"]}        # set
```

> **팁 — 일반 패턴:** `x = [fn(i) for i in something]`
> 텐서도 같은 방식으로 만들 수 있습니다: `torch.stack([fn(i) for i in items])`.
> 하지만 텐서에서는 반복문보다 **배열 전체 연산**이 훨씬 빠릅니다 ([텐서 연산](../pytorch/02-tensors-and-shapes.md) 참고).

**쌍(pair) 데이터에서 한쪽만 뽑아내기:**

```python
data = [("song1.wav", "jazz"), ("song2.wav", "rock"), ("song3.wav", "jazz")]
entire_labels = [pair[1] for pair in data]   # 각 쌍의 두 번째 요소(index 1) → ['jazz', 'rock', 'jazz']
```

## 2. lambda 와 내장 고차 함수

```python
from functools import reduce

nums    = [1, 2, 3, 4]
doubled = list(map(lambda x: x * 2, nums))         # [2, 4, 6, 8]
odds    = list(filter(lambda x: x % 2, nums))      # [1, 3]
total   = reduce(lambda acc, x: acc + x, nums)     # 10 reduce applies cumulatively
```

**`map` 을 이용한 문자열 파싱 + 언패킹:**

```python
date_string = "2025-08-22"
year, month, day = map(int, date_string.split("-"))   # 2025, 8, 22 (모두 int)
```

> `map`/`filter` 는 이터레이터를 반환합니다. 한 번 소비하면 비워지므로, 여러 번 쓰려면 `list(...)` 로 저장하세요 ([제너레이터](06-advanced.md#제너레이터와-코루틴) 참고).

## 3. 가변 인자: `*args`, `**kwargs`

```python
def log(msg, *args, **kwargs):
    print(msg, args, kwargs)

log("hi", 1, 2, level="debug")   # hi (1, 2) {'level': 'debug'}
```

- `*args` → 위치 인자를 **튜플**로 받음
- `**kwargs` → 키워드 인자를 **딕셔너리**로 받음
- 래퍼 함수나 데코레이터에서 인자를 그대로 넘길 때 특히 유용: `func(*args, **kwargs)`

## 4. 패킹과 언패킹

```python
a, b = 1, 2
a, b = b, a                    # 스왑 (임시 변수 불필요)

first, *middle, last = [1, 2, 3, 4, 5]   # first=1, middle=[2, 3, 4], last=5

def f(x, y, z): ...
f(*[1, 2, 3])                  # 리스트를 위치 인자로 펼침
f(**{"x": 1, "y": 2, "z": 3})  # 딕셔너리를 키워드 인자로 펼침
merged = {**d1, **d2}          # 딕셔너리 병합
```

## 5. `collections` 모듈 — 상황별 자료구조

| 자료구조 | 용도 | 예 |
|----------|------|----|
| `namedtuple` | 필드 이름이 있는 불변 튜플 | `Point = namedtuple("Point", "x y")` |
| `defaultdict` | 없는 키에 기본값 자동 생성 | `groups = defaultdict(list)` |
| `Counter` | 개수 세기 | `Counter(labels).most_common(3)` |
| `deque` | 양쪽 끝 삽입/삭제 O(1), 고정 길이 버퍼 | `deque(maxlen=100)` |

## 6. 정렬 심화

- `list.sort()` → **제자리(in-place)** 정렬, `None` 반환, 리스트에만 존재
- `sorted(iterable)` → **새 리스트** 반환, 모든 이터러블에 사용 가능

```python
songs = [("b.wav", 3.2), ("a.wav", 1.5), ("c.wav", 2.0)]
by_len   = sorted(songs, key=lambda s: s[1])                 # 길이순
by_multi = sorted(songs, key=lambda s: (-s[1], s[0]))        # 길이 내림차순, 같으면 이름순
```

**라벨 → 클래스 번호 만들기** (분류 모델에서 자주 쓰는 패턴):

```python
unique_labels = set(entire_labels)        # 중복 제거
classes = sorted(unique_labels)           # 라벨 자체를 정렬 → 순서 고정
classes.index("rock")                     # 'rock' 이 몇 번째인지 → 신경망 분류 번호로 사용
```

> `set` 은 순서가 보장되지 않으므로 반드시 `sorted` 로 순서를 고정해야 실행할 때마다 같은 번호가 나옵니다.
> 클래스가 많다면 `list.index` (O(n)) 대신 딕셔너리 `{c: i for i, c in enumerate(classes)}` 를 만들어 쓰세요.

## 7. 얕은 복사 vs 깊은 복사

```python
import copy

a = [[1, 2], [3, 4]]
b = a                  # 같은 객체를 가리킴 (복사 아님)
c = copy.copy(a)       # 얕은 복사: 바깥 리스트만 새로, 안쪽 리스트는 공유
d = copy.deepcopy(a)   # 깊은 복사: 안쪽까지 전부 새로 → 완전히 독립

a[0].append(99)
# b[0] == [1, 2, 99], c[0] == [1, 2, 99], d[0] == [1, 2]
```

> PyTorch 에서도 같은 함정이 있습니다. 텐서 슬라이싱(`t[0]`)이나 `view` 는 메모리를 공유하므로, 독립된 사본이 필요하면 `t.clone()` 을 쓰세요.

## 8. f-string 고급 기능

```python
from datetime import date
x, name = 3.14159, "acc"

f"{x:.2f}"          # '3.14'      소수점 제한
f"{name:>8}"        # '     acc'  오른쪽 정렬 (<: 왼쪽, ^: 가운데)
f"{1234567:,}"      # '1,234,567' 천 단위 구분
f"{0.873:.1%}"      # '87.3%'     퍼센트
f"{date.today():%Y-%m-%d}"   # 날짜 포맷
f"{x=}"             # 'x=3.14159' 디버깅용 (변수명 함께 출력)
```

## 9. 사용하지 않는 반복 변수: `_`

```python
for _ in range(3):   # 인덱스를 쓰지 않을 때 관례적으로 _ 사용
    train_one_epoch()
```
