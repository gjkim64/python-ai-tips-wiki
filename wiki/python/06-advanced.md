# 고급 주제

[← 모듈 · 패키지 · 가상환경 · 타입 힌트](05-modules-envs-typing.md) · [목차](../README.md) · 다음: [PyTorch: Dataset & DataLoader →](../pytorch/01-dataset-dataloader.md)

---

## 1. 메타프로그래밍 (Metaprogramming)

코드를 다루는 코드를 작성하는 기법으로, 파이썬에서 가장 강력하고 깊은 영역입니다.

### 데코레이터 (Decorators)

기존 함수나 클래스의 코드를 수정하지 않고 기능을 추가하거나 바꿉니다. `@classmethod`, `@staticmethod`, `@property` 도 데코레이터입니다.

```python
import functools, time

def timed(func):
    @functools.wraps(func)                 # 원래 함수의 이름/docstring 유지
    def wrapper(*args, **kwargs):          # 어떤 인자든 그대로 전달
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timed            # train = timed(train) 과 같음
def train(): ...
```

### 메타클래스 (Metaclasses)

클래스를 만드는 클래스입니다. 일반 객체가 클래스의 인스턴스이듯, 클래스는 메타클래스(기본값 `type`)의 인스턴스입니다.

```python
type(3)        # <class 'int'>
type(int)      # <class 'type'>
```

프레임워크를 설계할 때 클래스 생성 과정을 제어하는 데 씁니다(예: 하위 클래스 자동 등록). 대부분의 경우 더 단순한 `__init_subclass__` 나 클래스 데코레이터로 충분합니다.

### 디스크립터 프로토콜 (Descriptor Protocol)

속성에 접근할 때(`__get__`, `__set__`, `__delete__`)의 동작을 직접 정의합니다. `@property`, `@classmethod`, 그리고 함수가 메서드로 바인딩되는 것([OOP › 함수를 클래스 속성으로](03-oop.md#5-함수를-클래스-속성으로-저장할-수-있을까--가능)) 모두 이 프로토콜로 동작합니다.

## 2. 동시성 및 비동기 프로그래밍 (Concurrency & Async)

### GIL (Global Interpreter Lock)

CPython 은 한 번에 하나의 스레드만 파이썬 바이트코드를 실행합니다. 따라서:

| 작업 유형 | 해결책 |
|-----------|--------|
| **CPU 집중** (순수 파이썬 계산) | `multiprocessing` (멀티프로세싱) |
| **I/O 집중** (네트워크, 파일) | `threading` 또는 `asyncio` |
| **수치 연산** (NumPy, PyTorch) | 연산 자체가 C/CUDA 에서 GIL 을 풀고 실행되므로 큰 문제 없음 |

> PyTorch 의 `DataLoader(num_workers=4)` 는 데이터 로딩을 **여러 프로세스**로 병렬화합니다. GIL 을 피하기 위한 설계입니다.
> 참고: Python 3.13 부터 GIL 을 끈 실험적 free-threaded 빌드가 제공됩니다.

### asyncio

`async`/`await` 로 단일 스레드 안에서 I/O 작업을 멈춤(blocking) 없이 처리합니다. 웹 API 호출, 크롤러 속도 개선에 효과적입니다.

```python
import asyncio

async def fetch(i):
    await asyncio.sleep(1)       # I/O 대기 흉내
    return i

async def main():
    results = await asyncio.gather(*(fetch(i) for i in range(10)))  # 10개를 동시에 → 약 1초
    print(results)

asyncio.run(main())
```

### 제너레이터와 코루틴

`yield` 를 쓰면 실행을 일시 중지했다가 나중에 재개하는 함수가 됩니다. 값을 하나씩 **필요할 때 생성**하므로 메모리를 아낄 수 있고, 비동기 프로그래밍의 기반이 됩니다.

```python
def read_chunks(path, size=1024):
    with open(path, "rb") as f:
        while chunk := f.read(size):
            yield chunk
```

> **중요:** 제너레이터는 **한 번 쓰면 소진**됩니다.
> ```python
> g = (x * x for x in range(3))
> list(g)   # [0, 1, 4]
> list(g)   # []   ← 이미 비어 있음
> ```
> 여러 번 써야 한다면 처음부터 `list(...)` 로 저장하세요. `Path.rglob`, `map`, `filter`, `zip` 도 마찬가지입니다.

## 3. 고급 메모리 관리 및 최적화

### 가비지 컬렉션 (Garbage Collection)

파이썬은 두 가지 방식으로 메모리를 관리합니다.

1. **레퍼런스 카운팅**: 객체를 가리키는 참조가 0 이 되면 즉시 해제
2. **순환 참조 감지 GC**: `a.b = b; b.a = a` 처럼 서로를 가리켜 카운트가 0 이 안 되는 객체 묶음을 주기적으로 찾아 해제 (`gc` 모듈)

이 구조를 알아야 메모리 누수를 막을 수 있습니다.

> PyTorch 실전 팁: 학습 루프에서 `total_loss += loss` 처럼 텐서를 누적하면 연산 그래프 전체가 메모리에 계속 남습니다. 숫자만 필요하면 `total_loss += loss.item()` 을 쓰세요.

### `__slots__`

클래스에 `__slots__` 를 선언하면 인스턴스마다 생기는 `__dict__` 가 없어지고 고정된 공간만 사용합니다. 객체를 수백만 개 만들 때 메모리를 크게 줄일 수 있습니다.

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y
# Point(1, 2).z = 3  → AttributeError (정의되지 않은 속성 추가 불가)
```

## 4. 구조적 서브타이핑 (Structural Subtyping)

`typing.Protocol` 을 쓰면 상속 관계가 없어도 **특정 메서드를 가졌는지**로 타입을 검사합니다. 덕 타이핑([매직 메서드](04-magic-methods.md#2-덕-타이핑-상속-없이-그-타입처럼-동작하기))을 타입 검사기가 이해할 수 있게 표현한 것입니다.

```python
from typing import Protocol

class SizedIndexable(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int): ...

def first_and_count(x: SizedIndexable):
    return x[0], len(x)      # list, MyList, Dataset 모두 통과
```

## 5. 정적 분석 (Static Analysis)

**mypy** 는 타입 힌트를 바탕으로 코드를 **실행하기 전에** 타입 오류를 찾아냅니다.

```bash
pip install mypy
mypy my_project/
```

대형 프로젝트에서는 CI 파이프라인에 넣어 커밋마다 자동 검사하면 좋습니다.

---

## 🛠️ 무엇부터 시작하면 좋을까?

고급 주제를 공부할 때 추천하는 순서입니다.

1. **데코레이터와 매직 메서드를 직접 구현**해 보며 파이썬 객체가 내부적으로 어떻게 동작하는지 이해합니다.
   AI 코드에서 바로 쓰입니다: `Dataset`(`__len__`, `__getitem__`), `nn.Module`(`__call__`), `@torch.no_grad()`(데코레이터).
2. **asyncio** 로 비동기 프로그래밍을 익혀 웹 API 호출이나 크롤러의 속도를 개선해 봅니다.
3. 그다음 메타클래스, 디스크립터, 메모리 최적화, 정적 분석으로 넓혀 갑니다.
