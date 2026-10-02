# 매직 메서드 (Magic Methods) & 덕 타이핑 (Duck Typing)

[← 객체지향 프로그래밍](03-oop.md) · [목차](../README.md) · 다음: [모듈 · 패키지 · 가상환경 · 타입 힌트 →](05-modules-envs-typing.md)

실행 가능한 예제: [`examples/magic_methods.py`](../../examples/magic_methods.py)

---

## 1. Dunder Methods 란?

`__init__`, `__str__`, `__len__`, `__call__` 처럼 앞뒤로 밑줄 두 개(**d**ouble **under**score)가 붙은 메서드입니다.
이를 구현하면 내가 만든 클래스가 파이썬 내장 기능(`+` 연산자, `len()`, `for` 반복, `[]` 인덱싱 등)과 자연스럽게 호환됩니다.

| 메서드 | 연결되는 문법 |
|--------|---------------|
| `__init__` | `Obj(...)` 생성 |
| `__repr__` / `__str__` | `repr(obj)` / `print(obj)` |
| `__len__` | `len(obj)` |
| `__getitem__` | `obj[i]`, 그리고 `for x in obj` (fallback) |
| `__iter__` / `__next__` | `for x in obj` |
| `__contains__` | `x in obj` |
| `__add__`, `__mul__`, ... | `a + b`, `a * b` |
| `__eq__`, `__lt__`, ... | `==`, `<` |
| `__call__` | `obj(...)` — 객체를 함수처럼 호출 |
| `__enter__` / `__exit__` | `with obj:` |

## 2. 덕 타이핑: 상속 없이 "그 타입처럼" 동작하기

> "오리처럼 걷고 오리처럼 꽥꽥거리면 오리다."

파이썬은 객체의 **타입이 아니라 가진 메서드**를 봅니다. 매직 메서드를 잘 골라 구현하면, 기존 자료형을 전부 상속하지 않고도 **필요한 부분만** 그 자료형처럼 동작하는 클래스를 만들 수 있습니다.

```python
class MyList:
    """list 를 상속하지 않았지만 list 처럼 쓸 수 있는 클래스."""
    def __init__(self, items):
        self._items = list(items)
    def __len__(self):
        return len(self._items)
    def __getitem__(self, idx):
        return self._items[idx]
    def __repr__(self):
        return f"MyList({self._items})"

ml = MyList([10, 20, 30])
len(ml)          # 3
ml[1]            # 20
ml[0:2]          # [10, 20]  (슬라이스도 그대로 전달됨)
list(ml)         # [10, 20, 30]  ← __getitem__ 만 있어도 반복 가능
20 in ml         # True
```

### 왜 중요한가: PyTorch `Dataset`

PyTorch 의 `Dataset` 이 바로 이 원리입니다. `__len__` 과 `__getitem__` 두 개만 구현하면 `DataLoader` 가 인덱스로 샘플을 꺼내 배치를 만들어 줍니다.

```python
class XXX(Dataset):
    def __init__(self, ...): ...     # 데이터 목록 준비
    def __len__(self): ...           # 전체 샘플 수
    def __getitem__(self, idx): ...  # idx 번째 (입력, 라벨) 반환
```

→ 자세한 내용: [Dataset & DataLoader](../pytorch/01-dataset-dataloader.md)

마찬가지로 `nn.Module` 은 `__call__` 을 구현해 두었기 때문에 `model(x)` 처럼 객체를 함수처럼 호출할 수 있고, 그 안에서 `forward` 가 실행됩니다. → [nn.Module](../pytorch/03-nn-module.md)

## 3. 컨텍스트 매니저 만들기

`with` 문과 함께 자원을 안전하게 열고 닫는 패턴입니다.

**클래스 방식** (`__enter__` / `__exit__`):

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                         # as 뒤의 변수에 들어감
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.elapsed:.3f}s")
        return False                        # False: 예외가 있으면 그대로 전파

with Timer():
    train_one_epoch()
```

**함수 방식** (`contextlib`):

```python
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield                      # with 블록 본문이 여기서 실행
    finally:
        print(f"{time.perf_counter() - start:.3f}s")
```
