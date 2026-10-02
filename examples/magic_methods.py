"""wiki/python/03-oop.md, 04-magic-methods.md 예제. 표준 라이브러리만 사용.

실행: python examples/magic_methods.py
"""
import time
from contextlib import contextmanager


class MyList:
    """list 를 상속하지 않았지만 list 처럼 쓸 수 있는 클래스 (덕 타이핑)."""

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __repr__(self):
        return f"MyList({self._items})"


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        return False


@contextmanager
def timer(log):
    start = time.perf_counter()
    try:
        yield
    finally:
        log.append(time.perf_counter() - start)


def say_hello():
    return "Hello World!"


class Greeter:
    greet = say_hello


class StaticGreeter:
    greet = staticmethod(say_hello)


class Model:
    def __init__(self, lr):
        self.lr = lr

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if value <= 0:
            raise ValueError("learning rate 는 양수여야 함")
        self._lr = value


if __name__ == "__main__":
    ml = MyList([10, 20, 30])
    assert len(ml) == 3 and ml[1] == 20 and ml[0:2] == [10, 20]
    assert list(ml) == [10, 20, 30] and 20 in ml
    print("OK  MyList:", ml)

    with Timer() as t:
        sum(range(100_000))
    assert t.elapsed > 0
    log = []
    with timer(log):
        pass
    assert len(log) == 1
    print("OK  context managers")

    assert Greeter.greet() == "Hello World!"
    try:
        Greeter().greet()
    except TypeError as e:
        print("OK  instance call binds self ->", e)
    assert StaticGreeter().greet() == "Hello World!"
    print("OK  staticmethod avoids binding")

    m = Model(0.01)
    m.lr = 0.001
    try:
        m.lr = -1
    except ValueError:
        print("OK  @property setter validation")
