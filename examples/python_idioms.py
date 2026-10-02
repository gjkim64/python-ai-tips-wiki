"""wiki/python/01-core-idioms.md 예제. 표준 라이브러리만 사용.

실행: python examples/python_idioms.py
"""
import copy
from collections import Counter, defaultdict, deque, namedtuple
from functools import reduce


def comprehensions():
    data = [("song1.wav", "jazz"), ("song2.wav", "rock"), ("song3.wav", "jazz")]
    entire_labels = [pair[1] for pair in data]
    assert entire_labels == ["jazz", "rock", "jazz"]

    classes = sorted(set(entire_labels))
    assert classes == ["jazz", "rock"]
    assert classes.index("rock") == 1

    class_to_idx = {c: i for i, c in enumerate(classes)}
    assert [class_to_idx[label] for label in entire_labels] == [0, 1, 0]


def higher_order():
    nums = [1, 2, 3, 4]
    assert list(map(lambda x: x * 2, nums)) == [2, 4, 6, 8]
    assert list(filter(lambda x: x % 2, nums)) == [1, 3]
    assert reduce(lambda acc, x: acc + x, nums) == 10

    year, month, day = map(int, "2025-08-22".split("-"))
    assert (year, month, day) == (2025, 8, 22)


def varargs(msg, *args, **kwargs):
    return msg, args, kwargs


def unpacking():
    a, b = 1, 2
    a, b = b, a
    assert (a, b) == (2, 1)

    first, *middle, last = [1, 2, 3, 4, 5]
    assert (first, middle, last) == (1, [2, 3, 4], 5)

    assert varargs("hi", 1, 2, level="debug") == ("hi", (1, 2), {"level": "debug"})


def collections_demo():
    Point = namedtuple("Point", "x y")
    assert Point(1, 2).y == 2

    groups = defaultdict(list)
    for name, genre in [("a", "jazz"), ("b", "rock"), ("c", "jazz")]:
        groups[genre].append(name)
    assert groups["jazz"] == ["a", "c"]

    assert Counter(["jazz", "rock", "jazz"]).most_common(1) == [("jazz", 2)]

    buf = deque(maxlen=3)
    for i in range(5):
        buf.append(i)
    assert list(buf) == [2, 3, 4]


def sorting():
    songs = [("b.wav", 3.2), ("a.wav", 1.5), ("c.wav", 2.0)]
    assert [s[0] for s in sorted(songs, key=lambda s: s[1])] == ["a.wav", "c.wav", "b.wav"]
    assert songs.sort() is None  # sort() 는 제자리 정렬, None 반환


def copies():
    a = [[1, 2], [3, 4]]
    b = a
    c = copy.copy(a)
    d = copy.deepcopy(a)
    a[0].append(99)
    assert b[0] == [1, 2, 99]
    assert c[0] == [1, 2, 99]   # 얕은 복사: 안쪽 리스트 공유
    assert d[0] == [1, 2]       # 깊은 복사: 독립


def fstrings():
    x, name = 3.14159, "acc"
    assert f"{x:.2f}" == "3.14"
    assert f"{name:>8}" == "     acc"
    assert f"{1234567:,}" == "1,234,567"
    assert f"{0.873:.1%}" == "87.3%"
    assert f"{x=}" == "x=3.14159"


def generators_exhaust():
    g = (x * x for x in range(3))
    assert list(g) == [0, 1, 4]
    assert list(g) == []        # 한 번 쓰면 소진


if __name__ == "__main__":
    for fn in [comprehensions, higher_order, unpacking, collections_demo,
               sorting, copies, fstrings, generators_exhaust]:
        fn()
        print(f"OK  {fn.__name__}")
