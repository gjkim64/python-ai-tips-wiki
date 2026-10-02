# 객체지향 프로그래밍 (OOP) 기초

[← 예외 · 컨텍스트 매니저 · 파일 시스템](02-errors-context-files.md) · [목차](../README.md) · 다음: [매직 메서드 & 덕 타이핑 →](04-magic-methods.md)

---

## 1. 클래스와 인스턴스

```python
class Track:
    sample_rate = 22050             # 클래스 변수: 모든 인스턴스가 공유

    def __init__(self, path, genre):  # 생성자
        self.path = path            # 인스턴스 변수: 객체마다 따로
        self.genre = genre

t = Track("a.wav", "jazz")
```

> **주의:** 리스트 같은 가변 객체를 클래스 변수로 두면 모든 인스턴스가 같은 리스트를 공유합니다. 객체별 데이터는 `__init__` 에서 `self.xxx = []` 로 만드세요.  if you want certain list to be constant across instantiated instances perhaps use immutable object 

## 2. 캡슐화: `@property`

Getter/Setter 메서드 대신, 속성처럼 접근하면서 검증 로직을 넣을 수 있습니다.

```python
class Model:
    def __init__(self, lr):
        self.lr = lr                  # setter 를 거침

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        if value <= 0:
            raise ValueError("learning rate 는 양수여야 함")
        self._lr = value

m = Model(0.01)
m.lr = 0.001     # 메서드 호출 없이 속성처럼 사용
```

## 3. `@classmethod` 와 `@staticmethod`

| 종류 | 첫 인자 | 언제 쓰나 |
|------|---------|-----------|
| 인스턴스 메서드 | `self` | 객체 상태를 읽거나 바꿀 때 |
| `@classmethod` | `cls` | **대체 생성자**(`from_xxx`) 등 클래스 자체를 다룰 때 |
| `@staticmethod` | 없음 | 클래스와 관련은 있지만 상태가 필요 없는 유틸 함수 |

```python
class Track:
    def __init__(self, path, genre):
        self.path, self.genre = path, genre

    @classmethod
    def from_path(cls, p):            # 대체 생성자: 경로에서 장르 추출
        return cls(p, p.parent.name)

    @staticmethod
    def is_audio(name):
        return name.endswith((".wav", ".mp3"))
```

## 4. 상속과 오버라이딩, `super()`

```python
class Base:
    def __init__(self, name):
        self.name = name
    def describe(self):
        return f"I am {self.name}"

class Child(Base):
    def __init__(self, name, extra):
        super().__init__(name)        # 부모 생성자 호출
        self.extra = extra
    def describe(self):               # 오버라이딩
        return super().describe() + f" (+{self.extra})"
```

> PyTorch 의 모든 모델은 `nn.Module` 을 상속하고, `__init__` 첫 줄에서 반드시 `super().__init__()` 을 호출합니다. → [nn.Module](../pytorch/03-nn-module.md)

## 5. 함수를 클래스 속성으로 저장할 수 있을까? — 가능

파이썬에서 함수도 객체이므로 클래스 변수에 담을 수 있습니다.

```python
def say_hello():
    return "Hello World!"

class Greeter:
    greet = say_hello            # 함수를 클래스 속성으로 저장

print(Greeter.greet())           # 클래스로 접근 → 'Hello World!'
```

**주의할 점:** 인스턴스를 통해 호출하면 함수가 **메서드로 바인딩**되어 `self` 가 첫 인자로 자동 전달됩니다.

```python
Greeter().greet()   # TypeError: say_hello() takes 0 positional arguments but 1 was given
```

해결 방법:

```python
class Greeter:
    greet = staticmethod(say_hello)   # 바인딩 방지 → 인스턴스로 호출해도 OK
```

> 이 패턴은 PyTorch 에서 활성화 함수나 변환 함수를 설정값처럼 바꿔 끼울 때 쓰입니다. 예: `self.act = nn.ReLU()` 또는 `self.transform = torchaudio.transforms.MelSpectrogram(...)`. 이것들은 **호출 가능한 객체**(`__call__` 구현)라 바인딩 문제가 없습니다.
