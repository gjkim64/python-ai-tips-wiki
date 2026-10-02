# 예외 처리 · 컨텍스트 매니저 · 파일 시스템

[← 핵심 관용구](01-core-idioms.md) · [목차](../README.md) · 다음: [객체지향 프로그래밍 →](03-oop.md)

---

## 1. 예외 처리: `try-except-else-finally`

```python
try:
    f = open("config.json")
except FileNotFoundError as e:   # 예외 발생 시
    print("설정 파일 없음:", e)
else:                            # 예외가 없을 때만 실행
    print(f.read())
    f.close()
finally:                         # 성공/실패와 관계없이 항상 실행 (정리 작업)
    print("done")
```

| 블록 | 실행 시점 |
|------|-----------|
| `try` | 항상 먼저 |
| `except` | `try` 안에서 해당 예외 발생 시 |
| `else` | `try` 가 예외 없이 끝났을 때 |
| `finally` | 무조건 마지막에 |

> `except Exception:` 처럼 넓게 잡기보다 **구체적인 예외**를 잡는 것이 좋습니다. 그래야 예상치 못한 버그가 묻히지 않습니다.

### 커스텀 예외와 `raise`

```python
class InvalidAudioError(ValueError):
    """오디오 파일 형식이 잘못되었을 때."""

def load(path):
    if not path.endswith(".wav"):
        raise InvalidAudioError(f"wav 파일만 지원: {path}")
```

## 2. `assert` — "이 조건이 맞아야 나머지를 진행한다"

```python
assert waveform.ndim == 2, f"(channels, samples) 형태여야 함, 현재 {waveform.shape}"
```

- 조건이 거짓이면 `AssertionError` 를 내고 멈춥니다.
- 텐서 shape 처럼 **개발 중 가정을 검증**하는 데 적합합니다.
- 주의: `python -O` 로 실행하면 assert 문이 제거됩니다. 사용자 입력 검증처럼 반드시 필요한 검사에는 `if ...: raise` 를 쓰세요.

## 3. 컨텍스트 매니저: `with`

`with` 블록을 벗어나면 예외가 나더라도 자원이 자동으로 반환(파일 닫기 등)됩니다.

```python
with open("log.txt", "w", encoding="utf-8") as f:
    f.write("hello")
# 여기서 f 는 이미 닫혀 있음
```

직접 만드는 방법(`__enter__`/`__exit__`, `contextlib.contextmanager`)은 [매직 메서드](04-magic-methods.md#3-컨텍스트-매니저-만들기) 페이지를 참고하세요.

> PyTorch 에서 자주 보는 예: `with torch.no_grad():` — 추론(inference) 중 기울기 계산을 끕니다.

## 4. 파일 시스템: `os` vs `pathlib`

`pathlib.Path` 는 경로를 객체로 다룹니다. 새 코드에서는 `os.path` 보다 권장됩니다.

```python
from pathlib import Path

data_dir = Path("data/genres")

data_dir / "jazz" / "a.wav"      # 경로 결합 (OS 에 맞는 구분자 자동 사용)
p = Path("data/genres/jazz/a.wav")
p.name, p.stem, p.suffix         # 'a.wav', 'a', '.wav'
p.parent.name                    # 'jazz'  ← 폴더 이름을 라벨로 쓸 때 유용
p.exists(), p.is_file()
data_dir.mkdir(parents=True, exist_ok=True)
```

### 하위 폴더까지 재귀 검색: `rglob`

```python
wav_files = list(data_dir.rglob("*.wav"))   # data_dir 아래 모든 .wav 를 재귀 탐색
```

- `rglob` 은 **제너레이터**를 반환합니다. 한 번 순회하면 소진되므로, 길이를 세거나 여러 번 쓸 거라면 `list(...)` 로 저장하세요.
- 현재 폴더만 찾으려면 `glob("*.wav")` 를 씁니다.
- 실행할 때마다 순서를 같게 하려면 `sorted(data_dir.rglob("*.wav"))` 를 쓰세요.

**폴더 구조에서 (파일, 라벨) 쌍 만들기:**

```python
# data/genres/<genre>/<file>.wav 구조라고 가정
data = [(p, p.parent.name) for p in sorted(data_dir.rglob("*.wav"))]
```

| 작업 | `os` | `pathlib` |
|------|------|-----------|
| 경로 결합 | `os.path.join(a, b)` | `Path(a) / b` |
| 파일명 | `os.path.basename(p)` | `p.name` |
| 확장자 | `os.path.splitext(p)[1]` | `p.suffix` |
| 존재 여부 | `os.path.exists(p)` | `p.exists()` |
| 재귀 검색 | `os.walk(...)` | `p.rglob("*")` |
| 폴더 생성 | `os.makedirs(p, exist_ok=True)` | `p.mkdir(parents=True, exist_ok=True)` |
