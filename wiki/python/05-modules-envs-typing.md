# 모듈 · 패키지 · 가상환경 · 타입 힌트

[← 매직 메서드 & 덕 타이핑](04-magic-methods.md) · [목차](../README.md) · 다음: [고급 주제 →](06-advanced.md)

---

## 1. 모듈과 패키지 구조

- **모듈**: `.py` 파일 하나
- **패키지**: 모듈을 모아둔 폴더. `__init__.py` 가 있으면 일반 패키지로 인식됩니다.

```
my_project/
├── train.py
└── audio/
    ├── __init__.py        # 패키지 초기화. 여기서 import 하면 외부에 노출 가능
    ├── dataset.py
    └── model.py
```

```python
# audio/__init__.py
from .dataset import GenreDataset
from .model import GenreClassifier

# train.py
from audio import GenreDataset, GenreClassifier
```

### import 는 어떻게 모듈을 찾나: `sys.path`

`import x` 를 하면 파이썬은 `sys.path` 리스트의 경로를 **순서대로** 검색합니다.

1. 실행한 스크립트가 있는 폴더
2. 환경변수 `PYTHONPATH`
3. 표준 라이브러리, `site-packages` (pip 로 설치한 패키지)

```python
import sys
print(sys.path)
```

> **흔한 실수:** 내 파일 이름을 `torch.py`, `random.py` 처럼 지으면 진짜 라이브러리 대신 내 파일이 import 됩니다.

### `if __name__ == "__main__":`

```python
def main(): ...

if __name__ == "__main__":   # 직접 실행할 때만 동작, import 될 때는 실행 안 됨
    main()
```

## 2. 가상 환경 (Virtual Environments)

프로젝트마다 독립된 패키지 환경을 만들어 버전 충돌을 막습니다.

**venv (표준 라이브러리)**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install torch torchaudio
pip freeze > requirements.txt     # 현재 환경 기록
pip install -r requirements.txt   # 다른 곳에서 재현
```

**conda**

```bash
conda create -n audio python=3.11
conda activate audio
```

> PyTorch 는 CUDA 버전에 맞는 빌드를 설치해야 하므로 [pytorch.org](https://pytorch.org/get-started/locally/) 의 설치 명령을 그대로 쓰는 것이 안전합니다.

## 3. 타입 힌트 (Type Hinting)

변수와 함수 인자의 타입을 표시해 가독성을 높이고, 에디터 자동완성과 정적 검사(mypy)를 돕습니다.
**실행 시 강제되지는 않습니다** — 힌트일 뿐입니다.

```python
from pathlib import Path
import torch

def load_audio(name: str, sample_rate: int = 22050) -> torch.Tensor:
    ...

def collect(paths: list[Path]) -> dict[str, int]:
    ...

def find(label: str) -> int | None:   # None 일 수도 있음 (Python 3.10+)
    ...
```

| 표기 | 의미 |
|------|------|
| `list[int]`, `dict[str, float]` | 요소 타입 지정 (3.9+) |
| `int \| None` | `Optional[int]` 와 같음 (3.10+) |
| `tuple[torch.Tensor, int]` | `Dataset.__getitem__` 반환 타입에 자주 사용 |
| `Callable[[int], str]` | 함수 타입 (`from collections.abc import Callable`) |

→ 타입 힌트를 활용한 정적 검사는 [고급 주제 › 정적 분석](06-advanced.md#5-정적-분석-static-analysis) 참고
