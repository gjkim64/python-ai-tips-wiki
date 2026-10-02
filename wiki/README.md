# Python / PyTorch AI 프로그래밍 팁 Wiki

`python-interm-plus-8-22.txt` 에 흩어져 있던 메모를 주제별로 정리한 위키입니다.
파이썬 중급 → 고급 문법, 그리고 PyTorch 로 오디오(음악 장르) 분류 모델을 만들며 얻은 팁을 담고 있습니다.

## 목차

### Part 1. Python
| # | 페이지 | 주요 내용 |
|---|--------|-----------|
| 1 | [핵심 관용구 (Core Idioms)](python/01-core-idioms.md) | 컴프리헨션, lambda/map/filter/reduce, `*args`/`**kwargs`, 언패킹, collections, 정렬, 복사, f-string |
| 2 | [예외 · 컨텍스트 매니저 · 파일 시스템](python/02-errors-context-files.md) | try-except-else-finally, 커스텀 예외, `with`, `os`/`pathlib`, `assert` |
| 3 | [객체지향 프로그래밍 (OOP)](python/03-oop.md) | 클래스/인스턴스 변수, `@property`, `@classmethod`/`@staticmethod`, 상속과 `super()`, 함수를 클래스 속성으로 |
| 4 | [매직 메서드 & 덕 타이핑](python/04-magic-methods.md) | dunder 메서드, 내장 타입처럼 동작하는 클래스 만들기 |
| 5 | [모듈 · 패키지 · 가상환경 · 타입 힌트](python/05-modules-envs-typing.md) | import 원리, `__init__.py`, `sys.path`, venv/conda, typing |
| 6 | [고급 주제](python/06-advanced.md) | 데코레이터, 메타클래스, 디스크립터, 제너레이터, GIL, asyncio, GC, `__slots__`, Protocol, mypy, 학습 순서 |

### Part 2. PyTorch
| # | 페이지 | 주요 내용 |
|---|--------|-----------|
| 1 | [Dataset & DataLoader](pytorch/01-dataset-dataloader.md) | 커스텀 Dataset 클래스, 라벨 → 클래스 번호 매핑, DataLoader |
| 2 | [텐서 연산과 차원(Shape) 다루기](pytorch/02-tensors-and-shapes.md) | 전체 배열 연산, `arange`/`cumsum`, `squeeze`, transpose, flatten, 배치 차원 따라가기 |
| 3 | [nn.Module: `__init__` vs `forward`](pytorch/03-nn-module.md) | 모델 구조와 실행 로직의 분리, Linear 층 = 행렬 |
| 4 | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) | logit, softmax vs sigmoid, 정답 확률 뽑기(fancy indexing), `log(x + 1e-8)` |

### Part 3. 작업 환경
| 페이지 | 주요 내용 |
|--------|-----------|
| [Colab & 유용한 도구](workflow/colab-and-tools.md) | Colab 라이브 코딩 패턴, `gdown`, `tqdm`, `IPython.display.Audio`, `matplotlib` |

### 부록
- [용어집 (Glossary)](glossary.md)
- [예제 코드 (`examples/`)](../examples/) — 위키 내용을 직접 실행해 볼 수 있는 스크립트

## 디렉터리 구조

```
antig/
├── python-interm-plus-8-22.txt     # 원본 메모 (수정하지 않음)
├── wiki/
│   ├── README.md                   # 이 페이지 (홈 / 목차)
│   ├── glossary.md
│   ├── python/                     # Part 1
│   ├── pytorch/                    # Part 2
│   └── workflow/                   # Part 3
└── examples/
    ├── python_idioms.py            # 표준 라이브러리만 사용
    ├── magic_methods.py            # 표준 라이브러리만 사용
    ├── genre_dataset.py            # torch, torchaudio 필요
    └── shape_walkthrough.py        # torch 필요
```

## 원본 대비 수정 사항
원본 메모의 오타나 부정확한 부분은 정리하면서 바로잡았습니다.

- `torch.arrange` → `torch.arange`
- `torch.utils.data.Dataloader` → `torch.utils.data.DataLoader`
- `dates.string.split("-")` → `date_string.split("-")`
- GC 설명의 "순환 참조(Cyclic Redundancy)" → 순환 참조(Reference Cycle). Cyclic Redundancy는 CRC(오류 검출 부호)를 가리키는 다른 용어입니다.
- 참고문헌 번호 `[6, 7]` 등은 출처가 없어 삭제했습니다.
