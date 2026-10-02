# 용어집 (Glossary)

[← 목차](README.md)

| 용어 | 설명 | 관련 페이지 |
|------|------|-------------|
| **Batch (배치)** | 한 번에 모델에 넣는 샘플 묶음. 텐서의 0번 차원 | [Dataset & DataLoader](pytorch/01-dataset-dataloader.md) |
| **Comprehension** | `[f(x) for x in xs if cond]` 형태로 컬렉션을 만드는 문법 | [핵심 관용구](python/01-core-idioms.md) |
| **Context Manager** | `with` 문으로 자원을 열고 자동으로 닫는 객체 (`__enter__`/`__exit__`) | [매직 메서드](python/04-magic-methods.md) |
| **CrossEntropyLoss** | softmax + 정답 확률의 `-log` 평균을 한 번에 계산하는 다중 분류 손실 | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) |
| **Decorator** | 함수를 감싸 기능을 덧붙이는 `@문법` | [고급 주제](python/06-advanced.md) |
| **Descriptor** | 속성 접근(`__get__`/`__set__`)을 제어하는 프로토콜. `@property` 의 기반 | [고급 주제](python/06-advanced.md) |
| **Duck Typing** | 타입이 아니라 가진 메서드로 객체를 다루는 방식 | [매직 메서드](python/04-magic-methods.md) |
| **Dunder Method** | `__len__` 처럼 앞뒤 밑줄 두 개가 붙은 특수 메서드 | [매직 메서드](python/04-magic-methods.md) |
| **Epsilon (`1e-8`)** | `log(0)`, 0으로 나누기를 막기 위해 더하는 아주 작은 값 | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) |
| **Fancy Indexing** | 인덱스 배열로 원소를 골라내는 방법. `probs[arange(n), labels]` | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) |
| **forward** | `nn.Module` 에서 입력이 층을 통과하는 실행 로직을 정의하는 메서드 | [nn.Module](pytorch/03-nn-module.md) |
| **Generator** | `yield` 로 값을 하나씩 만드는 이터레이터. 한 번 쓰면 소진 | [고급 주제](python/06-advanced.md) |
| **GIL** | CPython 에서 한 번에 한 스레드만 바이트코드를 실행하게 하는 락 | [고급 주제](python/06-advanced.md) |
| **Logit** | softmax/sigmoid 를 적용하기 전 신경망의 원시 출력 | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) |
| **Mel Spectrogram** | 사람 청각에 맞춘 주파수 축(mel)으로 변환한 시간-주파수 표현. `[n_mels, time]` | [텐서 연산과 차원](pytorch/02-tensors-and-shapes.md) |
| **Metaclass** | 클래스를 만드는 클래스 (기본 `type`) | [고급 주제](python/06-advanced.md) |
| **Protocol** | 메서드 구성으로 타입을 정의하는 구조적 서브타이핑 (`typing.Protocol`) | [고급 주제](python/06-advanced.md) |
| **Sigmoid** | 각 값을 독립적으로 0~1 로 변환. 합이 1이 아님 | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) |
| **Softmax** | 값들을 합이 1인 확률 분포로 변환 | [출력 · 확률 · 손실](pytorch/04-logits-probs-loss.md) |
| **squeeze / unsqueeze** | 크기 1인 차원을 제거 / 추가 | [텐서 연산과 차원](pytorch/02-tensors-and-shapes.md) |
| **Type Hint** | `def f(name: str) -> int:` 처럼 타입을 표시하는 문법. 실행 시 강제되지 않음 | [모듈 · 타입 힌트](python/05-modules-envs-typing.md) |
| **venv** | 프로젝트별 독립 파이썬 환경 | [모듈 · 가상환경](python/05-modules-envs-typing.md) |
