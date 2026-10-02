# Dataset & DataLoader

[← 고급 주제](../python/06-advanced.md) · [목차](../README.md) · 다음: [텐서 연산과 차원 다루기 →](02-tensors-and-shapes.md)

전체 예제: [`examples/genre_dataset.py`](../../examples/genre_dataset.py)

---

## 1. 큰 흐름

1. **Dataset 클래스**를 정해진 형식(`__init__`, `__len__`, `__getitem__`)대로 만든다.
2. 이 Dataset 을 `torch.utils.data.DataLoader` 에 **파라미터로 넘긴다.**
3. DataLoader 의 다른 파라미터로 학습할 때 데이터를 불러오는 방식(batch size, shuffle, 병렬 로딩 등)을 지정한다.

```
파일들 ──▶ Dataset[idx] ──▶ (x, label) 한 개 ──▶ DataLoader ──▶ (X_batch, y_batch) 묶음
```

## 2. Dataset 클래스의 형식

```python
from torch.utils.data import Dataset

class XXX(Dataset):
    def __init__(self, ...):        # 데이터 목록 준비 (파일 경로, 라벨 등). 무거운 로딩은 여기서 하지 않음
        ...
    def __len__(self):              # 전체 샘플 수
        ...
    def __getitem__(self, idx):     # idx 번째 샘플을 (입력 텐서, 라벨 번호) 로 반환
        ...
```

이 세 메서드는 [매직 메서드](../python/04-magic-methods.md)입니다. 덕분에 `len(ds)`, `ds[0]` 이 동작하고, DataLoader 는 이 두 가지만으로 배치를 만들 수 있습니다.

## 3. 라벨(문자열) → 클래스 번호(정수)

신경망 분류기는 `"jazz"` 같은 문자열이 아니라 `0, 1, 2, ...` 같은 정수 라벨을 사용합니다.

```python
# self.data = [(path, "jazz"), (path, "rock"), ...]
entire_labels = [pair[1] for pair in self.data]   # 쌍에서 두 번째(index 1)만 추출
unique_labels = set(entire_labels)                 # 중복 제거
self.classes  = sorted(unique_labels)              # 라벨 자체를 정렬 → 순서 고정

label_idx = self.classes.index(label)              # label 이 몇 번째인지 → 분류 번호
```

- `sorted` 가 중요합니다. `set` 의 순서는 실행마다 달라질 수 있어, 정렬하지 않으면 저장한 모델과 라벨 번호가 어긋날 수 있습니다.
- 매 샘플마다 `list.index` 를 호출하는 대신 딕셔너리를 미리 만들어 두면 더 빠릅니다.
  ```python
  self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
  ```

## 4. 완성 예: 음악 장르 데이터셋

```python
from pathlib import Path
import torch, torchaudio
from torch.utils.data import Dataset

class GenreDataset(Dataset):
    def __init__(self, root: str, num_samples: int = 48000):
        root = Path(root)
        # data/genres/<genre>/<file>.wav  →  (경로, 장르) 쌍
        self.data = [(p, p.parent.name) for p in sorted(root.rglob("*.wav"))]
        self.classes = sorted({label for _, label in self.data})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_samples = num_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.data[idx]
        waveform, sr = torchaudio.load(path)          # [channels, samples]
        waveform = waveform.mean(dim=0, keepdim=True) # 모노로: [1, samples]
        waveform = waveform[:, : self.num_samples]    # 길이 자르기
        pad = self.num_samples - waveform.shape[1]
        if pad > 0:                                   # 짧으면 0 으로 채우기
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform, self.class_to_idx[label]     # [1, 48000], int
```

> 배치로 묶으려면 **모든 샘플의 shape 가 같아야** 합니다. 그래서 길이를 자르거나 채워서 맞춥니다.

## 5. DataLoader

```python
from torch.utils.data import DataLoader

ds = GenreDataset("data/genres")
loader = DataLoader(
    ds,
    batch_size=32,     # 한 번에 묶을 샘플 수
    shuffle=True,      # 에폭마다 순서 섞기 (학습용). 검증/테스트는 False
    num_workers=2,     # 별도 프로세스로 병렬 로딩 (Windows/노트북에서 문제 시 0)
    drop_last=False,   # 마지막에 남는 32개 미만 배치를 버릴지
)

for waveforms, labels in loader:
    print(waveforms.shape, labels.shape)   # torch.Size([32, 1, 48000]) torch.Size([32])
    break
```

DataLoader 는 `ds[i]` 로 꺼낸 샘플 32개를 **쌓아서(stack)** 맨 앞에 배치 차원을 추가합니다.
`[1, 48000]` 짜리 샘플 32개 → `[32, 1, 48000]`.
이 `1` 이 왜 문제가 되는지는 [다음 페이지](02-tensors-and-shapes.md#3-의미-없는-차원-제거-squeeze)에서 다룹니다.

## 6. 학습/검증 데이터 나누기

```python
from torch.utils.data import random_split

n_val = int(len(ds) * 0.2)
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
```

---

### 체크리스트
- [ ] `__getitem__` 이 `(텐서, 정수 라벨)` 을 반환하는가
- [ ] 모든 샘플의 shape 가 같은가 (자르기/채우기)
- [ ] 클래스 목록을 `sorted` 로 고정했는가
- [ ] 첫 배치의 shape 를 `print` 로 확인했는가
