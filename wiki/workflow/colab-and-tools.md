# Colab & 유용한 도구

[← 출력 · 확률 · 손실](../pytorch/04-logits-probs-loss.md) · [목차](../README.md)

---

## 1. Google Colab 라이브 코딩 패턴

클래스나 함수 같은 구조를 만들 때 **한 셀 안에서 정의하고 바로 아래에 테스트 호출**을 두면, 셀을 다시 실행할 때마다 즉시 검증됩니다.

```python
# ── 정의 ──
class GenreDataset(Dataset):
    ...

# ── 바로 아래에서 테스트 ──
ds = GenreDataset("data/genres")
print(len(ds), ds.classes)
x, y = ds[0]
print(x.shape, y)
assert x.shape == (1, 48000)
```

- 구조가 완성되면 테스트 부분은 지우거나 `if __name__ == "__main__":` 아래로 옮깁니다.
- `assert` 로 shape 같은 조건을 확인해 두면, 조건이 맞을 때만 다음 셀로 진행됩니다 ([assert](../python/02-errors-context-files.md#2-assert--이-조건이-맞아야-나머지를-진행한다)).

## 2. `gdown` — Google Drive 파일 내려받기

공유된 Google Drive 파일을 **파일 ID** 로 내려받습니다. Colab 에서 데이터셋을 가져올 때 편리합니다.

```bash
pip install gdown              # Colab 에는 기본 설치되어 있음
gdown <FILE_ID>                # 공유 링크 .../d/<FILE_ID>/view 에서 ID 부분
gdown --folder <FOLDER_URL>    # 폴더 전체
```

Colab 셀에서는 앞에 `!` 를 붙입니다: `!gdown <FILE_ID>`

> 원본 메모에 있던 실제 데이터셋 ID: `1-4elQY1C-n23u3QqomnLiI9CN9iPrWC3`

## 3. `tqdm` — 진행 표시줄

```python
from tqdm.auto import tqdm     # 노트북이면 위젯, 터미널이면 텍스트 막대를 자동 선택

for i in tqdm(range(1000)):
    ...

for wav, labels in tqdm(loader, desc="train"):
    ...
```

## 4. `IPython.display` — 노트북에서 오디오 재생

```python
import IPython.display as ipd

ipd.Audio("data/genres/jazz/a.wav")                    # 파일 재생
ipd.Audio(waveform.numpy(), rate=sample_rate)          # 배열 재생 (rate 필수)
```

데이터를 정말 제대로 불러왔는지 **직접 들어 보는 것**이 가장 빠른 확인 방법입니다.

## 5. `matplotlib` — 파형 일부 그려 보기

```python
import matplotlib.pyplot as plt

plt.plot(waveform[0, :500])    # 앞 500 샘플만 확대해서 보기
plt.show()

plt.imshow(mel[0].log(), origin="lower", aspect="auto")   # Mel 스펙트로그램 [80, 47]
plt.colorbar()
plt.show()
```

- 전체 신호는 너무 촘촘하므로 `[:500]` 처럼 잘라서 보면 파형이 잘 보입니다.
- 스펙트로그램은 `origin="lower"` 로 저주파가 아래에 오게 그립니다.
