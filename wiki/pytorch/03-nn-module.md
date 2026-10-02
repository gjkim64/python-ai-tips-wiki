# nn.Module: `__init__` vs `forward`

[← 텐서 연산과 차원 다루기](02-tensors-and-shapes.md) · [목차](../README.md) · 다음: [출력 · 확률 · 손실 →](04-logits-probs-loss.md)

---

## 1. 층 하나 = 행렬 하나

층이 하나뿐인 신경망은 **입력 크기 × 출력 크기 행렬**(과 bias)을 곱하는 것과 같습니다.

```python
layer = nn.Linear(80, 32)     # y = x @ W.T + b
layer.weight.shape            # torch.Size([32, 80])  ← 저장은 [출력, 입력] 순서
layer.bias.shape              # torch.Size([32])
```

선형 층만 여러 개 쌓으면 결국 행렬 하나와 같아지므로, 층 사이에 `ReLU` 같은 **비선형 활성화 함수**를 넣어야 깊은 신경망이 의미를 가집니다.

nn.linear is like matrix multiplication
below is how an input of 4 x 10 is converted into 4 x 6
to do this multiply weights of 10 x 6 (60 neurons with weights)

<img width="627" height="401" alt="image" src="https://github.com/user-attachments/assets/4c301232-51e8-43df-bd57-bed32c7135dd" />


## 2. 모델 클래스의 형식

`nn.Module` 은 모든 모델의 부모 클래스입니다.

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_mels=80, hidden=32, n_frames=47, n_classes=10):
        super().__init__()                      # 반드시 첫 줄에서 호출
        self.fc1 = nn.Linear(n_mels, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(n_frames * hidden, n_classes)
        self.act = nn.ReLU()

    def forward(self, x):                       # x: [B, 80, 47]
        x = x.transpose(1, 2)                   # [B, 47, 80]
        x = self.act(self.fc1(x))               # [B, 47, 32]
        x = self.act(self.fc2(x))               # [B, 47, 32]
        x = x.flatten(start_dim=1)              # [B, 1504]
        return self.out(x)                      # [B, 10]  (logits)
```

## 3. `__init__` 과 `forward` 의 차이

| | `__init__` | `forward` |
|---|---|---|
| 역할 | 네트워크의 **구조**(층과 파라미터) 정의 | **실행 로직**(입력이 층을 어떻게 통과하는지) 정의 |
| 호출 시점 | 모델을 만들 때 **한 번** | 데이터를 넣을 때마다 **매번** |
| 들어가는 것 | `nn.Linear`, `nn.Conv2d`, 변환 모듈 등 | 층 호출, reshape/transpose, 활성화 함수, 잔차 연결 등 |

비유하면 `__init__` 은 **부품 조립**, `forward` 는 **작동 순서**입니다.

> **규칙:** 학습할 파라미터가 있는 층은 반드시 `__init__` 에서 `self.xxx` 로 등록하세요. `forward` 안에서 `nn.Linear(...)` 를 새로 만들면 매번 새 가중치가 생겨 학습이 되지 않고, `model.parameters()` 에도 잡히지 않습니다.

### `model.forward(x)` 가 아니라 `model(x)` 로 호출

```python
logits = model(x)          # O
logits = model.forward(x)  # 피하기
```

`nn.Module` 은 [`__call__`](../python/04-magic-methods.md) 을 구현해 두었습니다. `model(x)` 를 호출하면 hook 처리 등을 거친 뒤 내부에서 `forward` 를 실행합니다. `forward` 를 직접 부르면 이 과정이 생략됩니다.

## 4. 전처리를 `forward` 에 넣어도 될까?

**Q.** 데이터 타입 변환이나 정규화 같은 전처리도 `forward` 에 들어가나?

**A.** 넣을 수 있습니다. 기준은 **"추론할 때도 항상 해야 하는 처리인가"** 입니다.

| 위치 | 적합한 처리 | 이유 |
|------|-------------|------|
| `Dataset.__getitem__` | 파일 읽기, 리샘플링, 길이 자르기/채우기, 데이터 증강(augmentation) | 샘플 단위·CPU 작업, `num_workers` 로 병렬화 가능 |
| `forward` (모델 안) | `squeeze`/`transpose` 같은 shape 정리, dtype 변환, 정규화, **Mel 변환** | 배치 단위로 GPU 에서 빠르게 실행. 모델을 저장·배포할 때 전처리도 함께 따라가 학습/추론 불일치를 막음 |

Mel 변환을 모델의 일부로 넣는 예:

```python
import torchaudio

class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, hop_length=1024, n_mels=80)
        self.classifier = Model()

    def forward(self, wav):                     # [B, 1, 48000]
        x = wav.squeeze(1)                      # [B, 48000]
        x = torch.log(self.mel(x) + 1e-8)       # [B, 80, 47], log 스케일 (1e-8: log(0) 방지)
        return self.classifier(x)               # [B, 10]
```

> 데이터 증강(랜덤 노이즈 등)은 학습할 때만 해야 하므로 Dataset 에 두거나, `forward` 안에서 `if self.training:` 으로 감싸세요.

## 5. 최소 학습 루프

```python
model = AudioModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for wav, labels in tqdm(train_loader):
        wav, labels = wav.to(device), labels.to(device)
        logits = model(wav)                  # [32, 10]
        loss = loss_fn(logits, labels)       # logits 와 정수 라벨을 그대로 넣음
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():                    # 검증 시 기울기 계산 끄기
        ...
```

## 6. in audio related class definition 

```python
...
self.mel = torchaudio.transform.MelSpectrogram ... # (just define the transform)
...
in forward method ...
 do something like self.mel (x) ...
```

##  7. flattening layer 는 locality 살리지 못하고, global property 추출 vs. CNN or maxpooling
Flattening can make things unnecessarily big dimensional ...
CNN/max pooling can actually compress information to a smaller piece yet contain important local info

## 8. Channels 

Good example is image with 3 RGB channels

it can go through a CNN layer that produces n channels 
input is 3 channels, output 3 channels ...
so if RGB 3 channel input ... still get 3 channel output?
because, each filter applied to each RGB separately then summed to produce merged output 
so three filters will produce 3 channel output

<img width="658" height="385" alt="image" src="https://github.com/user-attachments/assets/41787fd3-b777-4ed3-a31d-be992b129a5c" />

## 9. if you have H x W image and want to make input to CNN

CNN expects C x H x W tensor (as a rule)
if one image --> C = 1
do an unsqueez to make H x W --> into 1 x H x W

so e.g.
Spectrogram is 80 x 456
unsqueeze to 1 x 80 x 456
then 64 channel output model will compress (80--> 8, and 456 --> 55) and produce
let's say 
64 x 8 (freq) x 55 (time)

we can flatten 64 x 8 --> 512 and finally get

1 x 512 x 55 ...
flattened the freq side ... the 512 1D vector contains eight 64 bit info, first one being low freq data, to the eight being high freq data

## 10. also note that CNN can expect either C x H x W or N x C x H x W, where N is number of batch ...

## 11. you can also squeeze to eliminate meaningless dim when necessary

e.g. 512 x 1 --> 512 

## 12. Using Relu or sigmoid 

<img width="631" height="247" alt="image" src="https://github.com/user-attachments/assets/6124b03d-1341-4165-8c74-ccdb436abbfa" />

## 13. tanh also returns -1 ~ 1, and its center is zero ... good and bad, bad for gradient diminishing problem
but others that have center in small positive number may explode ... 
used much in RNN ..



## 14. you might consider loading big data into memory first (load_audio) so that when get_item is called it is fast processed

