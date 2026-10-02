# 출력 · 확률 · 손실 (Logits, Probabilities, Loss)

[← nn.Module](03-nn-module.md) · [목차](../README.md) · 다음: [Colab & 유용한 도구 →](../workflow/colab-and-tools.md)

---

## 1. Logit

신경망의 **가공하지 않은 출력**을 logit 이라고 합니다. 범위 제한이 없는 실수(음수 포함)이며 아직 확률이 아닙니다.

```python
logits = model(x)    # [32, 10]  — 32곡 × 10장르 점수
```

## 2. Softmax vs Sigmoid

| | Softmax | Sigmoid |
|---|---|---|
| 결과 | 각 값이 0~1, **합이 1** (확률 분포) | 각 값이 0~1, **합은 1이 아님** (독립적) |
| 용도 | **다중 클래스** 분류: 정답이 하나 (예: 장르 하나) | **이진 / 다중 라벨** 분류: 여러 개가 동시에 정답일 수 있음 (예: 악기 태그) |
| PyTorch | `torch.softmax(logits, dim=1)` | `torch.sigmoid(logits)` |
| 함께 쓰는 손실 | `nn.CrossEntropyLoss` | `nn.BCEWithLogitsLoss` |

```python
probs = torch.softmax(logits, dim=1)   # dim=1: 클래스 축으로 정규화 (dim=0 이면 배치 축 → 틀림)
probs.sum(dim=1)                       # 모두 1.0
pred = probs.argmax(dim=1)             # 샘플마다 가장 확률 높은 클래스 번호 [32]
```

## 3. 정답 클래스의 확률만 뽑기 (Fancy Indexing)

이제 32개 샘플의 확률 묶음과 32개 정답 라벨이 있습니다.

```python
probs  = [[ ... 10개 ... ],    # 샘플 0
          [ ... 10개 ... ],    # 샘플 1
          ... ]                # shape [32, 10]
labels = [2, 4, 0, 8, ...]     # shape [32]
```

학습의 목표는 **각 샘플에서 정답 위치의 확률이 1(100%)이 되는 것**입니다. 그 확률만 골라내려면:

```python
correct_probs = probs[torch.arange(len(probs)), labels]    # shape [32]
```

두 인덱스 배열을 짝지어 원소를 하나씩 꺼냅니다. 반복문 32번과 같은 결과를 한 번에 얻습니다.

```
행 인덱스: torch.arange(32) = [0, 1, 2, 3, ...]
열 인덱스: labels           = [2, 4, 0, 8, ...]
→ probs[0, 2], probs[1, 4], probs[2, 0], probs[3, 8], ...
```

NumPy 에서도 똑같이 동작합니다: `probs[np.arange(len(probs)), labels]`

## 4. log 와 `1e-8`: log(0) 방지

손실은 정답 확률에 log 를 씌워 계산합니다(**음의 로그 우도**, negative log-likelihood).

```python
loss = -torch.log(correct_probs + 1e-8).mean()
```

- `log(0) = -∞` 이므로 확률이 정확히 0이 되면 손실이 무한대(`inf`)가 되고, 이후 계산이 `nan` 으로 망가집니다.
- 아주 작은 값 `1e-8` (epsilon)을 더해 이를 막습니다.
- 정답 확률이 1 에 가까울수록 `-log(p)` 는 0 에 가까워집니다. 즉 손실이 작아집니다.

| 정답 확률 p | -log(p) |
|---|---|
| 1.0 | 0.000 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.0 (+1e-8) | 18.42 |

## 5. 실전에서는: `nn.CrossEntropyLoss`

위 과정(softmax → 정답 확률 추출 → `-log` → 평균)을 한 번에 해 주는 것이 `CrossEntropyLoss` 입니다.

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)    # softmax 를 거치지 않은 logits 를 그대로 넣음!
```

- 내부적으로 `log_softmax` 를 사용해 수치적으로 안정적이므로 `1e-8` 같은 보정이 필요 없습니다.
- **흔한 실수:** softmax 를 적용한 확률을 넣으면 softmax 가 두 번 적용되어 학습이 잘 안 됩니다.
- 직접 구현(3~4절)은 원리 이해용으로, 결과는 `CrossEntropyLoss` 와 거의 같습니다 ([`examples/shape_walkthrough.py`](../../examples/shape_walkthrough.py) 에서 확인).

## 6. 정확도 계산

```python
pred = logits.argmax(dim=1)                       # softmax 없이도 argmax 결과는 같음
accuracy = (pred == labels).float().mean().item()
```
