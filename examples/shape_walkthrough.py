"""wiki/pytorch/02 ~ 04 예제: 배치 shape 따라가기, logits, 정답 확률 추출, 손실.

데이터 없이 가짜 배치(torch.randn)로 실행된다.
필요: pip install torch torchaudio
실행: python examples/shape_walkthrough.py
"""
import torch
import torch.nn as nn
import torchaudio


class Model(nn.Module):
    def __init__(self, n_mels=80, hidden=32, n_frames=47, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(n_mels, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(n_frames * hidden, n_classes)
        self.act = nn.ReLU()

    def forward(self, x):                   # [B, 80, 47]
        x = x.transpose(1, 2)               # [B, 47, 80]
        x = self.act(self.fc1(x))           # [B, 47, 32]
        x = self.act(self.fc2(x))           # [B, 47, 32]
        x = x.flatten(start_dim=1)          # [B, 1504]
        return self.out(x)                  # [B, 10]


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, hop_length=1024, n_mels=80)
        self.classifier = Model()

    def forward(self, wav):                 # [B, 1, 48000]
        x = wav.squeeze(1)                  # [B, 48000]
        x = torch.log(self.mel(x) + 1e-8)   # [B, 80, 47]
        return self.classifier(x)           # [B, 10]


def main():
    torch.manual_seed(0)
    B = 32
    wav = torch.randn(B, 1, 48000)
    model = AudioModel()

    # 단계별 shape 출력
    x = wav.squeeze(1);              print("squeeze   ", tuple(x.shape))
    x = model.mel(x);                print("mel       ", tuple(x.shape))
    assert x.shape == (B, 80, 47)
    x = x.transpose(1, 2);           print("transpose ", tuple(x.shape))
    x = model.classifier.fc1(x);     print("fc1       ", tuple(x.shape))
    x = model.classifier.fc2(x);     print("fc2       ", tuple(x.shape))
    x = x.flatten(start_dim=1);      print("flatten   ", tuple(x.shape))

    logits = model(wav)
    print("logits    ", tuple(logits.shape))
    assert logits.shape == (B, 10)

    # softmax: 각 행의 합이 1
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(B))

    # 정답 클래스 확률만 뽑기 (fancy indexing)
    labels = torch.randint(0, 10, (B,))
    correct_probs = probs[torch.arange(len(probs)), labels]
    loop_version = torch.stack([probs[i, labels[i]] for i in range(B)])
    assert torch.equal(correct_probs, loop_version)

    # 직접 구현한 손실 vs CrossEntropyLoss
    manual = -torch.log(correct_probs + 1e-8).mean()
    builtin = nn.CrossEntropyLoss()(logits, labels)
    print(f"manual NLL = {manual:.6f}, CrossEntropyLoss = {builtin:.6f}")
    assert torch.allclose(manual, builtin, atol=1e-5)

    # squeeze() 를 차원 지정 없이 쓸 때의 함정
    single = torch.randn(1, 1, 48000)
    print("squeeze() on batch of 1:", tuple(single.squeeze().shape), "<- 배치 차원 사라짐")
    print("squeeze(1) on batch of 1:", tuple(single.squeeze(1).shape))


if __name__ == "__main__":
    main()
