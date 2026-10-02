"""wiki/pytorch/01-dataset-dataloader.md 예제.

필요: pip install torch torchaudio
데이터 구조: <root>/<genre>/<file>.wav

실행: python examples/genre_dataset.py data/genres
"""
import sys
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class GenreDataset(Dataset):
    def __init__(self, root: str, num_samples: int = 48000):
        root = Path(root)
        self.data = [(p, p.parent.name) for p in sorted(root.rglob("*.wav"))]
        self.classes = sorted({label for _, label in self.data})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_samples = num_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.data[idx]
        waveform, sr = torchaudio.load(path)           # [channels, samples]
        waveform = waveform.mean(dim=0, keepdim=True)  # 모노: [1, samples]
        waveform = waveform[:, : self.num_samples]
        pad = self.num_samples - waveform.shape[1]
        if pad > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform, self.class_to_idx[label]      # [1, 48000], int


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/genres"
    ds = GenreDataset(root)
    print(f"{len(ds)} files, classes = {ds.classes}")
    assert len(ds) > 0, f"{root} 아래에서 .wav 파일을 찾지 못함"

    x, y = ds[0]
    assert x.shape == (1, 48000), x.shape

    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    waveforms, labels = next(iter(loader))
    print(waveforms.shape, labels.shape)   # [B, 1, 48000], [B]
