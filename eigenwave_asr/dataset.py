"""LibriSpeech data loading — auto-detects platform paths."""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


def _find_data_root(data_dir, subset):
    """Try multiple path patterns to find data."""
    candidates = [
        # Kaggle
        f"/kaggle/input/librispeech/{subset}/LibriSpeech/{subset}",
        # Colab / local
        os.path.join(data_dir, subset, "LibriSpeech", subset),
        os.path.join(data_dir, "LibriSpeech", subset),
        os.path.join(data_dir, subset),
        # Anti-Gravity or custom
        f"/data/librispeech/{subset}",
        f"./data/{subset}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class LibriSpeechDataset(Dataset):
    """Platform-independent LibriSpeech loader."""

    def __init__(self, data_dir, subset, vocab, max_len=320000):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []
        self.use_torchaudio = False

        # Try to find data folder
        folder = _find_data_root(data_dir, subset)

        if folder:
            print(f"📂 Found data: {folder}")
            self._load_folder(folder)
        else:
            # Fallback: torchaudio auto-download
            try:
                from torchaudio.datasets import LIBRISPEECH
                print(f"📥 Downloading {subset} via torchaudio...")
                self.dataset = LIBRISPEECH(data_dir, url=subset, download=True)
                self.use_torchaudio = True
                print(f"✅ Loaded {len(self.dataset)} samples")
                return
            except Exception as e:
                print(f"⚠️  Cannot load {subset}: {e}")
                print(f"   Searched paths:")
                for c in [f"/kaggle/input/librispeech/{subset}/LibriSpeech/{subset}",
                          os.path.join(data_dir, subset)]:
                    print(f"     {c}")
                self.samples = [(torch.randn(16000), "test fallback")]

    def _load_folder(self, root):
        for spk in sorted(os.listdir(root)):
            spk_path = os.path.join(root, spk)
            if not os.path.isdir(spk_path):
                continue
            for chap in os.listdir(spk_path):
                chap_path = os.path.join(spk_path, chap)
                if not os.path.isdir(chap_path):
                    continue
                trans_file = os.path.join(chap_path, f"{spk}-{chap}.trans.txt")
                if not os.path.exists(trans_file):
                    continue
                trans = {}
                with open(trans_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            trans[parts[0]] = parts[1]
                for fname in os.listdir(chap_path):
                    if fname.endswith('.flac'):
                        uid = fname.replace('.flac', '')
                        if uid in trans:
                            self.samples.append(
                                (os.path.join(chap_path, fname), trans[uid])
                            )
        print(f"✅ Loaded {len(self.samples)} samples")

    def __len__(self):
        if self.use_torchaudio:
            return len(self.dataset)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_torchaudio:
            wav, sr, text, *_ = self.dataset[idx]
            wav = wav.squeeze(0)
        else:
            path, text = self.samples[idx]
            if isinstance(path, str):
                wav, sr = torchaudio.load(path)
                wav = wav.squeeze(0)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            else:
                wav = path
        if len(wav) > self.max_len:
            wav = wav[:self.max_len]
        return wav, torch.tensor(self.vocab.encode(text)), text


def collate(batch):
    wavs, toks, txts = zip(*batch)
    max_w = max(len(w) for w in wavs)
    max_t = max(len(t) for t in toks) or 1
    w_b = torch.zeros(len(wavs), max_w)
    t_b = torch.zeros(len(toks), max_t, dtype=torch.long)
    w_l, t_l = [], []
    for i, (w, t) in enumerate(zip(wavs, toks)):
        w_b[i, :len(w)] = w
        w_l.append(len(w))
        if len(t) > 0:
            t_b[i, :len(t)] = t
        t_l.append(len(t))
    return w_b, torch.tensor(w_l), t_b, torch.tensor(t_l), txts


def get_dataloader(data_dir, subset, vocab, batch_size=4, max_len=320000):
    dataset = LibriSpeechDataset(data_dir, subset, vocab, max_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate
    )
