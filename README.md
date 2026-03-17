# 🎙️ EigenWave-ASR

**A compact 27.8M parameter speech recognition system with learnable multi-scale temporal features**

Built from scratch on a single Kaggle GPU. Features a novel multi-scale Robin feature frontend
inspired by differential operator formulations from PDE theory, fused with a Conformer encoder
and CTC decoding.

---

## 📊 Results on LibriSpeech

| Test Set       | Greedy CTC | + KenLM Beam Search | Relative Improvement |
|----------------|-----------|---------------------|---------------------|
| **test-clean** | 8.77% WER | **6.88% WER**       | ↓ 21.6%             |
| **test-other** | 20.67% WER| **16.09% WER**      | ↓ 22.2%             |

Evaluated on full test sets (2,620 + 2,939 utterances).

---

## 🧠 Key Innovation: Multi-Scale Robin Features

Traditional ASR uses fixed mel-spectrograms or fixed delta features.
EigenWave-ASR introduces **learnable** weighted combinations of signal,
first derivative, and second derivative — per frequency bin, at multiple
temporal scales:
