# 🎙️ EigenWave-ASR

<div align="center">

### A Compact 27.8M Parameter Automatic Speech Recognition System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Kaggle%20%7C%20CPU-lightgrey.svg)](#)
[![Parameters](https://img.shields.io/badge/Parameters-27.8M-orange.svg)](#-model-summary)

**Multi-Scale Robin Features** · **Conformer Encoder** · **CTC Decoding** · **KenLM Beam Search**

*Built for efficient, high-quality speech recognition on consumer hardware.*

</div>

---

## ⚡ Quick Start (Windows)

```powershell
cd "C:\Users\YourName\Downloads\eigenwave_asr_project"

# Install dependencies
& "C:\Python311\python.exe" -m pip install -r requirements.txt
& "C:\Python311\python.exe" -m pip install kenlm

# Transcribe audio
& "C:\Python311\python.exe" run.py --lm_dir "kenlm_models" --audio "sample.wav"
📊 Results on LibriSpeech
<div align="center">
Test Set	Greedy WER	Beam + KenLM WER	Improvement
test-clean	8.77%	6.88%	↓ 21.6%
test-other	20.67%	16.09%	↓ 22.2%
</div>
💡 KenLM beam search rescoring provides a ~22% relative WER reduction across both test sets.

🏗️ Architecture
<div align="center"><!-- PLACE YOUR ARCHITECTURE IMAGE HERE --><!-- Uncomment the line below and put your image in assets/ folder --><!-- ![EigenWave-ASR Architecture](assets/architecture.png) --></div>
🧠 Model Summary
<div align="center">
text

┌─────────────────────────────────────────────┐
│           EigenWave-ASR  (27.8M)            │
├─────────────────────────────────────────────┤
│  📡  Multi-Scale Robin Frontend             │
│  🔄  12-Layer Conformer-Style Encoder       │
│  📏  384 Hidden Dimension                   │
│  👁️  8 Attention Heads                      │
│  🔤  CTC Output Layer                       │
└─────────────────────────────────────────────┘
</div>
🔬 Robin Learned Coefficients
The Robin frontend learns scale-specific spectral transformations automatically during training.

Scale	α (alpha)	β (beta)	γ (gamma)
Scale 1	0.310	0.582	0.062
Scale 3	0.390	0.072	-0.501
Scale 5	0.391	-0.062	-0.242
🧬 These coefficients emerge from training — they are not hand-tuned. Each scale captures different temporal dynamics of speech.

📁 Project Structure
text

eigenwave_asr_project/
│
├── 📄 run.py                    # Main entry point
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
├── 🧠 hybrid_step182000.pt     # Trained checkpoint
│
├── 📂 kenlm_models/
│   ├── 4-gram.arpa              # LibriSpeech 4-gram LM
│   └── 4-gram_lower.arpa        # Auto-generated lowercase version
│
└── 📂 eigenwave_asr/
    ├── __init__.py
    ├── model.py                 # EigenWave model + Robin frontend
    ├── vocab.py                 # Character vocabulary
    ├── dataset.py               # LibriSpeech data loading
    ├── decoder.py               # Greedy + Beam search decoders
    ├── lm_utils.py              # KenLM integration
    ├── tuning.py                # Optuna hyperparameter search
    └── transcribe.py            # Single-file transcription
🚀 How to Run on Windows
Step 1 → Install Python
Requirement	Details
Recommended	Python 3.11
Avoid	Python 3.13 (KenLM build issues)
Download	python.org/downloads/release/python-3110
⚠️ During installation, check "Add Python to PATH"

Step 2 → Open PowerShell in Project Folder
PowerShell

cd "C:\Users\YourName\Downloads\eigenwave_asr_project"
Step 3 → Install Dependencies
PowerShell

# Core dependencies
& "C:\Python311\python.exe" -m pip install -r requirements.txt

# KenLM (language model support)
& "C:\Python311\python.exe" -m pip install kenlm
<details> <summary>🔧 <b>If KenLM fails to install...</b> (click to expand)</summary><br>
Option A: Make sure Visual Studio Build Tools are installed

Download from visualstudio.microsoft.com
Select "Desktop development with C++"
Option B: Use Python 3.11 (not 3.12 or 3.13)

Option C: Skip KenLM entirely — the model still works with greedy decoding. You just won't get the beam search WER improvement.

</details>
Step 4 → Place Model Checkpoint
Put your trained checkpoint in the project root:

text

eigenwave_asr_project/
└── hybrid_step182000.pt    ← here
Or specify a custom path:

PowerShell

--checkpoint "C:\path\to\your\model.pt"
Step 5 → Place KenLM Model
text

eigenwave_asr_project/
└── kenlm_models/
    └── 4-gram.arpa    ← download and place here
📥 Download: openslr.org/resources/11/4-gram.arpa.gz

The code automatically creates 4-gram_lower.arpa on first run if needed.

🎤 Transcribe Audio
Single WAV File
PowerShell

& "C:\Python311\python.exe" run.py --lm_dir "kenlm_models" --audio "sample.wav"
Single MP3 File
PowerShell

& "C:\Python311\python.exe" run.py --lm_dir "kenlm_models" --audio "sample.mp3"
Long Audio / Chunked Mode
PowerShell

& "C:\Python311\python.exe" run.py --lm_dir "kenlm_models" --audio "podcast.wav" --long
🎵 Note: WAV (16kHz, mono) is the safest format. If MP3 loading fails, see Troubleshooting.

📈 Evaluate on LibriSpeech
Directory Structure
text

data/
└── librispeech/
    ├── test-clean/
    │   └── LibriSpeech/
    │       └── test-clean/
    └── test-other/
        └── LibriSpeech/
            └── test-other/
Run Evaluation
PowerShell

& "C:\Python311\python.exe" run.py --data_dir "data\librispeech"
🔍 Hyperparameter Tuning
Run Optuna Search
PowerShell

& "C:\Python311\python.exe" run.py --tune
Pre-Optimized Defaults
These are used automatically when --tune is not specified:

Parameter	Value
alpha	0.9268
beta	0.061
temperature	0.536
beam_width	50
🛠️ All Commands Reference
Task	Command
Transcribe WAV	python run.py --lm_dir "kenlm_models" --audio "sample.wav"
Transcribe MP3	python run.py --lm_dir "kenlm_models" --audio "sample.mp3"
Long audio	python run.py --lm_dir "kenlm_models" --audio "file.wav" --long
Full evaluation	python run.py --data_dir "data\librispeech"
Custom checkpoint	python run.py --checkpoint "path\to\model.pt" --audio "sample.wav"
Optuna tuning	python run.py --tune
🖥️ CPU Inference
This model is compact enough for real-time CPU inference.

Beam Width	Speed	Quality	Recommended For
1 (greedy)	⚡⚡⚡	⭐⭐	Quick testing
10–20	⚡⚡	⭐⭐⭐	Real-time CPU usage
50	⚡	⭐⭐⭐⭐	Best quality/speed tradeoff
⚠️ Troubleshooting
<details> <summary><b>🎵 Poor results on songs/music</b></summary><br>
This model is trained on speech, not music. Song transcription may be poor because:

Singing style differs from natural speech
Background music interferes with recognition
Lyrics transcription is a fundamentally different task
Best input types:

✅ Audiobooks
✅ Spoken English clips
✅ Interviews and lectures
✅ Your own voice recordings
❌ Songs with background music
</details><details> <summary><b>🔊 MP3 loading fails</b></summary><br>
Convert MP3 to WAV using ffmpeg:

PowerShell

ffmpeg -i input.mp3 -ac 1 -ar 16000 output.wav
Then transcribe:

PowerShell

& "C:\Python311\python.exe" run.py --lm_dir "kenlm_models" --audio "output.wav"
</details><details> <summary><b>🔧 KenLM won't install</b></summary><br>
You can still run the model in greedy-only mode. Beam search + LM just improves accuracy. The model works fine without it — you will get ~8.77% WER on test-clean instead of 6.88%.

</details><details> <summary><b>🐍 Python version issues</b></summary><br>
Python Version	Status
3.11	✅ Fully supported
3.10	✅ Should work
3.12	⚠️ KenLM may have issues
3.13	❌ KenLM build likely fails
</details>
📋 Requirements
text

torch>=2.0
torchaudio>=2.0
numpy
jiwer
tqdm
soundfile
optuna
pyctcdecode
Optional: kenlm (for beam search rescoring)

👤 Author
<div align="center">
Sakib Hasan

Independent research project on efficient ASR with learnable multi-scale temporal features.

⭐ If this project helped you, consider giving it a star!

</div> ```
Copy everything above starting from # 🎙️ EigenWave-ASR all the way to the last </div>. Paste it directly into your README.md file on GitHub.

For your architecture image:

Create folder assets/ in your repo
Put your image there as architecture.png
Find this line in the file:
text

<!-- ![EigenWave-ASR Architecture](assets/architecture.png) -->
Change it to:
text

![EigenWave-ASR Architecture](assets/architecture.png)
