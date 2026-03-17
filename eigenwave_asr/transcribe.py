"""Single-file audio transcription — full long-audio support."""

import os
import torch
import torch.nn.functional as F
import torchaudio


def probe_audio_duration(audio_path):
    """Get audio duration without full model inference."""
    ext = os.path.splitext(audio_path)[1].lower()

    # WAV/FLAC/OGG via soundfile
    if ext in [".wav", ".flac", ".ogg"]:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.frames / info.samplerate
        except Exception:
            pass

    # MP3/M4A via pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception:
        pass

    return None


def load_audio(audio_path, target_sr=16000, max_seconds=None):
    """
    Robust audio loader:
    - WAV/FLAC/OGG → soundfile
    - MP3/M4A → pydub
    - optional truncation
    """
    ext = os.path.splitext(audio_path)[1].lower()
    wav, sr = None, None

    # ── WAV / FLAC / OGG via soundfile ─────────────────────────────
    if ext in [".wav", ".flac", ".ogg"]:
        try:
            import soundfile as sf
            import numpy as np

            data, sr = sf.read(audio_path, always_2d=False)

            if data.ndim == 2:
                print("   🔄 Stereo → mono")
                data = data.mean(axis=1)

            wav = torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Could not load audio with soundfile: {e}")

    # ── MP3 / M4A / AAC via pydub ──────────────────────────────────
    elif ext in [".mp3", ".m4a", ".aac"]:
        try:
            from pydub import AudioSegment
            import numpy as np

            audio = AudioSegment.from_file(audio_path)
            sr = audio.frame_rate
            samples = np.array(audio.get_array_of_samples()).astype("float32")

            if audio.channels > 1:
                print("   🔄 Stereo → mono")
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)

            max_val = float(1 << (8 * audio.sample_width - 1))
            samples = samples / max_val
            wav = torch.tensor(samples, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(
                f"Could not load MP3/M4A file: {e}\n"
                f"Try converting to WAV first."
            )

    else:
        raise RuntimeError(
            f"Unsupported file format: {ext}\n"
            f"Use WAV, FLAC, OGG, MP3, or M4A"
        )

    if wav.dim() > 1:
        wav = wav.squeeze()

    # Resample
    if sr != target_sr:
        print(f"   🔄 Resampling {sr} → {target_sr} Hz")
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    original_duration = len(wav) / target_sr

    # Optional truncation
    if max_seconds is not None:
        max_samples = max_seconds * target_sr
        if len(wav) > max_samples:
            print(f"   ✂️  {original_duration:.1f}s → truncating to {max_seconds}s")
            wav = wav[:max_samples]

    return wav, len(wav) / target_sr, original_duration


def transcribe(model, decoder, vocab, audio_path, device,
               beam_width=100, temperature=1.0, max_seconds=20):
    """
    Auto mode:
    - if audio <= max_seconds → normal transcription
    - if audio > max_seconds → automatically switch to chunked mode
    """
    print(f"\n🎤 Loading: {audio_path}")

    duration = probe_audio_duration(audio_path)
    if duration is not None and duration > max_seconds:
        print(f"   📌 Long audio detected ({duration:.1f}s > {max_seconds}s)")
        print(f"   🔄 Automatically switching to chunked transcription mode...\n")
        return transcribe_long(
            model, decoder, vocab, audio_path, device,
            beam_width=beam_width,
            temperature=temperature,
            chunk_seconds=15,
            overlap_seconds=2,
            save_txt=True
        )

    # Short audio path
    wav, duration, original_duration = load_audio(audio_path, max_seconds=max_seconds)
    print(f"   Duration: {duration:.1f}s (original: {original_duration:.1f}s)")

    model.eval()
    with torch.no_grad():
        wav_tensor = wav.unsqueeze(0).to(device)
        logits = model(wav_tensor)

        greedy_ids = logits[0].argmax(dim=-1).cpu().tolist()
        greedy_text = vocab.decode_greedy(greedy_ids)

        beam_text = greedy_text
        if decoder is not None:
            log_probs = F.log_softmax(logits[0] / temperature, dim=-1)
            beam_text = decoder.decode(log_probs.cpu().numpy(), beam_width=beam_width)

    print(f"\n   📝 Greedy:     {greedy_text}")
    if decoder is not None:
        print(f"   📝 Beam+KenLM: {beam_text}")

    return greedy_text, beam_text


def transcribe_long(model, decoder, vocab, audio_path, device,
                    beam_width=100, temperature=1.0,
                    chunk_seconds=15, overlap_seconds=2,
                    save_txt=True):
    """
    Full long-audio transcription with chunking.
    
    Returns full transcript string.
    """
    print(f"\n🎤 Loading (chunked): {audio_path}")
    wav, duration, original_duration = load_audio(audio_path, max_seconds=None)
    print(f"   Duration: {original_duration:.1f}s")

    chunk_samples = chunk_seconds * 16000
    overlap_samples = overlap_seconds * 16000
    step = chunk_samples - overlap_samples
    n_chunks = max(1, (len(wav) - overlap_samples) // step + 1)

    print(f"   Chunks: {n_chunks} × {chunk_seconds}s (overlap {overlap_seconds}s)\n")

    all_texts = []
    model.eval()

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_samples, len(wav))
            chunk = wav[start:end]

            if len(chunk) < 1600:
                continue

            logits = model(chunk.unsqueeze(0).to(device))
            ids = logits[0].argmax(dim=-1).cpu().tolist()
            text = vocab.decode_greedy(ids)

            if decoder is not None:
                lp = F.log_softmax(logits[0] / temperature, dim=-1)
                text = decoder.decode(lp.cpu().numpy(), beam_width=beam_width)

            all_texts.append(text)
            print(f"   [{start/16000:6.1f}s - {end/16000:6.1f}s] {text[:100]}")

    full_text = " ".join(t.strip() for t in all_texts if t.strip())

    print(f"\n{'='*70}")
    print("📝 FULL TRANSCRIPT")
    print(f"{'='*70}")
    print(full_text[:3000])  # print first 3000 chars safely
    if len(full_text) > 3000:
        print("\n... [truncated in console, full text saved to file]")
    print(f"{'='*70}")

    if save_txt:
        txt_path = os.path.splitext(audio_path)[0] + "_transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\n💾 Full transcript saved to: {txt_path}")

    return full_text