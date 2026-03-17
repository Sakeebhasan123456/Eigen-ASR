
#!/usr/bin/env python3
"""
EigenWave-ASR — Main Entry Point
==================================

Usage:
    python run.py                                    # Evaluate LibriSpeech
    python run.py --audio recording.wav              # Transcribe short audio
    python run.py --audio song.mp3 --long            # Transcribe full song
    python run.py --audio podcast.mp3 --max_seconds 60  # First 60 seconds
    python run.py --tune                             # Optuna tuning
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

from eigenwave_asr.model import EnhancedHybridASR
from eigenwave_asr.vocab import Vocab
from eigenwave_asr.dataset import LibriSpeechDataset, collate, get_dataloader
from eigenwave_asr.lm_utils import get_kenlm_path
from eigenwave_asr.decoder import build_decoder, evaluate, check_dependencies
from eigenwave_asr.transcribe import transcribe, transcribe_long
from torch.utils.data import DataLoader


def _find_checkpoint(user_path=None):
    if user_path and os.path.exists(user_path):
        return user_path
    candidates = [
        "/kaggle/working/hybrid_best.pt",
        "/kaggle/working/hybrid_step182000.pt",
        "/kaggle/input/datasets/sakibhasanml/step182000/hybrid_step182000.pt",
        "/kaggle/input/datasets/sakibhasanml/step146000/hybrid_step146000.pt",
        "/kaggle/input/hybrid-checkpoint/hybrid_best.pt",
        "hybrid_best.pt", "hybrid_step182000.pt",
        "./checkpoints/hybrid_best.pt",
        "./models/hybrid_best.pt",
        "/data/models/hybrid_best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_model(vocab, device, checkpoint=None):
    model = EnhancedHybridASR(
        d_model=384, n_layers=12, n_heads=8,
        vocab_size=vocab.size, dropout=0.0
    ).to(device)

    checkpoint_path = _find_checkpoint(checkpoint)

    if checkpoint_path:
        print(f"📂 Loading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        if isinstance(state_dict, dict):
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v
                             for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        step = ckpt.get('step', '?')
        saved_wer = ckpt.get('wer', ckpt.get('best_wer', '?'))
        print(f"✅ Loaded (step={step}, saved_wer={saved_wer})")
        robin_stats = model.get_robin_stats()
        for scale, stats in robin_stats.items():
            print(f"   Robin {scale}: α={stats['alpha']:.3f}, "
                  f"β={stats['beta']:.3f}, γ={stats['gamma']:.3f}")
    else:
        print("⚠️  No checkpoint found — using random weights")
        print("   Provide: python run.py --checkpoint /path/to/model.pt")

    model.eval()
    n_params = model.count_parameters()
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    return model, checkpoint_path


def _find_data_dir():
    candidates = [
        "/kaggle/input/librispeech",
        "/data/librispeech",
        "./data/librispeech",
        "./LibriSpeech",
        os.path.expanduser("~/librispeech"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "./data"


def run_cpu_benchmark(model, vocab, data_dir, alpha, beta, temperature):
    print(f"\n{'='*70}")
    print(f"⏱️  CPU REAL-TIME BENCHMARK")
    print(f"{'='*70}\n")

    model_cpu = model.cpu()
    model_cpu.eval()

    try:
        test_data = LibriSpeechDataset(data_dir, "test-clean", vocab)
    except Exception:
        test_data = None

    kenlm_path = get_kenlm_path()
    cpu_decoder = None
    if check_dependencies() and kenlm_path:
        import io, contextlib
        from pyctcdecode import build_ctcdecoder as _build
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    cpu_decoder = _build(
                        labels=vocab.get_labels_for_decoder(),
                        kenlm_model_path=kenlm_path,
                        alpha=alpha, beta=beta,
                    )
                except Exception:
                    cpu_decoder = _build(labels=vocab.get_labels_for_decoder())

    test_beam_widths = [5, 10, 20, 50]
    test_durations = [1.0, 3.0, 5.0, 10.0, 20.0]

    print(f"  {'Beam':<6} {'Audio':<8} {'Model':<10} {'Decode':<10} "
          f"{'Total':<10} {'RTF':<8} {'Status':<10}")
    print(f"  {'─'*62}")

    for bw in test_beam_widths:
        rtfs = []
        for target_dur in test_durations:
            n_samples = int(target_dur * 16000)
            if test_data and len(test_data) > 0:
                wav, tok, txt = test_data[0]
                if len(wav) < n_samples:
                    wav = F.pad(wav, (0, n_samples - len(wav)))
                else:
                    wav = wav[:n_samples]
            else:
                wav = torch.randn(n_samples)
            actual_dur = len(wav) / 16000
            if bw == test_beam_widths[0] and target_dur == test_durations[0]:
                with torch.no_grad():
                    _ = model_cpu(wav.unsqueeze(0))
            with torch.no_grad():
                t0 = time.time()
                logits = model_cpu(wav.unsqueeze(0))
                model_time = time.time() - t0
                t1 = time.time()
                if cpu_decoder:
                    log_probs = F.log_softmax(
                        logits[0] / temperature, dim=-1).numpy()
                    text = cpu_decoder.decode(log_probs, beam_width=bw)
                else:
                    _ = logits[0].argmax(dim=-1)
                decode_time = time.time() - t1
            total_time = model_time + decode_time
            rtf = total_time / actual_dur
            rtfs.append(rtf)
            status = ("✅ RT" if rtf < 1.0
                      else ("⚠️ Slow" if rtf < 2.0 else "❌ Too slow"))
            print(f"  {bw:<6} {actual_dur:<8.1f}s {model_time:<10.3f}s "
                  f"{decode_time:<10.3f}s {total_time:<10.3f}s "
                  f"{rtf:<8.2f}x {status}")
        avg_rtf = sum(rtfs) / len(rtfs)
        rt_status = '✅ REAL-TIME' if avg_rtf < 1.0 else '❌ NOT RT'
        print(f"  {'':<6} {'AVG':<8} {'':<10} {'':<10} {'':<10} "
              f"{avg_rtf:<8.2f}x {rt_status}")
        print(f"  {'─'*62}")


def main(
    checkpoint=None,
    lm_dir="kenlm_models",
    alpha=0.9268,
    beta=0.061,
    temperature=0.536,
    beam_width=50,
    max_samples=None,
    max_seconds=20,
    tune=False,
    long=False,
    audio=None,
    data_dir=None,
):
    """EigenWave-ASR inference — works on any platform."""
    if data_dir is None:
        data_dir = _find_data_dir()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    print(f"📁 Data dir: {data_dir}")

    vocab = Vocab()
    print(f"📝 {vocab}")

    # ── Load Model ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Loading EigenWave-ASR Model")
    print(f"{'='*70}")
    model, checkpoint_path = load_model(vocab, device, checkpoint)

    # ── Setup KenLM ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Setting up Language Model")
    print(f"{'='*70}")
    has_beam = check_dependencies()
    kenlm_path = get_kenlm_path(lm_dir) if has_beam else None
    decoder = None
    if has_beam:
        decoder = build_decoder(vocab, kenlm_path, alpha=alpha, beta=beta)

    # ── Single file transcription ───────────────────────────────────
    if audio:
        if long:
            # Chunked mode for full songs/podcasts
            transcribe_long(
                model, decoder, vocab, audio, device,
                beam_width=beam_width, temperature=temperature,
                chunk_seconds=15, overlap_seconds=2
            )
        else:
            # Standard mode (truncates to max_seconds)
            transcribe(
                model, decoder, vocab, audio, device,
                beam_width=beam_width, temperature=temperature,
                max_seconds=max_seconds
            )
        return

    # ── Optuna Tuning ───────────────────────────────────────────────
    best_temp = temperature
    if tune:
        if not has_beam:
            print("❌ Cannot tune without pyctcdecode.")
        else:
            print(f"\n{'='*70}")
            print(f"🔍 Hyperparameter tuning on dev-clean...")
            print(f"{'='*70}")
            try:
                from eigenwave_asr.tuning import tune_all_params
                tune_loader = get_dataloader(
                    data_dir, "dev-clean", vocab, batch_size=4)
                best_alpha, best_beta, best_temp, _ = tune_all_params(
                    model, vocab, kenlm_path, tune_loader, device,
                    max_samples=300, beam_width=beam_width)
                alpha, beta = best_alpha, best_beta
                decoder = build_decoder(
                    vocab, kenlm_path, alpha=alpha, beta=beta)
            except ImportError as e:
                print(f"⚠️  {e}")
                print("   Using pre-optimized defaults.")

    # ── Evaluate test sets ──────────────────────────────────────────
    test_subsets = ["test-clean", "test-other"]
    all_results = {}
    for subset in test_subsets:
        print(f"\n{'='*70}")
        print(f"📊 Evaluating: {subset}")
        print(f"   α={alpha:.4f}, β={beta:.4f}, T={best_temp:.4f}, "
              f"beam={beam_width}")
        print(f"{'='*70}")
        try:
            sub_loader = get_dataloader(
                data_dir, subset, vocab, batch_size=4)
            result = evaluate(
                model, decoder, vocab, sub_loader, device,
                beam_width=beam_width, max_samples=max_samples,
                temperature=best_temp)
            all_results[subset] = result
        except Exception as e:
            print(f"  ⚠️ Could not evaluate {subset}: {e}")

    # ── Final Summary ───────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print(f"🏆 FINAL RESULTS")
        print(f"{'='*70}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Config:     α={alpha:.4f} β={beta:.4f} "
              f"T={best_temp:.4f} beam={beam_width}")
        print()
        print(f"  {'Subset':<15} {'Greedy':<12} {'Beam+LM':<12} "
              f"{'Improve':<12} {'N':<8}")
        print(f"  {'─'*59}")
        for subset, result in all_results.items():
            g = result['greedy_wer']
            b = result['beam_wer']
            n = result['n_samples']
            print(f"  {subset:<15} {g:<12.2%} {b:<12.2%} "
                  f"{g-b:<12.2%} {n:<8}")
        print(f"  {'─'*59}")
        print(f"{'='*70}")

    run_cpu_benchmark(model, vocab, data_dir, alpha, beta, best_temp)
    model.to(device)
    return all_results


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    in_notebook = any(
        x in sys.argv[0].lower()
        for x in ['jupyter', 'ipykernel', 'colab']
    ) if sys.argv else False

    if in_notebook or len(sys.argv) <= 1:
        main()
    else:
        import argparse
        parser = argparse.ArgumentParser(
            description="EigenWave-ASR Inference"
        )
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--lm_dir", type=str, default="kenlm_models")
        parser.add_argument("--alpha", type=float, default=0.9268)
        parser.add_argument("--beta", type=float, default=0.061)
        parser.add_argument("--temperature", type=float, default=0.536)
        parser.add_argument("--beam_width", type=int, default=50)
        parser.add_argument("--max_samples", type=int, default=None)
        parser.add_argument("--max_seconds", type=int, default=20,
                          help="Max audio duration in seconds (default: 20)")
        parser.add_argument("--tune", action="store_true")
        parser.add_argument("--long", action="store_true",
                          help="Chunked mode for long audio (songs, podcasts)")
        parser.add_argument("--audio", type=str, default=None)
        parser.add_argument("--data_dir", type=str, default=None)
        args = parser.parse_args()
        main(**vars(args))