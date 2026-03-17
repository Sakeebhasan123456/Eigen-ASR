"""
CTC Beam Search Decoder — works with or without KenLM.

If pyctcdecode is not installed, falls back to greedy-only mode.
"""

import os
import time
import torch
import torch.nn.functional as F

# ── Safe import ──────────────────────────────────────────────────────
_HAS_PYCTCDECODE = False
try:
    from pyctcdecode import build_ctcdecoder
    _HAS_PYCTCDECODE = True
except ImportError:
    pass


def check_dependencies():
    """Check if beam search dependencies are available."""
    if not _HAS_PYCTCDECODE:
        print("⚠️  pyctcdecode not installed!")
        print("   Install: pip install pyctcdecode")
        print("   Falling back to GREEDY decoding only.\n")
        return False
    return True


def wer(preds, refs):
    """Word Error Rate via edit distance."""
    errs, words = 0, 0
    for p, r in zip(preds, refs):
        pw, rw = p.lower().split(), r.lower().split()
        if not rw:
            continue
        m, n = len(rw), len(pw)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if rw[i-1] == pw[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        errs += dp[m][n]
        words += m
    return errs / max(words, 1)


def build_decoder(vocab, kenlm_model_path=None, alpha=0.5, beta=1.0):
    """Build CTC beam search decoder. Returns None if deps missing."""
    if not _HAS_PYCTCDECODE:
        print("⚠️  pyctcdecode not available — beam search disabled")
        return None

    labels = vocab.get_labels_for_decoder()
    print(f"\n🔧 Building beam search decoder...")
    print(f"   Vocab: {len(labels)} labels")

    if kenlm_model_path and os.path.exists(kenlm_model_path):
        # Lowercase ARPA if needed
        if kenlm_model_path.endswith('.arpa'):
            from .lm_utils import ensure_lowercase_arpa
            kenlm_model_path = ensure_lowercase_arpa(kenlm_model_path)

        print(f"   KenLM: {kenlm_model_path}")
        print(f"   α={alpha}, β={beta}")

        # Extract unigrams
        unigrams = None
        if kenlm_model_path.endswith('.arpa') or kenlm_model_path.endswith('.arpa.gz'):
            from .lm_utils import extract_unigrams_from_arpa
            valid_chars = set(vocab.chars)
            unigrams = extract_unigrams_from_arpa(kenlm_model_path, valid_chars)

        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_model_path,
            unigrams=unigrams,
            alpha=alpha, beta=beta,
        )
        print(f"   ✅ Decoder built WITH language model\n")
    else:
        if kenlm_model_path:
            print(f"   ⚠️  KenLM not found: {kenlm_model_path}")
        print(f"   Building decoder WITHOUT language model")
        decoder = build_ctcdecoder(labels=labels)

    return decoder


def evaluate(model, decoder, vocab, data_loader, device,
             beam_width=100, max_samples=None, temperature=1.0):
    """Evaluate with greedy and (optionally) beam search."""
    model.eval()
    greedy_preds, beam_preds, references = [], [], []
    n_samples = 0
    use_beam = decoder is not None

    print(f"\n{'='*70}")
    mode = f"beam_width={beam_width}, temp={temperature}" if use_beam else "greedy only"
    print(f"Running evaluation ({mode})...")
    print(f"{'='*70}")

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            wavs, w_lens, toks, t_lens, texts = batch
            wavs = wavs.to(device)
            logits = model(wavs)

            if use_beam:
                scaled_logits = logits / temperature
                log_probs = F.log_softmax(scaled_logits, dim=-1)

            for i in range(logits.size(0)):
                # Greedy
                greedy_ids = logits[i].argmax(dim=-1).cpu().tolist()
                greedy_text = vocab.decode_greedy(greedy_ids)
                greedy_preds.append(greedy_text)

                # Beam search (if available)
                if use_beam:
                    lp_np = log_probs[i].cpu().numpy()
                    beam_text = decoder.decode(lp_np, beam_width=beam_width)
                    beam_preds.append(beam_text)

                references.append(texts[i])
                n_samples += 1

            if max_samples and n_samples >= max_samples:
                break
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {n_samples} samples ({elapsed:.1f}s)")

    elapsed = time.time() - start_time

    g_wer = wer(greedy_preds, references)
    result = {
        'greedy_wer': g_wer,
        'beam_wer': g_wer,  # default = same as greedy
        'n_samples': n_samples,
        'greedy_preds': greedy_preds,
        'beam_preds': beam_preds if use_beam else greedy_preds,
        'references': references,
    }

    print(f"\n{'='*70}")
    print(f"RESULTS ({n_samples} samples, {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Greedy WER: {g_wer:.2%}")

    if use_beam:
        b_wer = wer(beam_preds, references)
        result['beam_wer'] = b_wer
        imp = g_wer - b_wer
        rel = (imp / g_wer * 100) if g_wer > 0 else 0
        print(f"  Beam WER:   {b_wer:.2%}")
        print(f"  Improvement: {imp:.2%} absolute ({rel:.1f}% relative)")

    print(f"{'='*70}")

    # Sample comparisons
    print(f"\n📋 Sample Comparisons (first 10):")
    print(f"{'─'*70}")
    for i in range(min(10, len(references))):
        ref = references[i].lower()
        grd = greedy_preds[i]
        g_ok = "✅" if grd.strip() == ref.strip() else "❌"
        print(f"  [{i+1}] REF:    {ref[:80]}")
        print(f"       GREEDY: {grd[:80]}  {g_ok}")
        if use_beam:
            bms = beam_preds[i]
            b_ok = "✅" if bms.strip() == ref.strip() else "❌"
            print(f"       BEAM:   {bms[:80]}  {b_ok}")
        print(f"{'─'*70}")

    return result
