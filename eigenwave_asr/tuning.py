"""
Optuna Bayesian hyperparameter tuning — LAZY import, no crash if missing.

Only imported when tune=True is explicitly set.
"""

import gc
import time
import torch
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder

from .lm_utils import ensure_lowercase_arpa, extract_unigrams_from_arpa
from .decoder import wer


def _import_optuna():
    """Lazy import optuna — only when actually needed."""
    try:
        import optuna
        from optuna.samplers import TPESampler
        return optuna, TPESampler
    except ImportError:
        raise ImportError(
            "\n❌ optuna is required for hyperparameter tuning.\n"
            "   Install: pip install optuna\n"
            "   Or run without --tune flag to use pre-optimized defaults."
        )
    except ValueError as e:
        # numpy compatibility issue
        raise ImportError(
            f"\n❌ optuna has a dependency conflict: {e}\n"
            "   Try: pip install 'optuna==3.6.1'\n"
            "   Or:  pip install 'numpy>=2.0' optuna\n"
            "   Or run without --tune flag to use pre-optimized defaults."
        )


def _decode_batch(all_logits, decoder, temperature, beam_width):
    preds = []
    for lg in all_logits:
        lp = F.log_softmax(lg / temperature, dim=-1).numpy()
        text = decoder.decode(lp, beam_width=beam_width)
        preds.append(text)
    return preds


def _discover_alpha_beta(decoder):
    """Auto-discover alpha/beta attributes in pyctcdecode."""
    for cattr in ['_language_model', 'language_model', '_model_container',
                   'model_container', '_lm', 'lm']:
        container = getattr(decoder, cattr, None)
        if container is None:
            continue
        alpha_name, beta_name = None, None
        for attr in dir(container):
            if attr.startswith('__'):
                continue
            val = getattr(container, attr, None)
            if isinstance(val, (int, float)):
                a_lower = attr.lower()
                if 'alpha' in a_lower or a_lower == 'lm_weight':
                    alpha_name = attr
                elif 'beta' in a_lower or 'word' in a_lower:
                    beta_name = attr
        if alpha_name and beta_name:
            return container, alpha_name, beta_name
    return None, None, None


def tune_all_params(model, vocab, kenlm_path, data_loader, device,
                    beam_width=100, max_samples=300):
    """
    Optuna TPE tuning — builds decoder ONCE, modifies in-place.
    
    Returns: (best_alpha, best_beta, best_temperature, trial_log)
    """
    # Lazy import — only crashes here if optuna missing, not at startup
    optuna, TPESampler = _import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n{'='*70}")
    print(f"🔍 OPTUNA BAYESIAN HYPERPARAMETER TUNING")
    print(f"   Beam width: {beam_width} | Samples: {max_samples}")
    print(f"{'='*70}\n")

    # ── Pre-compute logits ──────────────────────────────────────────
    model.eval()
    all_logits, all_refs = [], []

    print("⏳ Pre-computing logits...")
    t0 = time.time()
    with torch.no_grad():
        n = 0
        for batch in data_loader:
            wavs, w_lens, toks, t_lens, texts = batch
            wavs = wavs.to(device)
            logits = model(wavs)
            for i in range(logits.size(0)):
                all_logits.append(logits[i].cpu())
                all_refs.append(texts[i])
                n += 1
                if n >= max_samples:
                    break
            if n >= max_samples:
                break
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ {len(all_logits)} samples in {time.time()-t0:.1f}s\n")

    # ── Build decoder ONCE ──────────────────────────────────────────
    if kenlm_path and kenlm_path.endswith('.arpa'):
        kenlm_path = ensure_lowercase_arpa(kenlm_path)

    unigrams = None
    if kenlm_path and (kenlm_path.endswith('.arpa') or kenlm_path.endswith('.arpa.gz')):
        unigrams = extract_unigrams_from_arpa(kenlm_path, set(vocab.chars))

    labels = vocab.get_labels_for_decoder()

    print("🔧 Building decoder (KenLM loads ONCE)...")
    decoder = build_ctcdecoder(
        labels=labels, kenlm_model_path=kenlm_path,
        unigrams=unigrams, alpha=0.5, beta=1.0,
    )
    print("✅ Decoder ready\n")

    # ── Discover alpha/beta ─────────────────────────────────────────
    container, alpha_attr, beta_attr = _discover_alpha_beta(decoder)

    if container is not None:
        # Verify in-place works
        test_lp = F.log_softmax(all_logits[0], dim=-1).numpy()
        text_a = decoder.decode(test_lp, beam_width=50)
        setattr(container, alpha_attr, 2.0)
        setattr(container, beta_attr, 0.0)
        text_b = decoder.decode(test_lp, beam_width=50)
        setattr(container, alpha_attr, 0.5)
        setattr(container, beta_attr, 1.0)
        if text_a == text_b:
            container = None

    if container is None:
        print("⚠️  In-place update not supported — returning defaults")
        return 0.3, 1.5, 0.8, []

    print("✅ In-place parameter updates verified\n")

    # ── Run Optuna ──────────────────────────────────────────────────
    trial_log = []
    best_so_far = [1.0]

    def objective(trial):
        a = trial.suggest_float('alpha', 0.001, 2.0, log=True)
        b = trial.suggest_float('beta', 0.0, 5.0)
        t = trial.suggest_float('temperature', 0.5, 1.3)
        setattr(container, alpha_attr, a)
        setattr(container, beta_attr, b)
        preds = _decode_batch(all_logits, decoder, t, beam_width)
        w = wer(preds, all_refs)
        is_best = w < best_so_far[0]
        if is_best:
            best_so_far[0] = w
        trial_log.append({'trial': trial.number, 'alpha': a,
                          'beta': b, 'temp': t, 'wer': w})
        marker = " ⭐" if is_best else ""
        print(f"  Trial {trial.number:3d}: α={a:.4f} β={b:.3f} "
              f"T={t:.3f} → WER={w:.4%}{marker}")
        return w

    n_trials = 100
    print(f"📊 Running {n_trials} TPE trials...\n")

    sampler = TPESampler(seed=42, n_startup_trials=20, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    for p in [{'alpha': 0.3, 'beta': 1.5, 'temperature': 1.0},
              {'alpha': 0.3, 'beta': 1.5, 'temperature': 0.8},
              {'alpha': 0.1, 'beta': 1.0, 'temperature': 0.8},
              {'alpha': 0.5, 'beta': 2.0, 'temperature': 0.8}]:
        study.enqueue_trial(p)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_alpha, best_beta, best_temp = best['alpha'], best['beta'], best['temperature']
    best_wer = study.best_value

    # ── Beam width sweep ────────────────────────────────────────────
    print(f"\n📊 Beam width sweep...")
    best_bw = beam_width
    setattr(container, alpha_attr, best_alpha)
    setattr(container, beta_attr, best_beta)

    for bw in [10, 20, 50, 100, 200, 500]:
        preds = _decode_batch(all_logits, decoder, best_temp, bw)
        w = wer(preds, all_refs)
        marker = " ⭐" if w < best_wer else ""
        print(f"    beam={bw:3d} → WER={w:.4%}{marker}")
        if w < best_wer:
            best_wer = w
            best_bw = bw

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🏆 TUNING COMPLETE")
    print(f"   α={best_alpha:.4f}  β={best_beta:.4f}  "
          f"T={best_temp:.4f}  beam={best_bw}  WER={best_wer:.4%}")
    print(f"{'='*70}\n")

    return best_alpha, best_beta, best_temp, trial_log
