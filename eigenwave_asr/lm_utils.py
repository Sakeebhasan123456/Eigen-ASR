"""KenLM language model utilities — Windows compatible."""

import os
import time
import gzip
import shutil


def _find_existing_lm():
    """Search common locations for KenLM model."""
    candidates = [
        # Kaggle
        "/kaggle/input/librispeech-lm/4-gram.bin",
        r"C:\Users\SHAKIB\Downloads\eigenwave_asr_project (1)\kenlm_models\4-gram.arpa\4-gram.arpa"
        # Working directory
        "kenlm_models/4-gram.bin",
        "kenlm_models/4-gram.arpa",
        "kenlm_models/4-gram_lower.arpa",
        # Common local paths
        "./4-gram.arpa",
        "./4-gram.bin",
        "/data/lm/4-gram.arpa",
        "/data/lm/4-gram.bin",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c
    return None


def _decompress_gz(gz_path, out_path):
    """Decompress .gz file — works on Windows (no gunzip needed)."""
    print(f"   📦 Decompressing (Python gzip — works on all OS)...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"   ✅ Decompressed: {out_path}")
        return True
    except Exception as e:
        print(f"   ⚠️  Decompression failed: {e}")
        return False


def download_kenlm_model(lm_dir="kenlm_models"):
    """Download LibriSpeech 4-gram KenLM model."""
    os.makedirs(lm_dir, exist_ok=True)

    lm_path_gz = os.path.join(lm_dir, "4-gram.arpa.gz")
    lm_path = os.path.join(lm_dir, "4-gram.arpa")

    # Check if already exists
    if os.path.exists(lm_path) and os.path.isfile(lm_path):
        size = os.path.getsize(lm_path)
        if size > 1000000:  # > 1MB = real file, not empty
            print(f"✅ KenLM exists: {lm_path} ({size/1e9:.1f} GB)")
            return lm_path
        else:
            print(f"⚠️  {lm_path} is too small ({size} bytes) — re-downloading")
            os.remove(lm_path)

    # Check if .gz exists
    if os.path.exists(lm_path_gz):
        print(f"✅ Compressed file exists: {lm_path_gz}")
        if _decompress_gz(lm_path_gz, lm_path):
            return lm_path

    # Download
    url = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
    print(f"📥 Downloading 4-gram KenLM (~1.8 GB)...")
    print(f"   URL: {url}")
    print(f"   This will take a few minutes...\n")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, lm_path_gz)
        print(f"✅ Downloaded: {lm_path_gz}")

        # Decompress using Python (not gunzip!)
        if _decompress_gz(lm_path_gz, lm_path):
            return lm_path
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print(f"   Manual download: {url}")
        print(f"   Save as: {lm_path}")

    return None


def ensure_lowercase_arpa(arpa_path):
    """Convert ARPA to lowercase (one-time)."""
    if not arpa_path.endswith('.arpa'):
        return arpa_path

    lower_path = arpa_path.replace('.arpa', '_lower.arpa')
    if os.path.exists(lower_path):
        size = os.path.getsize(lower_path)
        if size > 1000000:
            print(f"   ✅ Lowercase ARPA exists: {lower_path}")
            return lower_path

    print(f"   🔄 Converting ARPA to lowercase (~1-2 min)...")
    start_t = time.time()
    line_count = 0

    try:
        with open(arpa_path, 'r', encoding='utf-8', errors='ignore') as fin:
            with open(lower_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    fout.write(line.lower())
                    line_count += 1
                    if line_count % 20_000_000 == 0:
                        elapsed = time.time() - start_t
                        print(f"      ... {line_count/1e6:.0f}M lines ({elapsed:.0f}s)")

        elapsed = time.time() - start_t
        print(f"   ✅ Done! {line_count:,} lines in {elapsed:.0f}s")
        return lower_path
    except Exception as e:
        print(f"   ⚠️  Failed: {e}")
        return arpa_path


def extract_unigrams_from_arpa(arpa_path, valid_chars=None, max_unigrams=500000):
    """Extract word-level unigrams from ARPA file."""
    unigrams = []
    in_unigram_section = False
    valid_chars = valid_chars or set("abcdefghijklmnopqrstuvwxyz'")

    print(f"   📖 Extracting unigrams...")

    try:
        with open(arpa_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line == '\\1-grams:':
                    in_unigram_section = True
                    continue
                if line.startswith('\\') and line != '\\1-grams:':
                    if in_unigram_section:
                        break
                    continue
                if not in_unigram_section or not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[1].strip().lower()
                    skip = {'<unk>', '<s>', '</s>', ''}
                    if word not in skip and all(c in valid_chars for c in word):
                        unigrams.append(word)
                if len(unigrams) >= max_unigrams:
                    break
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        return None

    unigrams = list(set(unigrams))
    print(f"   ✅ Extracted {len(unigrams):,} unigrams")
    return unigrams if unigrams else None


def get_kenlm_path(lm_dir="kenlm_models"):
    """Find or download KenLM model."""
    existing = _find_existing_lm()
    if existing:
        print(f"✅ Found KenLM: {existing}")
        return existing
    return download_kenlm_model(lm_dir)