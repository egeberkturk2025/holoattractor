"""eval_format_bias.py
Quantifies JPEG format bias: compressed formats impose a similarity floor ~0.77.
Paper finding: DCT-based JPEG encoding creates a similarity floor that masks
semantic variation, motivating format-invariant phase encoding.
Author: Ege Berk Turk, Kadir Has University
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase_encoder import encode_file


FORMAT_BIAS_TARGET = 0.77  # paper value: JPEG similarity floor


def compute_mean_cosine(vecs_a: np.ndarray, vecs_b: np.ndarray) -> float:
    T = min(len(vecs_a), len(vecs_b))
    if T == 0:
        return 0.0
    return float(np.mean([np.dot(vecs_a[i], vecs_b[i]) for i in range(T)]))


def eval_format_bias(jpeg_dir: str, png_dir: str) -> dict:
    """Measure similarity floor imposed by JPEG compression.

    Compares JPEG files against their PNG equivalents.
    High similarity even for semantically different images
    indicates a format-induced bias (the paper's 'format bias').

    Args:
        jpeg_dir: Directory of JPEG images.
        png_dir:  Directory of corresponding PNG images (same basenames).

    Returns:
        dict with format_bias_floor, n_pairs, and paper_target.
    """
    jpeg_files = sorted(Path(jpeg_dir).glob('*.jpg'))
    sims = []

    for jf in jpeg_files:
        pf = Path(png_dir) / (jf.stem + '.png')
        if not pf.exists():
            continue
        try:
            vecs_j = encode_file(str(jf))
            vecs_p = encode_file(str(pf))
            sims.append(compute_mean_cosine(vecs_j, vecs_p))
        except Exception as e:
            print(f'Warning: {jf}: {e}')

    # Cross-format bias: random JPEG vs random PNG
    random_sims = []
    files_j = sorted(Path(jpeg_dir).glob('*.jpg'))[:20]
    for i, jf in enumerate(files_j):
        for jf2 in files_j[i+1:i+4]:
            try:
                vecs_a = encode_file(str(jf))
                vecs_b = encode_file(str(jf2))
                random_sims.append(compute_mean_cosine(vecs_a, vecs_b))
            except Exception:
                continue

    return {
        'format_bias_floor':    float(np.mean(sims)) if sims else 0.0,
        'cross_jpeg_sim':       float(np.mean(random_sims)) if random_sims else 0.0,
        'n_pairs':              len(sims),
        'paper_target':         FORMAT_BIAS_TARGET,
    }


if __name__ == '__main__':
    jpeg_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/jpeg'
    png_dir  = sys.argv[2] if len(sys.argv) > 2 else 'data/png'
    print(f'Evaluating format bias: JPEG={jpeg_dir}, PNG={png_dir}')
    results = eval_format_bias(jpeg_dir, png_dir)
    print()
    print('=== Format Bias Results ===')
    print(f'JPEG->PNG similarity floor : {results["format_bias_floor"]:.4f}  (paper: ~{FORMAT_BIAS_TARGET})')
    print(f'Cross-JPEG similarity      : {results["cross_jpeg_sim"]:.4f}')
    print(f'Pairs evaluated            : {results["n_pairs"]}')
    bias = results['format_bias_floor'] - results['cross_jpeg_sim']
    print(f'Net format bias            : {bias:+.4f}')
