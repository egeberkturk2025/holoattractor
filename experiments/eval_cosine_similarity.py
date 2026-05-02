"""eval_cosine_similarity.py
Reproduces paper result: Phase-LSH cosine similarity = 0.91 across resolution variants.
Perceptual hashing baseline = 0.73.
Author: Ege Berk Turk, Kadir Has University
"""
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase_encoder import encode_file
from cosine_lsh import CosineLSH


def compute_pairwise_cosine(vecs_a: np.ndarray, vecs_b: np.ndarray) -> float:
    """Mean cosine similarity between matched chunk pairs."""
    T = min(len(vecs_a), len(vecs_b))
    if T == 0:
        return 0.0
    sims = [float(np.dot(vecs_a[i], vecs_b[i])) for i in range(T)]
    return float(np.mean(sims))


def eval_phase_lsh_similarity(corpus_dir: str) -> dict:
    """Evaluate cosine similarity for same-content variant pairs.

    Expects corpus_dir to contain sub-folders where each sub-folder
    holds resolution variants of the same image:
        corpus_dir/
            img_001/
                original.jpg
                resized_50pct.jpg
                resized_25pct.jpg
            img_002/
                ...

    Returns:
        dict with keys 'phase_lsh_sim' and 'n_pairs'.
    """
    same_sims = []
    diff_sims = []

    groups = []
    for group_dir in sorted(Path(corpus_dir).iterdir()):
        if not group_dir.is_dir():
            continue
        files = sorted(group_dir.glob('*.jpg')) + sorted(group_dir.glob('*.png'))
        if len(files) < 2:
            continue
        vecs_group = []
        for fpath in files:
            try:
                vecs = encode_file(str(fpath))
                vecs_group.append(vecs)
            except Exception as e:
                print(f'Warning: {fpath}: {e}')
        if len(vecs_group) >= 2:
            groups.append(vecs_group)

    # Same-content pairs (within group)
    for vecs_group in groups:
        for i in range(len(vecs_group)):
            for j in range(i + 1, len(vecs_group)):
                s = compute_pairwise_cosine(vecs_group[i], vecs_group[j])
                same_sims.append(s)

    # Different-content pairs (across groups)
    for gi in range(len(groups)):
        for gj in range(gi + 1, min(gi + 5, len(groups))):
            s = compute_pairwise_cosine(groups[gi][0], groups[gj][0])
            diff_sims.append(s)

    mean_same = float(np.mean(same_sims)) if same_sims else 0.0
    mean_diff = float(np.mean(diff_sims)) if diff_sims else 0.0

    return {
        'phase_lsh_sim':       mean_same,
        'cross_content_sim':   mean_diff,
        'n_same_pairs':        len(same_sims),
        'n_diff_pairs':        len(diff_sims),
        'paper_target':        0.91,
    }


if __name__ == '__main__':
    corpus = sys.argv[1] if len(sys.argv) > 1 else 'data/resolution_variants'
    print(f'Evaluating cosine similarity on: {corpus}')
    results = eval_phase_lsh_similarity(corpus)
    print()
    print('=== Results ===')
    print(f'Phase-LSH same-content sim : {results["phase_lsh_sim"]:.4f}  (paper: 0.91)')
    print(f'Cross-content sim          : {results["cross_content_sim"]:.4f}')
    print(f'Same-content pairs         : {results["n_same_pairs"]}')
    print(f'Different-content pairs    : {results["n_diff_pairs"]}')
    gap = results['phase_lsh_sim'] - results['cross_content_sim']
    print(f'Discriminative gap         : {gap:.4f}')
