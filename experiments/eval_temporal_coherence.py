"""eval_temporal_coherence.py
Measures temporal coherence gap = 0.78 between adjacent and distant sequence pairs.
TDNN trained with triplet bias achieves this gap.
Author: Ege Berk Turk, Kadir Has University
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase_encoder import encode_file
from tdnn_model import build_model


TEMPORAL_GAP_TARGET = 0.78
SEQ_LEN = 4


def embed_sequences(file_path: str, model, device: str = 'cpu') -> np.ndarray:
    """Get per-window TDNN embeddings for a file."""
    from phase_encoder import encode_file as _encode
    vecs = _encode(file_path)
    if len(vecs) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(vecs), 512), dtype=np.float32)
        vecs = np.concatenate([vecs, pad], axis=0)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(len(vecs) - SEQ_LEN + 1):
            w = torch.from_numpy(vecs[i:i+SEQ_LEN]).unsqueeze(0).to(device)
            e = model(w).squeeze(0).cpu().numpy()
            embeddings.append(e)
    return np.stack(embeddings) if embeddings else np.zeros((1, 128))


def eval_temporal_coherence(
    data_dir: str,
    checkpoint: str = None,
    device: str = 'cpu',
    max_files: int = 50,
) -> dict:
    """Compute adjacent vs. distant temporal coherence gap.

    Adjacent pair:  windows i and i+1 in the same file.
    Distant pair:   windows i and i+5 (or end) in the same file.

    Returns:
        dict with adjacent_sim, distant_sim, coherence_gap, paper_target.
    """
    model = build_model().to(device)
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    files = list(Path(data_dir).rglob('*.*'))[:max_files]
    files = [f for f in files if f.is_file()]

    adjacent_sims = []
    distant_sims  = []

    for fpath in files:
        try:
            embs = embed_sequences(str(fpath), model, device)  # (N, 128)
            N = len(embs)
            if N < 2:
                continue
            # Adjacent: consecutive windows
            for i in range(N - 1):
                s = float(np.dot(embs[i], embs[i+1]))
                adjacent_sims.append(s)
            # Distant: 5 steps apart
            step = min(5, N - 1)
            for i in range(N - step):
                s = float(np.dot(embs[i], embs[i+step]))
                distant_sims.append(s)
        except Exception as e:
            print(f'Warning: {fpath}: {e}')

    adj  = float(np.mean(adjacent_sims)) if adjacent_sims else 0.0
    dist = float(np.mean(distant_sims))  if distant_sims  else 0.0
    gap  = adj - dist

    return {
        'adjacent_sim':    adj,
        'distant_sim':     dist,
        'coherence_gap':   gap,
        'n_adjacent':      len(adjacent_sims),
        'n_distant':       len(distant_sims),
        'paper_target':    TEMPORAL_GAP_TARGET,
    }


if __name__ == '__main__':
    data_dir   = sys.argv[1] if len(sys.argv) > 1 else 'data'
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else 'checkpoint.pt'
    print(f'Evaluating temporal coherence on: {data_dir}')
    results = eval_temporal_coherence(data_dir, checkpoint)
    print()
    print('=== Temporal Coherence Results ===')
    print(f'Adjacent window similarity  : {results["adjacent_sim"]:.4f}')
    print(f'Distant  window similarity  : {results["distant_sim"]:.4f}')
    print(f'Coherence gap              : {results["coherence_gap"]:.4f}  (paper: {TEMPORAL_GAP_TARGET})')
