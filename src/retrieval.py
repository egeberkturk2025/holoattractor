"""retrieval.py
End-to-end retrieval pipeline: file -> phase vectors -> TDNN -> LSH query.
Author: Ege Berk Turk, Kadir Has University
"""
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

from phase_encoder import encode_file
from tdnn_model import build_model, HoloTDNN
from cosine_lsh import CosineLSH


SEQ_LEN = 4   # timesteps fed to TDNN (paper: 4 timesteps)


def embed_file(
    path: str,
    model: HoloTDNN,
    device: str = 'cpu',
) -> np.ndarray:
    """Encode a file into TDNN embeddings (one per sliding window).

    Args:
        path:   Path to file.
        model:  Loaded HoloTDNN model in eval mode.
        device: Torch device string.

    Returns:
        embeddings: (N_windows, 128) float32 numpy array.
    """
    phase_vecs = encode_file(path)   # (T, 512)
    T = len(phase_vecs)

    if T < SEQ_LEN:
        # Pad with zeros if file is too small
        pad = np.zeros((SEQ_LEN - T, phase_vecs.shape[1]), dtype=np.float32)
        phase_vecs = np.concatenate([phase_vecs, pad], axis=0)
        T = SEQ_LEN

    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(T - SEQ_LEN + 1):
            window = phase_vecs[start:start + SEQ_LEN]  # (4, 512)
            x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1,4,512)
            emb = model(x).squeeze(0).cpu().numpy()  # (128,)
            embeddings.append(emb)

    return np.stack(embeddings, axis=0)  # (N_windows, 128)


class HoloRetriever:
    """High-level retrieval interface.

    Usage::

        retriever = HoloRetriever('checkpoint.pt')
        retriever.index_directory('/path/to/corpus')
        results = retriever.query('/path/to/query_file.jpg', top_k=5)
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = 'cpu',
    ):
        self.device = device
        self.model = build_model().to(device)
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            self.model.load_state_dict(state)
        self.model.eval()
        self.index = CosineLSH()

    def index_file(self, path: str) -> None:
        """Add a single file to the retrieval index."""
        embs = embed_file(path, self.model, self.device)
        self.index.add_file(path, embs)

    def index_directory(self, directory: str, extensions: Tuple[str, ...] = ()) -> int:
        """Recursively index all matching files.

        Args:
            directory:  Root directory to scan.
            extensions: File extensions to include, e.g. ('.jpg', '.png').
                        Empty tuple means all files.

        Returns:
            Number of files indexed.
        """
        count = 0
        for root, _, files in os.walk(directory):
            for fname in files:
                if extensions and not fname.lower().endswith(extensions):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    self.index_file(fpath)
                    count += 1
                except Exception as e:
                    print(f'Warning: skipping {fpath}: {e}')
        return count

    def query(
        self,
        path: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve similar files for a query file.

        Args:
            path:   Query file path.
            top_k:  Maximum results to return.

        Returns:
            List of (file_path, similarity_score) sorted descending.
        """
        embs = embed_file(path, self.model, self.device)
        mean_emb = embs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-9:
            mean_emb /= norm
        return self.index.query(mean_emb, top_k=top_k)

    def save_index(self, path: str) -> None:
        self.index.save(path)

    def load_index(self, path: str) -> None:
        self.index = CosineLSH.load(path)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python retrieval.py <corpus_dir> <query_file>')
        sys.exit(1)
    corpus_dir, query_file = sys.argv[1], sys.argv[2]
    retriever = HoloRetriever()
    n = retriever.index_directory(corpus_dir)
    print(f'Indexed {n} files from {corpus_dir}')
    results = retriever.query(query_file, top_k=5)
    print(f'\nTop-5 similar to {query_file}:')
    for rank, (fpath, sim) in enumerate(results, 1):
        print(f'  {rank}. {fpath}  (sim={sim:.4f})')
