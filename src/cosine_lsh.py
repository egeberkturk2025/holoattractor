"""cosine_lsh.py
Locality-Sensitive Hashing with 128 random hyperplanes for cosine similarity.
File similarity = mean cosine across chunk-pair embeddings.
Author: Ege Berk Turk, Kadir Has University
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional


NUM_HYPERPLANES = 128   # as per paper
EMBED_DIM = 128         # TDNN output dimension


class CosineLSH:
    """LSH index for approximate cosine nearest-neighbour search.

    Each vector is projected onto 128 random hyperplanes;
    the sign of each projection forms a binary hash code.
    Buckets with matching codes are retrieved as candidates.
    Final ranking uses exact cosine similarity.
    """

    def __init__(
        self,
        num_hyperplanes: int = NUM_HYPERPLANES,
        embed_dim: int = EMBED_DIM,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        # Random projection matrix  (num_hyperplanes, embed_dim)
        self.hyperplanes = rng.standard_normal((num_hyperplanes, embed_dim)).astype(
            np.float32
        )
        self.buckets: dict = {}          # hash_code -> list of (key, vector)
        self.all_keys: List[str] = []
        self.all_vecs: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def _hash(self, vec: np.ndarray) -> bytes:
        """Compute binary hash code as bytes."""
        projected = self.hyperplanes @ vec  # (128,)
        bits = (projected >= 0).astype(np.uint8)
        # Pack 8 bits per byte -> 16 bytes
        return np.packbits(bits).tobytes()

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    def add(self, key: str, vec: np.ndarray) -> None:
        """Insert a single vector into the index.

        Args:
            key: Identifier (e.g. file path).
            vec: (embed_dim,) float32, should be L2-normalised.
        """
        code = self._hash(vec)
        self.buckets.setdefault(code, []).append((key, vec))
        self.all_keys.append(key)
        self.all_vecs.append(vec)

    def add_file(self, key: str, vecs: np.ndarray) -> None:
        """Index a file represented as multiple chunk embeddings.

        The file-level hash uses the mean embedding.

        Args:
            key:  File identifier.
            vecs: (T, embed_dim) embeddings for all chunks.
        """
        mean_vec = vecs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-9:
            mean_vec /= norm
        self.add(key, mean_vec)

    def query(
        self,
        vec: np.ndarray,
        top_k: int = 10,
        probe_radius: int = 1,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k most similar items.

        Args:
            vec:          Query vector (embed_dim,), L2-normalised.
            top_k:        Number of results.
            probe_radius: Number of bit flips for multi-probe LSH.

        Returns:
            List of (key, cosine_similarity) sorted descending.
        """
        candidates: dict = {}
        code = self._hash(vec)

        # Primary bucket
        for item_key, item_vec in self.buckets.get(code, []):
            candidates[item_key] = item_vec

        # Multi-probe: flip up to probe_radius bits
        code_bits = np.unpackbits(np.frombuffer(code, dtype=np.uint8))
        for i in range(len(code_bits)):
            if probe_radius < 1:
                break
            flipped = code_bits.copy()
            flipped[i] = 1 - flipped[i]
            probe_code = np.packbits(flipped).tobytes()
            for item_key, item_vec in self.buckets.get(probe_code, []):
                candidates[item_key] = item_vec

        # Rank by exact cosine similarity
        results = [
            (k, float(np.dot(vec, v))) for k, v in candidates.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # File similarity (paper definition)
    # ------------------------------------------------------------------

    @staticmethod
    def file_similarity(vecs_a: np.ndarray, vecs_b: np.ndarray) -> float:
        """Mean cosine similarity across paired chunk embeddings.

        Args:
            vecs_a: (T_a, D)
            vecs_b: (T_b, D)

        Returns:
            Mean cosine similarity (scalar).
        """
        T = min(len(vecs_a), len(vecs_b))
        if T == 0:
            return 0.0
        sims = [
            float(np.dot(vecs_a[i], vecs_b[i])) for i in range(T)
        ]
        return float(np.mean(sims))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'CosineLSH':
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    index = CosineLSH()
    rng = np.random.default_rng(0)
    # Index 100 random unit vectors
    for i in range(100):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        index.add(f'item_{i}', v)
    # Query with a random vector
    q = rng.standard_normal(EMBED_DIM).astype(np.float32)
    q /= np.linalg.norm(q)
    results = index.query(q, top_k=5)
    print('Top-5 results:')
    for key, sim in results:
        print(f'  {key}: cosine={sim:.4f}')
