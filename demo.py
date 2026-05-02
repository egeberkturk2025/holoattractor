"""demo.py
Quick demonstration of HoloAttractor retrieval pipeline.
Runs without any training data -- uses untrained TDNN weights.
Usage:  python demo.py [--query FILE] [--corpus DIR]
Author: Ege Berk Turk, Kadir Has University
"""
import sys
import os
import argparse
import tempfile
import numpy as np

# Make src/ importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phase_encoder import encode_file, extract_phase_vector, PHASE_DIM
from tdnn_model import build_model
from cosine_lsh import CosineLSH
from retrieval import HoloRetriever


def demo_phase_encoding():
    """Demo 1: Encode a synthetic byte sequence and show phase vector."""
    print('=== Demo 1: Phase Encoding ===')
    rng = np.random.default_rng(42)
    fake_chunk = bytes(rng.integers(0, 256, 4096, dtype=np.uint8))
    vec = extract_phase_vector(fake_chunk)
    print(f'Input:  4096-byte chunk')
    print(f'Output: {PHASE_DIM}-dim unit phase vector')
    print(f'Norm:   {np.linalg.norm(vec):.6f}  (should be ~1.0)')
    print(f'Mean:   {vec.mean():.4f}  Std: {vec.std():.4f}')
    print()


def demo_tdnn_embedding():
    """Demo 2: Embed a 4-timestep phase sequence with TDNN."""
    import torch
    print('=== Demo 2: TDNN Embedding ===')
    model = build_model()
    model.eval()
    print(f'Model parameters: {model.count_params():,}')
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((4, 512)).astype(np.float32)
    x = torch.from_numpy(seq).unsqueeze(0)  # (1, 4, 512)
    import torch.nn.functional as F
    with torch.no_grad():
        emb = model(x)
    print(f'Input:  (1, 4, 512) phase sequence')
    print(f'Output: {emb.shape}  L2 norm={emb.norm().item():.6f}')
    print()


def demo_lsh_index():
    """Demo 3: Build and query a small LSH index."""
    print('=== Demo 3: Cosine LSH Index ===')
    index = CosineLSH()
    rng = np.random.default_rng(7)
    # Index 50 random unit vectors
    for i in range(50):
        v = rng.standard_normal(128).astype(np.float32)
        v /= np.linalg.norm(v)
        index.add(f'doc_{i:03d}', v)
    # Query with a random vector
    q = rng.standard_normal(128).astype(np.float32)
    q /= np.linalg.norm(q)
    results = index.query(q, top_k=3)
    print(f'Indexed 50 vectors with 128 hyperplanes')
    print('Top-3 nearest neighbours:')
    for key, sim in results:
        print(f'  {key}  cosine={sim:.4f}')
    print()


def demo_file_retrieval(corpus_dir: str = None, query_file: str = None):
    """Demo 4: Index a directory and retrieve similar files."""
    print('=== Demo 4: File Retrieval ===')

    # If no real files provided, create synthetic temp files
    with tempfile.TemporaryDirectory() as tmp:
        if corpus_dir is None:
            corpus_dir = tmp
            rng = np.random.default_rng(99)
            for i in range(5):
                fpath = os.path.join(tmp, f'synthetic_{i}.bin')
                data = bytes(rng.integers(0, 256, 8192, dtype=np.uint8))
                with open(fpath, 'wb') as f:
                    f.write(data)

        if query_file is None:
            # Use first file in corpus as query
            candidates = [os.path.join(corpus_dir, fn)
                          for fn in os.listdir(corpus_dir)
                          if os.path.isfile(os.path.join(corpus_dir, fn))]
            if not candidates:
                print('No files found in corpus_dir, skipping.')
                return
            query_file = candidates[0]

        retriever = HoloRetriever()
        n = retriever.index_directory(corpus_dir)
        print(f'Indexed {n} files from {corpus_dir}')
        results = retriever.query(query_file, top_k=3)
        print(f'Query: {os.path.basename(query_file)}')
        print('Top-3 similar files:')
        for rank, (fpath, sim) in enumerate(results, 1):
            print(f'  {rank}. {os.path.basename(fpath)}  sim={sim:.4f}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HoloAttractor demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n  python demo.py\n  python demo.py --query img.jpg --corpus ./images'
    )
    parser.add_argument('--query',  type=str, default=None, help='Query file path')
    parser.add_argument('--corpus', type=str, default=None, help='Corpus directory')
    args = parser.parse_args()

    print('HoloAttractor Demo')
    print('Phase-Based Holographic Content-Addressable Memory')
    print('Author: Ege Berk Turk | Kadir Has University')
    print('-' * 55)
    print()

    demo_phase_encoding()
    demo_tdnn_embedding()
    demo_lsh_index()
    demo_file_retrieval(args.corpus, args.query)

    print('Demo complete. See src/ and experiments/ for full implementation.')
