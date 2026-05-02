"""phase_encoder.py
FFT-based phase vector extraction for HoloAttractor.
Author: Ege Berk Turk, Kadir Has University
"""
import numpy as np


CHUNK_SIZE = 4096   # bytes per chunk
PHASE_DIM  = 512    # output phase vector dimension (N/8 of FFT)


def extract_phase_vector(chunk: bytes) -> np.ndarray:
    """Convert a raw byte chunk into a 512-dim unit phase vector.

    Steps:
        1. Interpret bytes as float32 samples.
        2. Compute real FFT.
        3. Take angles of first N/8 complex bins.
        4. L2-normalise to unit sphere.

    Args:
        chunk: Raw bytes of length CHUNK_SIZE (4096).

    Returns:
        unit_vec: np.ndarray of shape (512,), float32.
    """
    # Pad or truncate to CHUNK_SIZE
    data = np.frombuffer(chunk.ljust(CHUNK_SIZE, b'\x00')[:CHUNK_SIZE],
                         dtype=np.uint8).astype(np.float32)

    # Real FFT -> complex spectrum
    spectrum = np.fft.rfft(data)  # length CHUNK_SIZE//2 + 1 = 2049

    # Take first PHASE_DIM bins (N/8 = 512)
    bins = spectrum[:PHASE_DIM]

    # Phase angles in [-pi, pi]
    phases = np.angle(bins).astype(np.float32)

    # L2 normalise
    norm = np.linalg.norm(phases)
    if norm < 1e-9:
        return np.zeros(PHASE_DIM, dtype=np.float32)
    return phases / norm


def encode_file(path: str) -> np.ndarray:
    """Encode an entire file into a matrix of phase vectors.

    Args:
        path: Path to file.

    Returns:
        vectors: np.ndarray of shape (T, PHASE_DIM) where T = number of chunks.
    """
    vectors = []
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            vectors.append(extract_phase_vector(chunk))
    if not vectors:
        return np.zeros((1, PHASE_DIM), dtype=np.float32)
    return np.stack(vectors, axis=0)  # (T, 512)


def delta_encode(vectors: np.ndarray) -> np.ndarray:
    """Apply delta encoding for 3.3x compression.

    Args:
        vectors: (T, PHASE_DIM)

    Returns:
        deltas: (T, PHASE_DIM) first row is original, rest are diffs.
    """
    deltas = np.diff(vectors, axis=0, prepend=vectors[:1])
    return deltas.astype(np.float32)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python phase_encoder.py <file>')
        sys.exit(1)
    vecs = encode_file(sys.argv[1])
    print(f'Encoded {len(vecs)} chunks, shape={vecs.shape}')
    print(f'Mean phase norm: {np.linalg.norm(vecs, axis=1).mean():.4f}')
