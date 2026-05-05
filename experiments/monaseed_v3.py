"""monaseed_v3.py
MonaSeed v3 - DCT seed + tiled residual codec.
Evaluates seed-only and seed+residual reconstruction quality
against JPEG baseline on a 480x716 test image.

Author: Ege Berk Turk
License: MIT
"""
import io
import struct
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dct2(x: np.ndarray) -> np.ndarray:
    return dctn(x.astype(np.float64), norm='ortho')


def _idct2(x: np.ndarray) -> np.ndarray:
    return idctn(x.astype(np.float64), norm='ortho')


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(np.mean((original.astype(np.float64) -
                         reconstructed.astype(np.float64)) ** 2))
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------------------
# Seed codec
# ---------------------------------------------------------------------------

class SeedCodec:
    """Encode an image as a small set of top-energy DCT coefficients."""

    def __init__(self, seed_fraction: float = 0.005):
        self.seed_fraction = seed_fraction

    def encode(self, gray: np.ndarray):
        """Return (seed_bytes, dct_full, top_k_mask)."""
        h, w = gray.shape
        dct = _dct2(gray)

        n_total = h * w
        k = max(1, int(round(n_total * self.seed_fraction)))

        # Select top-k by absolute magnitude
        flat = dct.ravel()
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices]

        # Pack: header + index/value pairs
        header = struct.pack('>HHI', h, w, k)
        idx_bytes = indices.astype(np.int32).tobytes()
        val_bytes = values.astype(np.float32).tobytes()
        seed_bytes = zlib.compress(header + idx_bytes + val_bytes, level=9)
        return seed_bytes, dct, indices

    def decode(self, seed_bytes: bytes):
        """Return reconstructed grayscale float64 array."""
        raw = zlib.decompress(seed_bytes)
        h, w, k = struct.unpack('>HHI', raw[:8])
        idx = np.frombuffer(raw[8: 8 + k * 4], dtype=np.int32)
        val = np.frombuffer(raw[8 + k * 4:], dtype=np.float32)

        dct_sparse = np.zeros(h * w, dtype=np.float64)
        dct_sparse[idx] = val.astype(np.float64)
        dct_sparse = dct_sparse.reshape(h, w)
        return np.clip(_idct2(dct_sparse), 0, 255)


# ---------------------------------------------------------------------------
# Residual codec (tiled 8x8 block DCT)
# ---------------------------------------------------------------------------

class ResidualCodec:
    def __init__(self, block_size: int = 8, quant_step: int = 16):
        self.block_size = block_size
        self.quant_step = quant_step

    def _pad(self, arr):
        b = self.block_size
        h, w = arr.shape
        ph, pw = (-h) % b, (-w) % b
        return np.pad(arr, ((0, ph), (0, pw))), h, w

    def _block_dct(self, arr):
        b = self.block_size
        h, w = arr.shape
        out = np.empty_like(arr, dtype=np.float32)
        for i in range(0, h, b):
            for j in range(0, w, b):
                out[i:i+b, j:j+b] = dctn(
                    arr[i:i+b, j:j+b].astype(np.float32), norm='ortho')
        return out

    def _block_idct(self, arr):
        b = self.block_size
        h, w = arr.shape
        out = np.empty_like(arr, dtype=np.float32)
        for i in range(0, h, b):
            for j in range(0, w, b):
                out[i:i+b, j:j+b] = idctn(
                    arr[i:i+b, j:j+b].astype(np.float32), norm='ortho')
        return out

    def compress(self, residual: np.ndarray) -> bytes:
        padded, orig_h, orig_w = self._pad(residual.astype(np.float32))
        dct_coeffs = self._block_dct(padded)
        quantised = np.round(dct_coeffs / self.quant_step).astype(np.int16)
        header = struct.pack('>HHhH', orig_h, orig_w,
                             self.quant_step, self.block_size)
        return zlib.compress(header + quantised.tobytes(), level=9)

    def decompress(self, data: bytes) -> np.ndarray:
        raw = zlib.decompress(data)
        orig_h, orig_w, quant_step, block_size = struct.unpack('>HHhH', raw[:8])
        b = block_size
        ph = orig_h + ((-orig_h) % b)
        pw = orig_w + ((-orig_w) % b)
        quantised = np.frombuffer(raw[8:], dtype=np.int16).reshape(ph, pw)
        dct_coeffs = quantised.astype(np.float32) * quant_step
        rec = self._block_idct(dct_coeffs)
        return rec[:orig_h, :orig_w]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def make_test_image(h: int = 480, w: int = 716, seed: int = 0) -> np.ndarray:
    """Generate a synthetic grayscale test image (0-255, uint8)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 220, (h, w), dtype=np.uint8)
    # Add smooth gradient to simulate natural image structure
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    gradient = (yy * xx * 80).astype(np.uint8)
    return np.clip(base.astype(np.int32) + gradient, 0, 255).astype(np.uint8)


def jpeg_size(arr: np.ndarray, quality: int = 85) -> int:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=quality)
    return buf.tell()


def jpeg_reconstruct(arr: np.ndarray, quality: int = 85) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


def run_benchmark(image: np.ndarray, seed_fractions=(0.001, 0.005, 0.02),
                  quant_step: int = 16):
    print(f"Image shape: {image.shape}")
    print()

    # JPEG baseline
    j_size = jpeg_size(image)
    j_rec = jpeg_reconstruct(image)
    j_psnr = psnr(image, j_rec)
    print(f"{'JPEG Q85 (baseline)':<35} size={j_size/1024:6.2f} kB  "
          f"PSNR={j_psnr:6.2f} dB")
    print('-' * 70)

    for alpha in seed_fractions:
        sc = SeedCodec(seed_fraction=alpha)
        seed_bytes, dct_full, indices = sc.encode(image.astype(np.float64))
        seed_rec = sc.decode(seed_bytes)
        seed_rec_u8 = np.clip(seed_rec, 0, 255).astype(np.uint8)

        seed_psnr = psnr(image, seed_rec_u8)
        seed_kb = len(seed_bytes) / 1024
        ratio = j_size / len(seed_bytes)

        # Residual
        residual = image.astype(np.float32) - seed_rec.astype(np.float32)
        rc = ResidualCodec(block_size=8, quant_step=quant_step)
        res_bytes = rc.compress(residual)
        res_rec = rc.decompress(res_bytes)
        full_rec = np.clip(seed_rec.astype(np.float32) + res_rec, 0, 255)
        full_psnr = psnr(image, full_rec.astype(np.uint8))
        total_kb = (len(seed_bytes) + len(res_bytes)) / 1024

        print(f"alpha={alpha*100:.1f}%")
        print(f"  Seed only  : {seed_kb:6.2f} kB  PSNR={seed_psnr:6.2f} dB  "
              f"({ratio:.1f}x smaller than JPEG)")
        print(f"  + Residual : {total_kb:6.2f} kB  PSNR={full_psnr:6.2f} dB  "
              f"(residual={len(res_bytes)/1024:.2f} kB, Q={quant_step})")
        print()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    image = make_test_image(480, 716)
    run_benchmark(image)
