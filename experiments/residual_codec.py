"""residual_codec.py
Tiled 8x8 block DCT residual codec for MonaSeed.
Author: Ege Berk Turk
License: MIT
"""
import numpy as np
import zlib
from scipy.fft import dctn, idctn


class ResidualCodec:
    """Compress/decompress an image residual using tiled 8x8 DCT + zlib."""

    def __init__(self, block_size: int = 8, quant_step: int = 16):
        self.block_size = block_size
        self.quant_step = quant_step

    # ------------------------------------------------------------------
    def _pad(self, arr: np.ndarray):
        """Pad array so height and width are multiples of block_size."""
        h, w = arr.shape[:2]
        ph = (-h) % self.block_size
        pw = (-w) % self.block_size
        if ph or pw:
            arr = np.pad(arr, ((0, ph), (0, pw)) if arr.ndim == 2
                         else ((0, ph), (0, pw), (0, 0)))
        return arr, h, w

    def _block_dct(self, arr: np.ndarray) -> np.ndarray:
        """Apply 2-D DCT to every (block_size x block_size) tile."""
        b = self.block_size
        h, w = arr.shape
        out = np.empty_like(arr, dtype=np.float32)
        for i in range(0, h, b):
            for j in range(0, w, b):
                out[i:i+b, j:j+b] = dctn(
                    arr[i:i+b, j:j+b].astype(np.float32),
                    norm='ortho'
                )
        return out

    def _block_idct(self, arr: np.ndarray) -> np.ndarray:
        """Invert the tiled DCT."""
        b = self.block_size
        h, w = arr.shape
        out = np.empty_like(arr, dtype=np.float32)
        for i in range(0, h, b):
            for j in range(0, w, b):
                out[i:i+b, j:j+b] = idctn(
                    arr[i:i+b, j:j+b].astype(np.float32),
                    norm='ortho'
                )
        return out

    # ------------------------------------------------------------------
    def compress(self, residual: np.ndarray) -> bytes:
        """Encode residual to compressed bytes.

        Parameters
        ----------
        residual : np.ndarray
            Float32 array of shape (H, W) or (H, W, C).

        Returns
        -------
        bytes
            Compressed payload.
        """
        is_color = residual.ndim == 3
        channels = [residual[:, :, c] for c in range(residual.shape[2])] \
            if is_color else [residual]

        parts = []
        orig_shape = residual.shape
        for ch in channels:
            padded, orig_h, orig_w = self._pad(ch)
            dct_coeffs = self._block_dct(padded)
            quantised = np.round(dct_coeffs / self.quant_step).astype(np.int16)
            parts.append(quantised.tobytes())

        header = np.array(
            [orig_shape[0], orig_shape[1],
             orig_shape[2] if is_color else 1,
             self.quant_step, self.block_size],
            dtype=np.int32
        ).tobytes()
        payload = header + b''.join(parts)
        return zlib.compress(payload, level=9)

    def decompress(self, data: bytes) -> np.ndarray:
        """Decode compressed bytes back to a float32 residual array."""
        payload = zlib.decompress(data)
        header = np.frombuffer(payload[:20], dtype=np.int32)
        orig_h, orig_w, n_ch, quant_step, block_size = header
        self.quant_step = int(quant_step)
        self.block_size = int(block_size)

        b = self.block_size
        ph = orig_h + ((-orig_h) % b)
        pw = orig_w + ((-orig_w) % b)
        ch_bytes = ph * pw * 2  # int16 = 2 bytes

        channels = []
        offset = 20
        for _ in range(n_ch):
            raw = np.frombuffer(payload[offset:offset + ch_bytes], dtype=np.int16)
            offset += ch_bytes
            quantised = raw.reshape(ph, pw).astype(np.float32)
            dct_coeffs = quantised * self.quant_step
            reconstructed = self._block_idct(dct_coeffs)
            channels.append(reconstructed[:orig_h, :orig_w])

        if n_ch == 1:
            return channels[0]
        return np.stack(channels, axis=-1)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal((480, 716)).astype(np.float32) * 30.0

    codec = ResidualCodec(block_size=8, quant_step=16)
    t0 = time.time()
    compressed = codec.compress(dummy)
    t1 = time.time()
    recovered = codec.decompress(compressed)
    t2 = time.time()

    mse = float(np.mean((dummy - recovered) ** 2))
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    print(f"Compressed size : {len(compressed) / 1024:.2f} kB")
    print(f"Compress time   : {(t1-t0)*1000:.1f} ms")
    print(f"Decompress time : {(t2-t1)*1000:.1f} ms")
    print(f"Residual PSNR   : {psnr:.2f} dB")
