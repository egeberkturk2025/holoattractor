"""tdnn_model.py
TDNN (Time-Delay Neural Network) for temporal sequence embedding.
Architecture: 512->256->128->128, dilation=2, 411K params.
Author: Ege Berk Turk, Kadir Has University
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNLayer(nn.Module):
    """Single TDNN layer with configurable context and dilation."""

    def __init__(self, in_dim: int, out_dim: int, context: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim, out_dim,
            kernel_size=context,
            dilation=dilation,
            padding=dilation * (context - 1) // 2
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class HoloTDNN(nn.Module):
    """TDNN encoder for HoloAttractor phase sequences.

    Input:  (B, 4, 512)  -- 4 timesteps x 512-dim phase vectors
    Output: (B, 128)     -- L2-normalised embedding

    Total parameters: ~411,264
    Receptive field with dilation=2: 7 timesteps
    """

    def __init__(self, input_dim: int = 512, embed_dim: int = 128):
        super().__init__()
        # TDNN-1: 512->256, context=2, dilation=1
        self.tdnn1 = TDNNLayer(input_dim, 256, context=2, dilation=1)
        # TDNN-2: 256->128, context=3, dilation=2
        self.tdnn2 = TDNNLayer(256, 128, context=3, dilation=2)
        # TDNN-3: 128->128, context=2, dilation=1
        self.tdnn3 = TDNNLayer(128, 128, context=2, dilation=1)
        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Final projection
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, D) - batch, timesteps, phase_dim.

        Returns:
            emb: L2-normalised embedding (B, embed_dim).
        """
        # Conv1d expects (B, C, L): transpose to (B, D, T)
        x = x.transpose(1, 2)          # (B, 512, T)
        x = self.tdnn1(x)              # (B, 256, T)
        x = self.tdnn2(x)              # (B, 128, T)
        x = self.tdnn3(x)              # (B, 128, T)
        x = self.pool(x).squeeze(-1)   # (B, 128)
        x = self.fc(x)                 # (B, 128)
        return F.normalize(x, p=2, dim=1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(input_dim: int = 512, embed_dim: int = 128) -> HoloTDNN:
    """Factory function."""
    return HoloTDNN(input_dim=input_dim, embed_dim=embed_dim)


if __name__ == '__main__':
    model = build_model()
    print(f'Parameters: {model.count_params():,}')
    dummy = torch.randn(4, 4, 512)   # batch=4, T=4, D=512
    out = model(dummy)
    print(f'Output shape: {out.shape}')  # (4, 128)
    print(f'L2 norms: {torch.norm(out, dim=1)}')  # should be ~1.0
