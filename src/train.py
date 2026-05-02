"""train.py
Training script for HoloAttractor TDNN model.
Adam lr=1e-3, 20 epochs, batch=64, triplet loss margin=0.3.
Dataset: 23,021 sequences from 83MB JPEG corpus.
Author: Ege Berk Turk, Kadir Has University
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple

from phase_encoder import encode_file
from tdnn_model import build_model
from triplet_loss import TripletLoss


# -----------------------------------------------------------------------
# Hyperparameters (paper values)
# -----------------------------------------------------------------------
LR           = 1e-3
EPOCHS       = 20
BATCH_SIZE   = 64
MARGIN       = 0.3
SEQ_LEN      = 4
EMBED_DIM    = 128
SEED         = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class PhaseSequenceDataset(Dataset):
    """Dataset of (anchor, positive, negative) phase-sequence triplets.

    Anchor and positive come from the same file (different windows).
    Negative comes from a different file.
    """

    def __init__(self, data_dir: str, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.sequences: List[Tuple[np.ndarray, str]] = []  # (seq, file_id)

        # Collect phase sequences from all files
        file_paths = sorted(Path(data_dir).rglob('*'))
        file_paths = [p for p in file_paths if p.is_file()]

        for fpath in file_paths:
            try:
                vecs = encode_file(str(fpath))  # (T, 512)
                file_id = str(fpath)
                for start in range(len(vecs) - seq_len + 1):
                    window = vecs[start:start + seq_len].astype(np.float32)
                    self.sequences.append((window, file_id))
            except Exception:
                continue

        # Group by file_id for triplet sampling
        self.file_to_seqs: dict = {}
        for i, (_, fid) in enumerate(self.sequences):
            self.file_to_seqs.setdefault(fid, []).append(i)

        self.file_ids = list(self.file_to_seqs.keys())
        print(f'Dataset: {len(self.sequences):,} sequences from {len(self.file_ids)} files')

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_seq, anchor_fid = self.sequences[idx]

        # Positive: another window from the same file
        pos_idxs = [i for i in self.file_to_seqs[anchor_fid] if i != idx]
        if pos_idxs:
            pos_seq = self.sequences[random.choice(pos_idxs)][0]
        else:
            pos_seq = anchor_seq  # fallback: same window

        # Negative: a window from a different file
        neg_fid = random.choice([f for f in self.file_ids if f != anchor_fid])
        neg_idx = random.choice(self.file_to_seqs[neg_fid])
        neg_seq = self.sequences[neg_idx][0]

        return (
            torch.from_numpy(anchor_seq),
            torch.from_numpy(pos_seq),
            torch.from_numpy(neg_seq),
        )


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train(
    data_dir: str,
    save_path: str = 'checkpoint.pt',
    device: str = 'cpu',
) -> None:
    set_seed()

    dataset = PhaseSequenceDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device != 'cpu'),
    )

    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = TripletLoss(margin=MARGIN)

    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for anchor, pos, neg in loader:
            anchor = anchor.to(device)  # (B, 4, 512)
            pos    = pos.to(device)
            neg    = neg.to(device)

            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)

            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f'  -> Saved checkpoint to {save_path}')

    print(f'Training complete. Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HoloAttractor TDNN')
    parser.add_argument('data_dir', type=str, help='Directory of training files')
    parser.add_argument('--save', type=str, default='checkpoint.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    EPOCHS = args.epochs
    print(f'Training on {args.device} for {EPOCHS} epochs')
    train(args.data_dir, save_path=args.save, device=args.device)
