"""triplet_loss.py
Triplet loss with cosine distance for HoloAttractor training.
L = max(0, d_a,p - d_a,n + margin),  d = 1 - cosine_sim
Author: Ege Berk Turk, Kadir Has University
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss using cosine distance.

    Formula from paper:
        L = max(0, d(a,p) - d(a,n) + margin)
        d = 1 - cosine_similarity

    Args:
        margin: Minimum desired gap between positive and negative distances.
                Paper uses 0.3.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            anchor:   (B, D) L2-normalised embeddings.
            positive: (B, D) L2-normalised embeddings.
            negative: (B, D) L2-normalised embeddings.

        Returns:
            loss: Scalar mean triplet loss over the batch.
        """
        # Cosine similarity in [-1, 1]
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)  # (B,)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)  # (B,)

        # Cosine distance in [0, 2]
        d_pos = 1.0 - sim_pos
        d_neg = 1.0 - sim_neg

        # Hinge loss
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet mining variant.

    For each anchor, selects hardest positive (most distant)
    and hardest negative (closest) within the batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch-hard triplet loss.

        Args:
            embeddings: (B, D) L2-normalised embeddings.
            labels:     (B,) integer class labels.

        Returns:
            loss: Scalar.
        """
        B = embeddings.size(0)
        # Pairwise cosine distance matrix
        sim = torch.mm(embeddings, embeddings.t())  # (B, B)
        dist = 1.0 - sim  # cosine distance

        label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)

        # Hardest positive: same label, max distance
        pos_dist = dist.clone()
        pos_dist[~label_mat] = 0.0
        hardest_pos, _ = pos_dist.max(dim=1)  # (B,)

        # Hardest negative: different label, min distance
        neg_dist = dist.clone()
        neg_dist[label_mat] = float('inf')
        hardest_neg, _ = neg_dist.min(dim=1)  # (B,)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


if __name__ == '__main__':
    loss_fn = TripletLoss(margin=0.3)
    a = F.normalize(torch.randn(8, 128), dim=1)
    p = F.normalize(torch.randn(8, 128), dim=1)
    n = F.normalize(torch.randn(8, 128), dim=1)
    print(f'Triplet loss: {loss_fn(a, p, n).item():.4f}')
