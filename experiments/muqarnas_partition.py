"""muqarnas_partition.py
Fraktal (Mukarnas) partitioning: 4x4x4 = 64-way node bölmesi.
Octree baseline ile karsilastirmak icin P0 prototipi.

Yazar: Ege Berk Turk
Lisans: MIT
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ortak tip tanimlari
# ---------------------------------------------------------------------------
Point3D = np.ndarray  # shape (3,)
Ray = Tuple[Point3D, Point3D]  # (origin, direction) – direction normalized


@dataclass
class AABB:
    """Axis-Aligned Bounding Box."""
    min_pt: np.ndarray
    max_pt: np.ndarray

    def center(self) -> np.ndarray:
        return (self.min_pt + self.max_pt) * 0.5

    def half_extents(self) -> np.ndarray:
        return (self.max_pt - self.min_pt) * 0.5

    def contains(self, pt: np.ndarray) -> bool:
        return bool(np.all(pt >= self.min_pt) and np.all(pt <= self.max_pt))

    def intersects_ray(self, origin: np.ndarray, direction: np.ndarray,
                       hit_radius: float = 0.0) -> bool:
        """Slab-method AABB-ray intersection (with optional expansion)."""
        lo = self.min_pt - hit_radius
        hi = self.max_pt + hit_radius
        tmin, tmax = -np.inf, np.inf
        for i in range(3):
            if abs(direction[i]) < 1e-9:
                if origin[i] < lo[i] or origin[i] > hi[i]:
                    return False
            else:
                t1 = (lo[i] - origin[i]) / direction[i]
                t2 = (hi[i] - origin[i]) / direction[i]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))
        return tmax >= max(tmin, 0.0)


# ---------------------------------------------------------------------------
# Octree baseline
# ---------------------------------------------------------------------------

class OctreeNode:
    MAX_POINTS = 64
    MAX_DEPTH = 8

    def __init__(self, bounds: AABB, depth: int = 0):
        self.bounds = bounds
        self.depth = depth
        self.points: List[np.ndarray] = []
        self.children: Optional[List['OctreeNode']] = None

    def _split(self):
        c = self.bounds.center()
        mn, mx = self.bounds.min_pt, self.bounds.max_pt
        corners = [
            (mn, c),
            (np.array([c[0], mn[1], mn[2]]), np.array([mx[0], c[1], c[2]])),
            (np.array([mn[0], c[1], mn[2]]), np.array([c[0], mx[1], c[2]])),
            (np.array([c[0], c[1], mn[2]]), np.array([mx[0], mx[1], c[2]])),
            (np.array([mn[0], mn[1], c[2]]), np.array([c[0], c[1], mx[2]])),
            (np.array([c[0], mn[1], c[2]]), np.array([mx[0], c[1], mx[2]])),
            (np.array([mn[0], c[1], c[2]]), np.array([c[0], mx[1], mx[2]])),
            (c, mx),
        ]
        self.children = [
            OctreeNode(AABB(lo.copy(), hi.copy()), self.depth + 1)
            for lo, hi in corners
        ]
        for pt in self.points:
            for ch in self.children:
                if ch.bounds.contains(pt):
                    ch.insert(pt)
                    break
        self.points = []

    def insert(self, pt: np.ndarray):
        if self.children is not None:
            for ch in self.children:
                if ch.bounds.contains(pt):
                    ch.insert(pt)
                    return
        self.points.append(pt)
        if (len(self.points) > self.MAX_POINTS
                and self.depth < self.MAX_DEPTH):
            self._split()

    def query_ray(self, origin: np.ndarray, direction: np.ndarray,
                  hit_radius: float, stats: dict) -> List[np.ndarray]:
        stats['visited_nodes'] += 1
        if not self.bounds.intersects_ray(origin, direction, hit_radius):
            return []
        if self.children is not None:
            hits = []
            for ch in self.children:
                hits.extend(ch.query_ray(origin, direction, hit_radius, stats))
            return hits
        # Leaf: test each point
        hits = []
        for pt in self.points:
            stats['tested_points'] += 1
            d = np.cross(pt - origin, direction)
            if np.dot(d, d) <= hit_radius ** 2:
                hits.append(pt)
        return hits


class OctreeIndex:
    """Root-level octree wrapper."""

    def __init__(self, points: np.ndarray, hit_radius: float = 0.03):
        self.hit_radius = hit_radius
        mn = points.min(axis=0) - 1e-4
        mx = points.max(axis=0) + 1e-4
        self.root = OctreeNode(AABB(mn, mx))
        for pt in points:
            self.root.insert(pt)

    def query_ray(self, origin: np.ndarray,
                  direction: np.ndarray) -> Tuple[List[np.ndarray], dict]:
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        stats = {'visited_nodes': 0, 'tested_points': 0}
        hits = self.root.query_ray(origin, direction, self.hit_radius, stats)
        return hits, stats


# ---------------------------------------------------------------------------
# Muqarnas (fraktal) tree: 4x4x4 = 64-way branching
# ---------------------------------------------------------------------------

class MuqarnasNode:
    """64-way branching node (4 divisions per axis)."""
    DIVISIONS = 4          # divisions per axis -> 4^3 = 64 children
    MAX_POINTS = 64
    MAX_DEPTH = 5          # shallower limit to bound memory

    def __init__(self, bounds: AABB, depth: int = 0):
        self.bounds = bounds
        self.depth = depth
        self.points: List[np.ndarray] = []
        self.children: Optional[List['MuqarnasNode']] = None

    def _split(self):
        d = self.DIVISIONS
        mn, mx = self.bounds.min_pt, self.bounds.max_pt
        step = (mx - mn) / d
        self.children = []
        for ix in range(d):
            for iy in range(d):
                for iz in range(d):
                    lo = mn + step * np.array([ix, iy, iz])
                    hi = lo + step
                    self.children.append(
                        MuqarnasNode(AABB(lo.copy(), hi.copy()), self.depth + 1)
                    )
        for pt in self.points:
            idx = np.minimum(
                ((pt - mn) / step).astype(int), d - 1
            )
            child_idx = idx[0] * d * d + idx[1] * d + idx[2]
            self.children[child_idx].insert(pt)
        self.points = []

    def insert(self, pt: np.ndarray):
        if self.children is not None:
            d = self.DIVISIONS
            mn, mx = self.bounds.min_pt, self.bounds.max_pt
            step = (mx - mn) / d
            idx = np.minimum(((pt - mn) / step).astype(int), d - 1)
            child_idx = int(idx[0]) * d * d + int(idx[1]) * d + int(idx[2])
            self.children[child_idx].insert(pt)
            return
        self.points.append(pt)
        if (len(self.points) > self.MAX_POINTS
                and self.depth < self.MAX_DEPTH):
            self._split()

    def query_ray(self, origin: np.ndarray, direction: np.ndarray,
                  hit_radius: float, stats: dict) -> List[np.ndarray]:
        stats['visited_nodes'] += 1
        if not self.bounds.intersects_ray(origin, direction, hit_radius):
            return []
        if self.children is not None:
            hits = []
            for ch in self.children:
                hits.extend(
                    ch.query_ray(origin, direction, hit_radius, stats))
            return hits
        hits = []
        for pt in self.points:
            stats['tested_points'] += 1
            d_vec = np.cross(pt - origin, direction)
            if np.dot(d_vec, d_vec) <= hit_radius ** 2:
                hits.append(pt)
        return hits


class MuqarnasIndex:
    """Root-level Muqarnas (64-way) wrapper."""

    def __init__(self, points: np.ndarray, hit_radius: float = 0.03):
        self.hit_radius = hit_radius
        mn = points.min(axis=0) - 1e-4
        mx = points.max(axis=0) + 1e-4
        self.root = MuqarnasNode(AABB(mn, mx))
        for pt in points:
            self.root.insert(pt)

    def query_ray(self, origin: np.ndarray,
                  direction: np.ndarray) -> Tuple[List[np.ndarray], dict]:
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        stats = {'visited_nodes': 0, 'tested_points': 0}
        hits = self.root.query_ray(origin, direction, self.hit_radius, stats)
        return hits, stats


# ---------------------------------------------------------------------------
# Dataset uretici
# ---------------------------------------------------------------------------

def make_dataset(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Nokta bulutu uret.

    kind: 'uniform' | 'clustered' | 'surface'
    """
    if kind == 'uniform':
        return rng.uniform(0, 1, (n, 3)).astype(np.float32)
    elif kind == 'clustered':
        n_clusters = 8
        centers = rng.uniform(0.1, 0.9, (n_clusters, 3))
        pts = []
        per = n // n_clusters
        for c in centers:
            pts.append(rng.normal(c, 0.05, (per, 3)))
        pts_arr = np.clip(np.vstack(pts)[:n], 0, 1).astype(np.float32)
        return pts_arr
    elif kind == 'surface':
        # Kure yuzeyine yakin noktalar
        pts = rng.standard_normal((n, 3)).astype(np.float32)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / (norms + 1e-9)
        noise = rng.uniform(0.45, 0.55, (n, 1)).astype(np.float32)
        pts = pts * noise + 0.5
        return np.clip(pts, 0, 1)
    else:
        raise ValueError(f'Bilinmeyen dataset turu: {kind}')


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    pts = make_dataset('uniform', 5000, rng)
    oct_idx = OctreeIndex(pts)
    muq_idx = MuqarnasIndex(pts)

    origin = np.array([0.5, 0.5, -0.1], dtype=np.float32)
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    h_o, s_o = oct_idx.query_ray(origin, direction)
    h_m, s_m = muq_idx.query_ray(origin, direction)

    print(f'Octree   hits={len(h_o)} visited={s_o["visited_nodes"]} tested={s_o["tested_points"]}')
    print(f'Muqarnas hits={len(h_m)} visited={s_m["visited_nodes"]} tested={s_m["tested_points"]}')
