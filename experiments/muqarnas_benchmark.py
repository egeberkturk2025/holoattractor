"""muqarnas_benchmark.py
Octree vs MuqarnasTree P0 benchmark runner.
Cikarti: experiments/results/muqarnas_benchmark.json

Yazar: Ege Berk Turk  |  Lisans: MIT
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from muqarnas_partition import OctreeIndex, MuqarnasIndex, make_dataset

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------
CONFIG = {
    'n_points': 50_000,
    'hit_radius': 0.03,
    'random_seed': 42,
    'dataset_types': ['uniform', 'clustered', 'surface'],
    'ray_configs': {
        'random': 1000,
        'grid_scan': 1024,
    },
}

OUT_DIR = Path(__file__).parent / 'results'
OUT_FILE = OUT_DIR / 'muqarnas_benchmark.json'


# ---------------------------------------------------------------------------
# Ray üreticiler
# ---------------------------------------------------------------------------

def make_random_rays(n: int, rng: np.random.Generator):
    origins = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    dirs = rng.standard_normal((n, 3)).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + 1e-12)
    return origins, dirs


def make_grid_scan_rays(n: int):
    """Grid tarama: XY düzleminde ızgara orijinleri, Z+ yönü."""
    side = int(n ** 0.5)
    xs = np.linspace(0.05, 0.95, side)
    ys = np.linspace(0.05, 0.95, side)
    xx, yy = np.meshgrid(xs, ys)
    origins = np.stack([
        xx.ravel(), yy.ravel(),
        np.full(side * side, -0.1)
    ], axis=1).astype(np.float32)
    dirs = np.tile([0, 0, 1.0], (side * side, 1)).astype(np.float32)
    return origins[:n], dirs[:n]


# ---------------------------------------------------------------------------
# Tek koşu istatistikleri
# ---------------------------------------------------------------------------

def run_queries(index, origins, dirs) -> Dict[str, Any]:
    visited_list, tested_list, time_list = [], [], []
    for origin, direction in zip(origins, dirs):
        t0 = time.perf_counter()
        _, stats = index.query_ray(origin, direction)
        t1 = time.perf_counter()
        visited_list.append(stats['visited_nodes'])
        tested_list.append(stats['tested_points'])
        time_list.append(t1 - t0)

    vn = np.array(visited_list)
    tp = np.array(tested_list)
    ts = np.array(time_list)
    return {
        'visited_nodes_mean': float(np.mean(vn)),
        'visited_nodes_p95': float(np.percentile(vn, 95)),
        'tested_points_mean': float(np.mean(tp)),
        'tested_points_p95': float(np.percentile(tp, 95)),
        'query_time_s': float(np.sum(ts)),
        'query_time_mean_ms': float(np.mean(ts) * 1000),
        'n_queries': len(origins),
    }


# ---------------------------------------------------------------------------
# Ana benchmark döngüsü
# ---------------------------------------------------------------------------

def benchmark() -> Dict[str, Any]:
    rng = np.random.default_rng(CONFIG['random_seed'])
    results: Dict[str, Any] = {'config': CONFIG, 'results': {}}

    for ds_type in CONFIG['dataset_types']:
        print(f'\n--- Dataset: {ds_type} ---')
        pts = make_dataset(ds_type, CONFIG['n_points'], rng)

        print('  Building OctreeIndex...')
        t0 = time.perf_counter()
        oct_idx = OctreeIndex(pts, hit_radius=CONFIG['hit_radius'])
        oct_build = time.perf_counter() - t0

        print('  Building MuqarnasIndex...')
        t0 = time.perf_counter()
        muq_idx = MuqarnasIndex(pts, hit_radius=CONFIG['hit_radius'])
        muq_build = time.perf_counter() - t0

        results['results'][ds_type] = {}

        for ray_type, n_rays in CONFIG['ray_configs'].items():
            print(f'  Ray type: {ray_type} ({n_rays} rays)')
            ray_rng = np.random.default_rng(CONFIG['random_seed'] + hash(ray_type) % 1000)
            if ray_type == 'random':
                origins, dirs = make_random_rays(n_rays, ray_rng)
            else:  # grid_scan
                origins, dirs = make_grid_scan_rays(n_rays)

            oct_stats = run_queries(oct_idx, origins, dirs)
            muq_stats = run_queries(muq_idx, origins, dirs)

            oct_stats['build_time_s'] = oct_build
            muq_stats['build_time_s'] = muq_build

            visited_ratio = (muq_stats['visited_nodes_mean'] /
                             max(oct_stats['visited_nodes_mean'], 1e-6))

            results['results'][ds_type][ray_type] = {
                'octree': oct_stats,
                'muqarnas': muq_stats,
                'visited_nodes_ratio_muq_over_oct': round(visited_ratio, 3),
                'pass': visited_ratio <= 0.85,  # >=15% iyilesme kriteri
            }

            print(f'    Octree  : visited_mean={oct_stats["visited_nodes_mean"]:.1f}')
            print(f'    Muqarnas: visited_mean={muq_stats["visited_nodes_mean"]:.1f}')
            print(f'    Ratio   : {visited_ratio:.3f} ({"PASS" if visited_ratio<=0.85 else "FAIL"})')

    return results


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = benchmark()
    with open(OUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nSonuclar kaydedildi: {OUT_FILE}')
