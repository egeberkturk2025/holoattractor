# HoloAttractor

> Holographic Content-Addressable Memory — byte-level similarity without hashing

**Author:** Ege Berk Türk
**Affiliation:** Kadir Has University, Mechatronics Engineering, İstanbul
**License:** MIT  

---

## What is this?

HoloAttractor is a research system that answers one question:

> *"Have I seen this content before?"* — without storing hashes, without exact matching.

It uses FFT phase encoding + TDNN temporal memory to recognize content by its holographic signature.

---

## Key Results

| Experiment | Result |
|-----------|--------|
| Phase-LSH similarity (Mona Lisa cross-resolution) | **0.91** |
| TDNN temporal coherence gap | **0.78** (pos=0.81, neg=0.03) |
| Training loss (epoch 1→20) | 0.10 → 0.002 |
| Model parameters | 411K |
| Format bias discovery | JPEG DCT floor ~0.77 |

---

## Architecture

```
File bytes
  └─► FFT phase encoding     (512-dim phase vector)
        └─► Cosine LSH index  (semantic similarity)
              └─► TDNN memory (4×512 → 128 embedding)
                    └─► "Seen before?" answer
```

---

## Research Findings

### Finding 1: Temporal Coherence (Proven)
Phase-based TDNN learns that adjacent byte sequences belong together.
- Separation gap = **0.78** >> 0.5 threshold
- Model distinguishes sequential vs random content

### Finding 2: Format Bias (Discovered)
Model trained on JPEG byte-phase cannot distinguish Mona Lisa from Cat photo.
- Both are JPEG → DCT compressed data has similar phase statistics
- Pixel-domain phase and byte-phase occupy **disjoint embedding regions**
- Solution requires multi-domain co-training

---

## Project Structure

```
src/holodb/core/
  phase_projection.py       # FFT phase encoding
  semantic_lsh.py           # Cosine similarity index  
  tdnn_sequence_memory.py   # TDNN temporal memory (411K params)
  sphere_store.py           # SphereStore CRUD

experiments/
  train_tdnn.py             # Training script
  validate_phase7.py        # Validation (gap=0.78)
  phase_sequences.json      # 500 training sequences
```

---

## Citation

```bibtex
@misc{turk2026holoattractor,
  title   = {HoloAttractor: Holographic Content-Addressable Memory},
  author  = {Ege Berk T{\"{u}}rk},
  year    = {2026},
  url     = {https://github.com/egeberkturk2025/holoattractor}
}
```

---

*Copyright (c) 2026 Ege Berk Türk — MIT License*


---

## About the Author

**Ege Berk Türk** is a Mechatronics Engineering student at Kadir Has University, İstanbul. His research interests lie at the intersection of:

- **Holographic & Associative Memory** — content-addressable storage without hashing
- **Large Language Models (LLMs)** — memory architectures, retrieval-augmented generation
- **Temporal Neural Networks** — TDNN, sequence memory, phase-based encoding
- **Embedded AI** — running intelligent systems on resource-constrained hardware
- **Signal Processing** — FFT-based feature extraction, phase analysis

HoloAttractor is his first published research project, exploring whether byte-level holographic signatures can replace traditional hash-based deduplication in content-addressable memory systems.

> *"What if a system could recognize content it has seen before — not by its hash, but by the shape of its information?"*

**Contact:** egeberkturk2025@gmail.com  
**GitHub:** [egeberkturk2025](https://github.com/egeberkturk2025)  
**arXiv:** [egeberkturk2025](https://arxiv.org/a/egeberkturk2025)
