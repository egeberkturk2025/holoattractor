# Riskli Deney Planı #12 — Mükarnas (Fraktal Partitioning)

> **Durum:** P0 tamamlandı — kriter sağlanmadı. P0.1 revizyonları önerildi.
> **Tarih:** 2026-05-08
> **Yazar:** Ege Berk Türk

---

## 1. Motivasyon

HoloAttractor'un nokta-bulut sorgulama katmanı (HoloDB / SphereStore) halihazırda
standart octree (8-way branşing) kullanıyor. Mimari belgesinde
([docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)) belirtildiği üzere sorgu adım
maliyetini düşürmek kritik yol üzerinde.

Fraktal/Mükarnas partitioning hipotezi:
> 4×4×4 = 64-way branşing, özellikle kümeli ve yüzey-dağılımlı veri
> kümelerinde octree'ye göre daha az node gezerek daha önce ray-hit
> kararı verebilir.

---

## 2. Risk Değlendirmesi

| Risk | Olasılık | Etki | Açıklama |
|------|----------|------|----------|
| visited_nodes artışı | Yüksek | Yüksek | 64 child/node ray-testi eşleşme sayısını arttirabiliyor |
| Bellek artışı | Orta | Orta | 64 child vs 8 child ≈ 8× daha fazla node nesnesi |
| Traversal stack derinliği | Düşük | Düşük | MAX_DEPTH=5 ile sınırlı |
| Core kodunu bozma | Düşük | Çok Yüksek | Tamamen izole prototip; core'a dokunulmuyor |

---

## 3. Kapsam (Core'a dokunulmaz)

Bu deney tamamen `experiments/` klasörü içinde izole:

- `experiments/muqarnas_partition.py` — OctreeIndex + MuqarnasIndex
- `experiments/muqarnas_benchmark.py` — benchmark runner
- `experiments/results/muqarnas_benchmark.json` — çıktı

**Core (`src/`) hiçbir zaman değiştirilmez.**

---

## 4. Benchmark Konfigurasyonu

```python
CONFIG = {
    'n_points': 50_000,
    'hit_radius': 0.03,
    'dataset_types': ['uniform', 'clustered', 'surface'],
    'ray_configs': {'random': 1000, 'grid_scan': 1024},
}
```

- **Hit tanımı:** nokta-to-ray mesafesi ≤ `hit_radius = 0.03`
- **Metrik proxy:** `visited_nodes_mean` (query_steps'e karşılık)

---

## 5. Kabul Kriteri

P0 kabul edilmiş sayılır eğer:

1. **En az bir dataset'te** `visited_nodes_mean` için **≥15% iyileşme**
   (oran ≤ 0.85 Muqarnas/Octree)
2. Overhead ≤ 2× (`visited_nodes_ratio_muq_over_oct ≤ 2.0`)

Ikisi birden sağlanmalıdır.

---

## 6. P0 Sonuçları (Kısa Özet)

Detaylı analiz için:
[wiki/muqarnas_p0_benchmark_sonuclari.md](muqarnas_p0_benchmark_sonuclari.md)

| Dataset | Ray tipi | Oct visited_mean | Muq visited_mean | Oran | PASS? |
|---------|----------|-----------------|-----------------|------|-------|
| uniform | random | 71.8 | 199.1 | 2.77 | FAIL |
| uniform | grid_scan | 121.0 | 321.0 | 2.65 | FAIL |
| clustered | random | 58.3 | 161.4 | 2.77 | FAIL |
| clustered | grid_scan | 99.7 | 267.2 | 2.68 | FAIL |
| surface | random | 83.6 | 221.9 | 2.65 | FAIL |
| surface | grid_scan | 138.4 | 358.1 | 2.59 | FAIL |

**P0 kararı:** Kriter sağlanmadı. Fikir kapatılmıyor; P0.1 revizyonu öneriliyor.

---

## 7. Teşhis: Neden Başarısız?

Düzgün (uniform) 4×4×4 bölmede bir ray çok sayıda örtüşen AABB ile
kesişiyor: bir eksen boyunca geçen ray 4×4 = 16 child için slab-intersection
yapmak zorunda. Octree'de bu sayı 2×2 = 4.

Traversal maliyet artışı ~ (4/2)^2 = 4×; geri kazanılan tested_points avantajı
bunu telafi edemiyor.

---

## 8. P0.1 Revizyon Önerileri

### 8.1 Adaptif Branşing

```
if node_point_count > HIGH_DENSITY_THRESHOLD:
    split into 64 children  # Muqarnas mod
    else:
    split into 8 children   # Octree mod
```

Hedef: düşük yoğunluklu node'larda octree maliyet korunur,
yalnizca yoğun külerde 64-way devreye girer.

### 8.2 Front-to-Back Child Sıralama

Child AABB'lerini ray ile kesim mesafesine göre sırala;
ıskalanacak child'lara erken prune uygula.

Beklenen etki: visited_nodes %20–40 düşüş (octree'de de geçerli).

---

## 9. Godot Entegrasyon Notu

Godot'a taşıma (`raw/sessions/session_2026-04-28_full_holovoxe_development.md`)
nükteye alındı:

> Godot entegrasyonu `visited_nodes_mean` metriklerinde net kazanım
> üretene kadar başlatilmaz.

P0.1 başarılı olursa Godot entegrasyon sprint'i planlanacak.

---

## 10. İlgili Dosyalar

- Prototip: [experiments/muqarnas_partition.py](../experiments/muqarnas_partition.py)
- Benchmark runner: [experiments/muqarnas_benchmark.py](../experiments/muqarnas_benchmark.py)
- Sonuç JSON: [experiments/results/muqarnas_benchmark.json](../experiments/results/muqarnas_benchmark.json)
- Sonuç wiki: [wiki/muqarnas_p0_benchmark_sonuclari.md](muqarnas_p0_benchmark_sonuclari.md)
- Mimari: [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
