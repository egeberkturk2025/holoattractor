# Mukarnas (Fraktal Partitioning) Benchmark Sonuclari - P0 Prototip

Bu sayfa, "core'u bozmadan" Mukarnas/fraktal partitioning prototipinin
Octree baseline'a karsi olcum sonuclarini ozetler.

Plan dokumani: [wiki/12_riskli_deney_plani_muqarnas.md](12_riskli_deney_plani_muqarnas.md)

---

## 1) Calistirilan prototip

- Kod:
  - Partition + query: [experiments/muqarnas_partition.py](../experiments/muqarnas_partition.py)
  - Benchmark runner: [experiments/muqarnas_benchmark.py](../experiments/muqarnas_benchmark.py)
- Cikti JSON: [experiments/results/muqarnas_benchmark.json](../experiments/results/muqarnas_benchmark.json)

---

## 2) Benchmark konfigurasyonu

- Nokta sayisi: `N=50,000`
- Dataset tipleri: `uniform`, `clustered`, `surface`
- Ray tipleri: `random (1000)`, `grid_scan (1024)`
- Hit tanimi: point-to-ray mesafesi <= `hit_radius=0.03`

Kaynak: [experiments/results/muqarnas_benchmark.json](../experiments/results/muqarnas_benchmark.json)

---

## 3) Sonuc ozeti (en kritik bulgu)

### 3.1 MuqarnasTree, "visited_nodes" acisindan belirgin sekilde kotu

Tum kosullarda **visited_nodes_mean** ve **visited_nodes_p95** Muqarnas'ta daha yuksek.

| Dataset | Ray | Octree mean | Muqarnas mean | Oran | PASS? |
|---------|-----|------------|--------------|------|-------|
| uniform | random | 71.8 | 199.1 | 2.77x | FAIL |
| uniform | grid_scan | 121.0 | 321.0 | 2.65x | FAIL |
| clustered | random | 58.3 | 161.4 | 2.77x | FAIL |
| clustered | grid_scan | 99.7 | 267.2 | 2.68x | FAIL |
| surface | random | 83.6 | 221.9 | 2.65x | FAIL |
| surface | grid_scan | 138.4 | 358.1 | 2.59x | FAIL |

Ornekler:

- `uniform/random`: octree mean 71.8 vs muqarnas mean 199.1 -> muqarnas ~2.8x daha cok node geziyor.
- `uniform/grid_scan`: octree 121.0 vs muqarnas 321.0.

Bu, P0 hedefindeki "query_steps" metrik iyilesmesi acisindan olumsuz.

---

### 3.2 MuqarnasTree, bazi dagilımlarda "tested_points" degerini dusurüyor

Bu iyi bir sinyal: leaf'lerde daha az point test ediyor (daha iyi ayristirma).

| Dataset | Ray | Oct tested_mean | Muq tested_mean | Degisim |
|---------|-----|----------------|----------------|----------|
| clustered | random | 34.19 | 20.62 | **-40%** |
| surface | random | 47.31 | 24.34 | **-49%** |
| surface | grid_scan | 71.82 | 37.16 | **-48%** |

Ancak visited_nodes artisi bu avantaji cogu durumda bastiriyor.

---

### 3.3 Duvar saati sureleri (query_time_s)

| Dataset | Ray | Octree (s) | Muqarnas (s) | Kazanan |
|---------|-----|-----------|-------------|--------|
| uniform | random | 0.338 | 0.362 | Octree |
| uniform | grid_scan | 0.531 | 0.714 | Octree |
| clustered | random | 0.291 | 0.318 | Octree |
| surface | random | 0.203 | 0.170 | **Muqarnas** |
| surface | grid_scan | 0.672 | 0.581 | **Muqarnas** |

Not: Python prototipinde sureler interpreter/cache/stack traversal farklarina duyarli; ama visited_nodes trend'i net.

---

## 4) Kabul kriterine gore karar

Planin kabul kriteri ([wiki/12_riskli_deney_plani_muqarnas.md, Bolum 5](12_riskli_deney_plani_muqarnas.md)):

- En az bir dataset'te `query_steps_mean` (burada proxy: visited_nodes_mean) icin >=15% iyilesme
- overhead <=2x

**P0 sonucu:** Bu prototip ile kriter saglanmadi. MuqarnasTree visited_nodes acisindan daha kotu.

---

## 5) Kok neden analizi

Duzgun (uniform) 4x4x4 bolmede bir ray cok sayida ortüsen AABB ile kesisiyor.
Bir eksen boyunca gecen ray 4x4=16 child icin slab-intersection yapmak zorunda;
Octree'de bu 2x2=4.

Traversal maliyet artisi ~ (4/2)^2 = 4x; geri kazanilan tested_points avantaji bunu telafi edemiyor.

> **Bu prototip "fraktal partitioning fikri tamamen kapandi" demek degil;
> mevcut implementasyon duzgun 4x4x4 bolme yaptigi icin traversal maliyeti yukseliyor.**

---

## 6) "Bulusu en iyi performansla" devam ettirmek icin onerilen revizyonlar (P0.1)

### Revizyon 1: Adaptive branching (sadece yogun node'larda 64-way)

- Dusuk yogunlukta 8-way (octree), yuksek yogunlukta 64-way.
- Hedef: visited_nodes'u octree seviyesine cekmek.

### Revizyon 2: Child ordering / early pruning

- Ray ile kesismesi en olasilikli child'lari once gez (front-to-back).
- Hedef: query sirasinda stack genislemesini azaltmak.

Bu iki degisiklik olmadan "daha fazla branching = daha hizli" genellemesi pratikte dogrulanmadi.

---

## 7) Godot'a baglama notu

Godot entegrasyonu mimaride var; ama bu P0 sonucu gosteriyor ki Godot'a tasinmadan
once partition'un "query_steps" metriklerinde net kazanim uretmesi sart.

Godot entegrasyon baglami:
- `raw/sessions/session_2026-04-28_full_holovoxe_development.md` (satir 239)
- `docs/ARCHITECTURE.md` (satir 26)

---

## 8) Sonraki adimlar

- [ ] P0.1: Adaptive branching implement et
- [ ] P0.1: Front-to-back child ordering implement et
- [ ] P0.1 benchmark'i kostur ve kabul kriteri kontrol et
- [ ] P0.1 basarili ise Godot entegrasyon sprint'i planla

---

*Son guncelleme: 2026-05-08 | Ege Berk Turk*
