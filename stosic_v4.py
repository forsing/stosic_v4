from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v4.py — 7-node krug (K=7 / prilagodjenje 7/39) — Fisher-lokalna frekvencija u ćeliji (7/39)

Nadogradnja: v1–v3.

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/wasserstein_metric.pvs
  — lokalna struktura oko mere (geodezijski / blizina u prostoru raspodela)
  luces-pvs-theories-main/transport_stability.pvs / fisher_voronoi.pvs
  — stabilnost mape u ćeliji; d_F kao metrika blizine

Mapiranje na 7/39:
  ista Fisher–Voronoi podela kao v1 (ceo CSV, SEED=39, bez randoma)
  u ćeliji r_curr: svako istorijsko izvlačenje i dobija težinu
      w_i = 1 / (EPS + d_F(p_i, p_last))
  skor broja = suma w_i preko izvlačenja u ćeliji koja sadrže taj broj
  next = top 7 po skoru, jedna kombinacija
"""

from typing import List

import numpy as np

from stosic_v1 import (
    EPS,
    K_REGIMES,
    MAX_NUM,
    draw_to_prob,
    fisher_rao,
    kmeans_fisher,
    load_draws,
)
from stosic_v2 import top7_from_freq


def local_fisher_frequency(
    draws: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    regime: int,
    p_last: np.ndarray,
) -> np.ndarray:
    """Težinska frekvencija u ćeliji: bliži Fisher → veći doprinos."""
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for i in range(len(draws)):
        if int(labels[i]) != regime:
            continue
        w = 1.0 / (EPS + fisher_rao(probs[i], p_last))
        for n in draws[i]:
            skor[int(n) - 1] += w
    return skor


def predict_next(draws: np.ndarray) -> List[int]:
    """Jedna next — Fisher-lokalna frekvencija u poslednjoj ćeliji."""
    probs = np.vstack([draw_to_prob(d) for d in draws])
    _, labels = kmeans_fisher(probs, k=K_REGIMES)

    r_curr = int(labels[-1])
    p_last = probs[-1]
    skor = local_fisher_frequency(draws, probs, labels, r_curr, p_last)

    # ako ćelija prazna (ne bi smelo) — fallback ravna frekvencija svih
    if float(skor.sum()) <= 0:
        for d in draws:
            for n in d:
                skor[int(n) - 1] += 1.0

    return top7_from_freq(skor)


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[7, x, 14, y, 17, z, 32]
"""



"""
v4: lokalna Fisher težina oko poslednjeg stanja (transport / W₂ duh); 

v4: u ćeliji poslednjeg režima — frekvencija težena sa 1/(ε + d_F) ka poslednjem stanju.
"""




"""
Stosić jezgro	Ocena

v1
fisher_voronoi (d_F, centroid, nearest_regime)
Da — najbliže strogo

v2
regime_sequence / prelaz iz istog PVS
Operaciona frekvencija prelaza je moja mapa, ali ideja je njegova

v3
dual_observability (obe karte)
Delimično — A+B = remix v1+v2 signala, ne čitanje PVS formula

v4
blizina u prostoru mera / stabilnost
Slabo — 1/(ε+d_F) težine su moja konstrukcija, ne direktan teorem

v5
entropy_along_geodesic (H na mid)
Uglavnom da — filter po H(mid)≤max je blizu teoremima

v6
velocity_asymmetry / lie shape direction
Uglavnom da — √p razlike; suma max(u,0) je naša agregacija
Zaključak: strogo čist je uglavnom v1 (i solidno v5/v6/v2). v3 i naročito v4 su više moje operacione verzije „u duhu“ nego doslovan Stosić.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
