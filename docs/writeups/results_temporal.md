# Results — Temporal Gap-Filling

## Setup

We evaluate **long-gap temporal imputation**: contiguous windows of length *L*
∈ {24 h, 48 h} are removed from otherwise-observed sensors, and each sensor's own
missing hours are reconstructed. This is the axis GraPhy's memoryless spatial
formulation leaves open — GraPhy predicts a held-out sensor from *other* sensors
at the same hour and never uses a target's own history.

**Model (OURS).** A physics-GNN (GraPhyNet: diffusion + convection + local
modules) predicts a *correction* on top of a learned spatio-temporal prior

  prior = b · persistence + (1 − b) · RK-elev,  b = σ(β), learned,

where RK-elev is regression-kriging with an elevation-drift mean (OLS trend on
station elevation + IDW of the residual) and persistence is carry-forward of the
last known value. All lag/persistence/IDW inputs are built from the leak-safe
`known` mask (observed ∧ ¬test-gap ∧ ¬train-gap).

**Baselines.** *ST-kriging* is the prior-only prediction — the same learned blend
of RK-elev and persistence with **no GNN correction**. Because OURS and ST-kriging
share this prior, the reported gap isolates the GNN correction's contribution.
*Persistence* is carry-forward alone.

**Protocol.** Proper **5-fold cross-validation**: each sensor's timeline is tiled
into non-overlapping length-*L* windows and the *j*-th window is assigned to fold
*j* mod 5, so every clean window is held out exactly once (a true partition, no
random test overlap). We report mean ± std over folds; all folds use HRRR wind.

## Main result

| City | Gap | OURS MAE | ST-kriging | Persistence | Δ (OURS vs ST-krig) | Folds won |
|------|-----|---------:|-----------:|------------:|:-------------------:|:---------:|
| **Fresno** | 24 h | **4.16 ± 0.10** | 4.49 | 7.63 | **−7.2%** | 5 / 5 |
| **Fresno** | 48 h | **4.75 ± 0.41** | 5.06 | 9.13 | **−6.1%** | 5 / 5 |
| SLC        | 24 h | **5.94 ± 1.24** | 6.14 | 7.71 | −3.3% | 5 / 5 |
| SLC        | 48 h | **7.41 ± 1.37** | 7.54 | 9.34 | −1.8% | 5 / 5 |
| **Pittsburgh** | 24 h | **3.89 ± 0.09** | 4.32 | 5.43 | **−9.9%** | 5 / 5 |
| **Pittsburgh** | 48 h | **4.28 ± 0.25** | 4.66 | 6.52 | **−8.2%** | 5 / 5 |

MAE in µg/m³. **OURS beats the strong space+time baseline in all 30/30 folds.**

## Findings

**1. The GNN's advantage requires temporal signal.** In the pure spatial
(memoryless) task, regression-kriging with an elevation-drift mean *beats* our GNN
(≈ −10% vs IDW) — consistent with the IGNNK authors' own observation that kriging
wins on static/smooth/dense fields. The GNN earns its keep only once given an
information edge over static interpolation: the target's own history. On the
temporal-gap task it reverses the result, beating even a kriging baseline that
already uses terrain *and* persistence.

**2. The margin scales with spatiotemporal richness.** Pittsburgh (dense) −8 to
−10% > Fresno −6 to −7% > SLC (sparse, complex terrain) −2 to −3%. The denser and
more temporally-informative the network, the more the GNN correction adds — again
the IGNNK regime.

**3. We fill the gap GraPhy leaves open, on GraPhy's own city.** GraPhy's reported
2.38 MAE is on the memoryless spatial task; our −6 to −7% Fresno win is on the
orthogonal temporal-gap axis. These are complementary, not competing, results — a
temporal memory is exactly the component GraPhy's design omits.

**4. Wind is not the lever.** Enabling HRRR advection barely moves the margin
(SLC −7.5% with wind vs −7.1% wind-zero under Monte-Carlo). The win comes from the
terrain-mean prior + temporal memory + GNN nonlinearity, not from modeled
transport — a useful negative result for a surface-sensor regime.

**5. Honest limitation — SLC.** Under exhaustive K-fold, SLC's absolute MAE is
higher and its margin thinner than earlier Monte-Carlo estimates suggested
(−2/−3% vs −5/−7%), because full coverage forces evaluation on every winter-
inversion window the random-gap sampling under-represented. The GNN still wins
every fold, but the sparse, high-variance terrain city is where its temporal edge
is smallest — the honest weak spot and the natural target for future work.
