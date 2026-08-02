# Results

**Analyzing Data.** We report mean absolute error (MAE, µg/m³) on held-out data
under a strict inductive protocol, with dispersion given as ±1 SD across random
seeds (spatial task) or across folds (temporal task). Throughout, we separate two
evaluation regimes that use the *same* model and inputs but hold out data
differently. In the **spatial** task, whole sensors are removed and predicted from
the remaining sensors at the same hour — the memoryless inductive-kriging protocol
of Wu et al. (IGNNK), which can only exploit a covariate's *spatial* variation. In
the **temporal** task, contiguous multi-hour windows ("gaps") are removed from
otherwise-present sensors and each sensor predicts *its own* missing hours, so the
target's history and any covariate that varies in *time* can act. Every comparison
uses identical train/test splits for the learned model and the baseline, and every
PurpleAir series is Barkjohn-corrected before evaluation.

## 1. On flat terrain, a learned correction does not beat IDW — at any density

On Fresno's flat Central Valley network, the base diffusion/convection/local graph
model ties or loses to plain IDW in every configuration we tested. The result was
insensitive to capacity, training length, regularization, and data volume: across a
density sweep from N = 6 to N = 22 sensors (three random sub-networks and multiple
seeds at each density), the learned correction never robustly fell below the IDW
baseline computed on the identical split **(Figure 2)**. Because a sparse,
undersampled network is the most plausible place for a learned model to add value,
this sweep was designed specifically to rule out data scarcity as the explanation —
and no such crossover appeared at any density, on either an IDW-prior or a
kriging-prior variant. On the densest clean Fresno network the IDW floor reaches
2.83 µg/m³, already below literature-reported IDW baselines on comparable networks,
leaving essentially no smooth-field structure for a learned residual to recover
without overfitting the small (16–30 sensor) training set **(Figure 1)**.

This is a controlled *negative* result, and it is the load-bearing first step of the
argument: a generic learned spatial correction is not automatically better than the
cheap baseline, and the reason is a property of the data, not a weak baseline or a
data-starved model.

## 2. Terrain is the one spatial condition that flips the outcome

The picture changes on Salt Lake City's mountain-basin terrain, where elevation
varies meaningfully in space. We first isolated the terrain signal with a purely
statistical control: a regression-kriging prior with an elevation-drift mean
function (RK-elev) — an OLS trend on sensor elevation plus IDW of the residual,
which reduces exactly to IDW when the elevation coefficient is zero. RK-elev cut
MAE from the IDW baseline's 4.06 to 3.65 µg/m³ (−10%), winning every seed, showing
that the terrain signal lives in the interpolation *mean function*, not in model
capacity. Building on this, our terrain-aware hybrid adds an elevation-informed
decay kernel (down-weighting sensor pairs separated by a large elevation gap) and a
matching gate on the diffusion and convection modules. On SLC this drives MAE from
4.80 down to 4.19 µg/m³ — a 12.7% reduction that flips the learned model from
*losing* to IDW to *beating* it **(Figure 3)**. The win is robust at the seed level,
not merely on average: the terrain-aware model wins on nearly every individual run.
Critically, the same mechanism produces no change on flat Fresno — the gate
self-disables where Δelevation is near zero — so the added complexity is spent only
where the physics justifies it.

## 3. The covariate nulls confirm the mechanism is spatial variance, not covariate count

The terrain result was not found by throwing every available covariate at the
model; it is the one covariate that carries a real, measured spatial signal. We
tested temperature, satellite AOD, and reanalysis wind as candidate covariates, in
both node-feature and edge-gate form, and each was null within noise (ΔMAE of
+0.02–0.05) **(Figure 4)**. The explanation is common-mode structure at metro scale:
temperature's spatial standard deviation is roughly 1% of its temporal standard
deviation; HRRR wind direction varies by only ~15–18° across all sensors at any
given hour (tested on Pittsburgh's 95-sensor network with real reanalysis wind), so
every sensor effectively sees the same wind and the direction-routing convection
module has nothing to exploit; and a SPIN-style masked AOD spatial-gradient training
constraint actively *hurt* on SLC winter, because a wintertime inversion decouples
surface PM2.5 from the MODIS column that the gradient supervises. A covariate helps a
spatial model only if it varies in space — elevation does, and at the city scale the
others do not.

The natural objection is that these covariates were only tested where they *cannot*
vary — the memoryless spatial task. We therefore re-ran the same three covariates on
the long-gap temporal task (Section 4), where each does vary in time and a stale
persistence estimate should, in principle, leave room for them. They remain null.
Added to the winning temporal model under identical 5-fold folds, a per-node
temperature gate moved MAE by +0.02% (24 h) and +0.47% (48 h); an AOD node-feature by
+1.0% and +0.65% — both the *wrong* direction, improving at most 2 of 5 folds; and
zeroing HRRR wind changed the margin by ≤1 pp on dense Pittsburgh and was a wash or
slightly favorable elsewhere (temperature and AOD data exist only for Pittsburgh, so
those two rest on one city × two gaps; wind spans all three). The mechanism is now
sharper: the target sensor's own persistence and lag-24/lag-168 history already
encode whatever temperature, AOD, or wind would contribute, so the covariates are
redundant in *both* regimes — common-mode in space, history-subsumed in time. The
project's edge is the terrain-mean prior plus temporal memory plus the GNN's
nonlinear correction, never covariate fusion.

## 4. The payoff: on the long-gap temporal task, the learned correction beats even a strong spatiotemporal-kriging baseline

The memoryless spatial task is the regime where a learned correction is least likely
to help, because a smooth field is already near-optimally described by distance. The
regime where a learned model *should* earn its complexity is one with structure that
static interpolation cannot see — and long-gap temporal imputation is exactly that
case. Here the architecture is unchanged; time enters only as leak-free lag features
and as a learnable prior blend:

  prior = b · persistence + (1 − b) · RK-elev,   prediction = prior + GNN correction,

where persistence carries each sensor's last known value forward through the gap,
RK-elev supplies the terrain-aware spatial estimate, and b = σ(β) is learned. The
prior alone — a space-*and*-time interpolator using both terrain mean and temporal
memory — is a deliberately strong baseline we call **ST-kriging**; it is a far
harder floor to beat than the IDW of the spatial task. Our full model adds the GNN's
nonlinear correction on top.

Under proper 5-fold cross-validation (every clean gap window held out exactly once,
folds mutually disjoint), with real HRRR wind, across three cities and two gap
lengths, the GNN correction beats the ST-kriging baseline in **all 30 of 30 folds**
**(Figure 5)**:

| City | Gap | OURS (MAE) | ST-kriging | Persistence | OURS vs ST-kriging | Folds won |
|------|-----|-----------|-----------|-------------|--------------------|-----------|
| Pittsburgh | 24 h | 3.89 ± 0.09 | 4.32 | 5.62 | **−9.9%** | 5/5 |
| Pittsburgh | 48 h | 4.28 ± 0.25 | 4.66 | 6.61 | **−8.2%** | 5/5 |
| Fresno | 24 h | 4.16 ± 0.10 | 4.49 | — | **−7.2%** | 5/5 |
| Fresno | 48 h | 4.75 ± 0.41 | 5.06 | — | **−6.1%** | 5/5 |
| SLC | 24 h | 5.94 ± 1.24 | 6.14 | — | **−3.3%** | 5/5 |
| SLC | 48 h | 7.41 ± 1.37 | 7.54 | — | **−1.8%** | 5/5 |

The sign holds everywhere; the *magnitude* scales with how much spatiotemporal
signal the network provides — largest on dense Pittsburgh (−8 to −10%), smallest on
sparse SLC terrain (−2 to −3%) — precisely the IGNNK regime, in which a learned graph
model earns more the richer the field it is fed. The learned blend confirms the
mechanism: b ≈ 0.49 on dense, dynamic Pittsburgh (more temporal signal for the GNN
to correct) versus ≈ 0.31 on sparse SLC (leaning on the terrain-mean prior). Wind is
not the lever — the SLC margin is −7.5% with full HRRR wind versus −7.1% with wind
zeroed — so the win comes from the terrain-mean prior, temporal memory, and the GNN's
nonlinear correction, not from advection.

## 5. A deploy-or-not diagnostic: predicting where the correction pays, before training

If the GNN's advantage is dictated by the structure of the data, that structure
should be measurable *a priori* — from an unlabeled network, before any model is
trained — and used to decide whether the correction is worth its complexity at all.
We test this directly. Subsampling each city to 18 distinct sub-networks and
regressing the realized 5-fold margin on label-free network statistics, the strongest
single predictor is sensor **density** (nodes/km²; R² = 0.67, Spearman 0.80),
cleanly above a leave-one-out interpolation-headroom statistic (R² = 0.39) — the
denser the graph, the more the GNN earns, exactly the IGNNK regime made quantitative.

Sweeping gap length {6, 12, 24, 48, 72 h} across all three cities adds the temporal
axis and reveals a non-monotone structure: the margin over ST-kriging is an
**inverted-U in gap length**, peaking at an intermediate horizon (≈24–48 h) and
falling off at both ends — at short gaps persistence is near-perfect and leaves the
GNN nothing to add, at long gaps every method's temporal information has decayed and
the spatial prior dominates for all. A two-axis model, margin ~ headroom +
staleness², captures this at R² = 0.75 (staleness alone is uninformative, R² = 0.05,
precisely because the relationship is a hump rather than a slope).

Most importantly, this sweep produced the first **observed crossover**: on sparse SLC
at a 6 h gap, the GNN *loses* — −2.97% MAE versus ST-kriging, 0 of 5 folds won. It is
the one regime in which the learned correction is demonstrably not worth training, and
it falls exactly where the rule predicts (lowest spatial headroom × lowest temporal
staleness), converting the deploy threshold from an extrapolation into an empirically
bracketed boundary **(Figure 6)**. Framed plainly, this is an instance of the
classical algorithm-selection problem specialized to graph-based PM2.5 imputation:
we do not merely report that a learned model wins, but map the conditions — sensor
density and gap horizon — under which it stops being worth it, validated against a
regime where it actually fails.

**Synthesis.** The two regimes tell one story. A learned correction cannot beat cheap
interpolation on the memoryless spatial task regardless of density; terrain is the
only spatial condition under which it recovers a marginal, self-disabling advantage;
and on the long-gap temporal task — where the field carries structure that static
space+time kriging cannot capture — the same learned correction robustly beats even a
strong spatiotemporal baseline across three cities, two gap lengths, and every one of
30 cross-validation folds. Model complexity is justified by the physical and temporal
structure of the data, and we make the condition operational: an a-priori,
label-free rule on sensor density and gap horizon predicts the correction's value
(R² = 0.75) and is anchored by an observed regime — sparse network, short gap — in
which the learned model measurably loses and should not be deployed.

---

### Figure callouts

- **[Figure 1]** Grouped bar chart, MAE (µg/m³) for {IDW, base graph model, terrain-aware hybrid}, one panel per city (Fresno, SLC), error bars = ±1 SD across seeds. *(existing plan)*
- **[Figure 2]** Density sweep line plot, N = 6→22 vs MAE, three lines = IDW / GNN-kriging-prior / GNN-IDW-prior, flat Fresno. Controlled negative result — no crossover. *(existing plan; `experiments/logs/density_sweep/results.csv`)*
- **[Figure 3]** SLC bar chart, {gate off, gate on} × {base, IDW-anchored hybrid}, IDW reference line — 4.80→4.19, the spatial headline. *(existing plan)*
- **[Figure 4]** Covariate-null summary (temperature, AOD, wind), ΔMAE within ±0.05 of baseline. *(existing plan, optional)*
- **[Figure 5 — NEW]** Temporal payoff: grouped bars of OURS vs ST-kriging (and persistence where available), grouped by city × gap length, error bars = ±1 SD across 5 CV folds; annotate "wins 30/30 folds." *(build from the §I K-fold table above.)*
- **[Figure 6 — NEW]** Deploy-or-not diagnostic: margin (OURS vs ST-kriging, %) vs gap length, one line per city (Pittsburgh/Fresno/SLC), horizontal zero line; circle the SLC-6 h point at −3.0% (0/5 folds) as the observed crossover. Inset or twin panel: margin vs sensor density across the 18 sub-networks (R² = 0.67). *(build from `experiments/logs/diagnostic/gap_points.csv` and `density_points.csv`.)*
