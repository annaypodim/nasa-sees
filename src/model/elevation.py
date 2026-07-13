"""
elevation gate  (a MODULATOR, not a 4th module)
=============================================================================
elevation doesn't MOVE PM2.5 the way wind does -- it changes how easily the
other two transport terms propagate across an edge. so it isn't a thing to
fuse; it's a per-edge scalar in [0, 1] that DAMPENS diffusion and convection
when two sensors are far apart in height:

    gate(Δ) = exp(−Δ / h_up)     if Δ > 0   (dst higher than src -> uphill)
            = exp( Δ / h_down)   if Δ < 0   (dst lower  than src -> downhill)
            = 1                  at Δ = 0

    Δ = elev[dst] − elev[src]   (SIGNED, metres)

WHY SIGNED (why two scales, not one exp(−|Δ|/h)):
    valley-pooling is asymmetric -- cold air and pollutants sink and pool
    downhill far more readily than they climb. a symmetric magnitude gate
    collapses that before the model can see it. h_up and h_down let uphill and
    downhill decay at their own learned rates, while keeping the clean exp form.

    the SCALE h is SHARED across diffusion and convection: elevation is one
    physical quantity, so one gate models it, and both transport terms read the
    same modulation instead of each learning a redundant copy.

HOW IT'S USED (one extra scalar, no separate softmax branch):
    diffusion  : edge_weight  ->  edge_weight * gate   (weaker coupling uphill)
    convection : message      ->  message     * gate   (wind carries less across height)

with placeholder (constant) elevation Δ ≡ 0, so gate ≡ 1 and this is a no-op
until a real DEM lands -- exactly the intended "ready but inert" state.

run the demo:   .venv/bin/python elevation.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElevationGate(nn.Module):
    """Per-edge multiplicative gate in (0, 1] from signed Δelevation.

    Two learnable height scales (uphill / downhill) kept strictly positive via
    softplus, with a small floor so the gate can't blow up near h -> 0.
    """

    def __init__(self, h_init: float = 200.0, h_floor: float = 1.0):
        super().__init__()
        # raw param -> softplus -> positive scale. softplus(y) ≈ y for y >> 0, so
        # for a typical h_init (hundreds of metres) raw ≈ h_init to <1e-8; using
        # the exact inverse log(expm1(h_init)) overflows float32 past ~20.
        raw = float(h_init) if h_init > 20 else float(torch.log(torch.expm1(torch.tensor(float(h_init)))))
        self.raw_h_up = nn.Parameter(torch.tensor(raw))
        self.raw_h_down = nn.Parameter(torch.tensor(raw))
        self.h_floor = h_floor

    def scales(self):
        """Current (positive) uphill / downhill height scales in metres."""
        h_up = F.softplus(self.raw_h_up) + self.h_floor
        h_down = F.softplus(self.raw_h_down) + self.h_floor
        return h_up, h_down

    def forward(self, delev):
        """delev: signed Δelevation per edge [E]  ->  gate [E] in (0, 1]."""
        h_up, h_down = self.scales()
        # uphill (Δ>0) decays with h_up, downhill (Δ<0) with h_down; Δ=0 -> 1
        h = torch.where(delev >= 0, h_up, h_down)
        return torch.exp(-delev.abs() / h)


# ===========================================================================
# LEARNED terrain gates (richer than the fixed-form ElevationGate above)
# ---------------------------------------------------------------------------
# The ElevationGate uses exp(-|Δ|/h) with just TWO learned scalars (h_up,h_down).
# These two modules keep the exp ENVELOPE (so g(0)=1 exactly -> no self-damping,
# and the gate stays in (0,1] -> stable) but make the decay RATE a learned
# function of the edge's terrain (and, for convection, its wind alignment):
#
#     g(Δ) = exp( -(|Δ|/s) * rate )        rate = softplus(MLP(features)) >= 0
#
# rate depends on Δ (and w_A), so the model learns a per-edge, sign-asymmetric,
# NON-exponential terrain response instead of two global height scales. At Δ=0
# the |Δ| factor forces g=1 regardless of the MLP -> the "no damping at equal
# height" prior is baked in and can't be trained away.
# ===========================================================================
class TerrainGate(nn.Module):
    """Diffusion gate: learned g_D(Δelev) in (0,1] with a Δ-dependent decay rate.

    The faithful-model analog of the terrain-aware IDW *kernel* (our biggest SLC
    lever). Replaces the two-scalar exp with a small MLP that reads the signed and
    absolute height gap, so uphill/downhill and near/far gaps get their own
    (learned, smooth) damping -- richer than h_up/h_down alone.
    """

    def __init__(self, scale: float = 200.0, hidden: int = 16):
        super().__init__()
        self.scale = scale
        # features: [Δ/s, |Δ|/s, sign(Δ)] -> a positive decay rate per edge
        self.rate_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, delev):
        """delev: signed Δelevation per edge [E] -> gate [E] in (0,1]."""
        d = delev / self.scale
        feats = torch.stack([d, d.abs(), torch.sign(delev)], dim=1)   # [E, 3]
        rate = F.softplus(self.rate_mlp(feats)).squeeze(1)            # [E] >= 0
        return torch.exp(-d.abs() * rate)                            # g(0)=1


class DrainageGate(nn.Module):
    """Convection gate: directional drainage g_C(Δelev, w_A) in (0,1].

    Cold-air/pollutant drainage runs DOWNHILL, preferentially when the wind blows
    that way. The decay rate reads the height gap AND the wind alignment w_A =
    cos(wind_dir - edge_bearing), plus their interaction w_A*sign(Δ) -- so downslope
    (Δ<0) transport that is also downwind can be damped at a different (lower) rate
    than upslope transport, modelling downslope drainage distinctly from upslope.
    With wind off (w_A=None) it degrades gracefully to a Δ-only drainage gate.
    """

    def __init__(self, scale: float = 200.0, hidden: int = 16):
        super().__init__()
        self.scale = scale
        # base features [Δ/s, |Δ|/s, sign(Δ)] (+2 wind features when w_A is given).
        # Two input heads so the same module works with or without wind, without
        # retraining a different shape: a zero wind-alignment reproduces the base.
        self.rate_mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, delev, wind_align=None):
        """delev [E] signed; wind_align [E] in [-1,1] or None -> gate [E] in (0,1]."""
        d = delev / self.scale
        sign = torch.sign(delev)
        if wind_align is None:
            wa = torch.zeros_like(d)
        else:
            wa = wind_align
        feats = torch.stack([d, d.abs(), sign, wa, wa * sign], dim=1)  # [E, 5]
        rate = F.softplus(self.rate_mlp(feats)).squeeze(1)            # [E] >= 0
        return torch.exp(-d.abs() * rate)                            # g(0)=1


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gate = ElevationGate(h_init=200.0)
    h_up, h_down = gate.scales()
    print(f"initial scales:  h_up={h_up.item():.1f} m   h_down={h_down.item():.1f} m")

    # a spread of signed height differences, downhill (negative) to uphill (positive)
    delev = torch.tensor([-400., -200., -50., 0., 50., 200., 400.])
    g = gate(delev)
    print("\n  Δelev(m)   gate")
    for d, gv in zip(delev.tolist(), g.tolist()):
        bar = "#" * int(gv * 30)
        print(f"  {d:+7.0f}   {gv:5.3f}  {bar}")
    print("\nΔ=0 -> gate=1 (no damping); |Δ| large -> gate->0 (edge nearly cut).")
