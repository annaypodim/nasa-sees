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
