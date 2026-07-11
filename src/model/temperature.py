"""
temperature gate  (a MODULATOR, not a 4th module)
=============================================================================
temperature doesn't MOVE PM2.5 the way wind does, and unlike elevation it isn't
a pairwise quantity either -- an inversion is a REGIONAL COLUMN STATE, not a
property of a sensor pair. three sensors sitting in the same air feel the same
cap regardless of the vectors between them. so this gate is PER-NODE, not
per-edge; a per-edge temperature gate would double-count geometry the elevation
gate already owns.

    gate(s) = exp(a * s)                    s = standardized surface temperature

    per NODE, CENTERED AT 1 for a neutral airmass. the elevation gate is a
    per-EDGE modulator; this is its per-NODE sibling. it scales BOTH transport
    terms' contributions at the destination node inside a GraPhyLayer, while
    local (a node source/sink) and the layer's outer residual stay ungated:

        d = diffusion(h)  * temp_gate[:, None]   both transports, per dst node
        c = convection(h) * temp_gate[:, None]
        l = local(h)                             ungated
        h_out = h + fuse(l, d, c)                outer residual = ungated self-term

    NEUTRAL airmass (s ~ 0) -> gate ~ 1: transport flows at its baseline.
    COLD / inversion (gate < 1): the node DECOUPLES from its neighbours, its
    transport shrinks, so h_out stays near h -- PM2.5 trapped. HOT / convective
    (gate > 1): buoyant mixing ENHANCES transport above baseline. unlike the
    elevation gate this is NOT capped at 1 -- heat must be able to amplify, not
    just "return to normal".

WHY LEARNABLE, UNCONSTRAINED SIGN:
    for HORIZONTAL graph transport the sign is genuinely arguable -- cold
    surface usually means a stable, capped, stagnant column (suppress), but the
    net effect on sensor-to-sensor exchange is empirical. `a` carries both the
    direction AND the steepness; `b` is the threshold. we let the data pick the
    sign rather than bake in an assumption.

WHY SURFACE TEMPERATURE (and not PBLH / a lapse rate):
    we only have surface temperature -- no vertical profile, no boundary-layer
    height. so `s` is a surface PROXY for stability, not a true inversion
    strength. temp-only keeps it interpretable and matches the elevation gate's
    single-signal shape; richer proxies (diurnal, RH) can come later.

INERT AT INIT (ready-but-inert, like the elevation gate):
    a = 0  ->  exp(0) = 1 everywhere  ->  no-op until trained.
    with a placeholder constant temperature this stays a no-op regardless.

standardize the temperature before feeding it in (zero mean, unit variance over
the training window) so `a` and `b` live on a sane scale -- see standardize().

run the demo:   .venv/bin/python -m src.model.temperature
"""
import torch
import torch.nn as nn


class TemperatureGate(nn.Module):
    """Per-node multiplicative gate from a standardized surface-temp signal.

    gate(s) = exp(a * s), CENTERED AT 1 for a neutral airmass (s=0). `a` learns
    both sign (direction) and steepness; cold can suppress below 1 (inversion
    trapping) and hot can amplify above 1 (convective enhancement). Initialised
    inert: a=0 -> gate == 1 everywhere.

    NB not capped at 1 (that's the elevation gate's job). exp() grows fast and
    is unbounded, so `max_gain` bounds the gate to a plausible multiplicative
    band [1/max_gain, max_gain] -- a runaway `a` on an extreme temperature can't
    then explode the aggregated message. Default 2.5: convective enhancement of
    horizontal exchange is modest (~2-2.5x), and suppression bottoms at 1/2.5.
    """

    def __init__(self, max_gain: float = 2.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))   # a=0 -> exp(0)=1, inert
        self.log_gain = float(torch.log(torch.tensor(float(max_gain))))

    def forward(self, s):
        """s: standardized surface temperature per node [N]  ->  gate [N] in [1/max_gain, max_gain]."""
        return torch.exp((self.a * s).clamp(-self.log_gain, self.log_gain))


def standardize(temp_c, mean=None, std=None, eps: float = 1e-6):
    """Surface temperature (°C) -> zero-mean, unit-variance signal for the gate.

    Pass the TRAINING-window mean/std at inference so train and test share one
    scale; omit them to compute from `temp_c` itself. Returns (s, mean, std).
    """
    t = torch.as_tensor(temp_c, dtype=torch.float)
    m = t.mean() if mean is None else torch.as_tensor(mean, dtype=torch.float)
    sd = t.std() if std is None else torch.as_tensor(std, dtype=torch.float)
    return (t - m) / (sd + eps), m, sd


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gate = TemperatureGate()
    print(f"init:  a={gate.a.item():.3f}  (inert -> gate == 1)   band=[{1/2.5:.2f}, 2.5]")

    # a spread of surface temperatures, cold to warm
    temp_c = torch.tensor([-10., -5., 0., 5., 10., 20., 30.])
    # standardize against a realistic Pittsburgh full-year climatology, so the
    # numbers below reflect what the trained gate would actually see.
    s, m, sd = standardize(temp_c, mean=11.0, std=9.0)
    print(f"\nstandardized against Pittsburgh climatology mean={m.item():.0f}C std={sd.item():.0f}C")

    print("\n--- at init (inert) ---")
    g = gate(s)
    for t, gv in zip(temp_c.tolist(), g.tolist()):
        print(f"  {t:+6.0f}C   gate={gv:5.3f}")

    # LEARNED example: a>0 -> cold suppresses (inversion), hot amplifies (convective).
    # a=0.3 gives a plausible ~1.9x at a hot 30C and ~0.5x at a cold -10C.
    with torch.no_grad():
        gate.a.fill_(0.3)
    print("\n--- example learned gate (a=+0.3: neutral=1, cold<1, hot>1) ---")
    g = gate(s)
    for t, gv in zip(temp_c.tolist(), g.tolist()):
        bar = "#" * int(gv * 15)
        flag = "  <- baseline" if abs(gv - 1.0) < 0.05 else ""
        print(f"  {t:+6.0f}C   gate={gv:5.3f}  {bar}{flag}")
    print("\ncold -> gate<1 (trapped, decoupled); neutral -> 1; hot -> gate>1 (enhanced).")
