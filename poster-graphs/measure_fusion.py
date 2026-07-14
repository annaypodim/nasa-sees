"""Measure the real fusion module shares [diffusion, convection, local].

Monkeypatches Fusion.forward to record the per-node softmax weights ONLY during
eval (model.eval() -> submodule.training == False), so training passes are
excluded. Runs eval_inductive.main() once per city (default config, wind=hrrr),
then averages the recorded weights over all eval nodes/layers/hours/seeds.

Column order matches model.py active-module order: [diffusion, convection, local].
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import src.model.fusion as fmod          # noqa: E402
import eval_inductive as ev              # noqa: E402

_orig_forward = fmod.Fusion.forward
REC = []


def _patched(self, *outs):
    out, w = _orig_forward(self, *outs)
    if not self.training:                 # eval phase only
        REC.append(w.detach().mean(0).cpu().numpy())   # mean over nodes -> [n_modules]
    return out, w


fmod.Fusion.forward = _patched

LABELS3 = ["diffusion", "convection", "local"]
results = {}
for city in ["fresno", "slc"]:
    REC.clear()
    sys.argv = ["eval_inductive", "--city", city, "--wind", "hrrr",
                "--seeds", "0,1,2", "--epochs", "120"]
    try:
        ev.main()
    except SystemExit:
        pass
    arr = np.stack(REC)                   # [n_eval_calls, n_modules]
    mean = arr.mean(axis=0)
    n = arr.shape[1]
    labels = LABELS3 if n == 3 else [f"module_{i}" for i in range(n)]
    results[city] = {lab: float(v) for lab, v in zip(labels, mean)}
    print(f"\n[fusion shares] {city}: " +
          "  ".join(f"{lab}={v:.3f}" for lab, v in results[city].items()) +
          f"   (from {arr.shape[0]} eval calls)")

out = ROOT / "poster-graphs" / "fusion_shares.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nsaved -> {out}")
