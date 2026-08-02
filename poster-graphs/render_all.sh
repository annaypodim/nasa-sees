#!/usr/bin/env bash
# Regenerate every poster graph. Run from anywhere:
#   bash poster-graphs/render_all.sh
# To change a title: open the script, edit the TITLE = "..." line near the top,
# then rerun this (or just that one script).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/../.venv/bin/python"
cd "$DIR"
for s in terrain_gate_slc density_sweep_flat convection_contribution \
         sensor_map hrrr_commonmode_schematic covariate_spatial_temporal; do
    echo "-- $s"
    "$PY" "$s.py" 2>&1 | grep -v NotOpenSSL || true
done
echo "done. topo_corr_*.png come from ../poster-graphs/run_topo_corr.py (needs network)."
