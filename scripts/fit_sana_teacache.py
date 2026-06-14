#!/usr/bin/env python3
"""Fit / diagnose the TeaCache-style rescale polynomial for SANA-Video.

The DiT (SGLANG_SANA_TEACACHE_CALIB=<path>) writes one line per computed step:

    branch  input_relL1  block0out_relL1  output_relL1

Two candidate CHEAP signals are logged per step:
  - input_relL1     : rel-L1 of block-0's modulated INPUT  (what TeaCache uses)
  - block0out_relL1 : rel-L1 of block-0's OUTPUT           (what FBCache uses)
The EXPENSIVE target is output_relL1 = rel-L1 of the full block-stack residual
(the quantity the cache reuses). A skip-scheduler only works if a cheap signal
predicts the target; this script reports the correlation/fit for BOTH so we can
tell whether any prediction-based cache is viable, and emit coeffs if so.

numpy.polyfit returns highest-degree-first == the DiT Horner order -> paste into
--coeffs / SGLANG_SANA_TEACACHE_COEFFS.

Usage: python scripts/fit_sana_teacache.py <pairs.txt> [degree]
"""
from __future__ import annotations

import sys

import numpy as np


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def fit_report(name, x, y, want_deg):
    print(f"\n----- signal: {name}  (corr={np.corrcoef(x, y)[0,1]:.4f}) -----")
    print(f"  x in [{x.min():.4f}, {x.max():.4f}] mean {x.mean():.4f}")
    xs = np.linspace(0.0, x.max(), 200)
    print(f"  {'deg':>3s} {'R^2':>8s} {'mono↑':>6s} {'min':>8s}   coeffs")
    chosen = None
    for deg in range(1, max(want_deg, 4) + 1):
        c = np.polyfit(x, y, deg)
        ps = np.polyval(c, xs)
        mono = bool(np.all(np.diff(ps) >= -1e-9))
        r2v = r2(y, np.polyval(c, x))
        cs = ",".join(f"{v:.6e}" for v in c)
        print(f"  {deg:>3d} {r2v:>8.4f} {str(mono):>6s} {ps.min():>8.3f}   {cs}")
        if deg == want_deg:
            chosen = (c, r2v, mono, ps.min())
    return chosen


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/yitongl/sana_video/calib_pairs.txt"
    want_deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    cols = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) == 4:
                cols.append((float(p[1]), float(p[2]), float(p[3])))
            elif len(p) == 3:  # back-compat: branch input output (no block0)
                cols.append((float(p[1]), float("nan"), float(p[2])))
    if not cols:
        print(f"ERROR: no rows in {path}")
        sys.exit(1)

    a = np.array(cols, dtype=np.float64)
    x_in, x_b0, y = a[:, 0], a[:, 1], a[:, 2]
    print(f"rows: {len(cols)}")
    print(f"output_relL1 (target) in [{y.min():.4f}, {y.max():.4f}] mean {y.mean():.4f}")

    c_in = fit_report("block0-INPUT (TeaCache)", x_in, y, want_deg)
    has_b0 = not np.isnan(x_b0).any()
    c_b0 = fit_report("block0-OUTPUT (FBCache)", x_b0, y, want_deg) if has_b0 else None

    print("\n=== verdict ===")
    corr_in = abs(np.corrcoef(x_in, y)[0, 1])
    if has_b0:
        corr_b0 = abs(np.corrcoef(x_b0, y)[0, 1])
        print(f"corr(input,out)={corr_in:.3f}   corr(block0out,out)={corr_b0:.3f}")
        best_name, best = ("block0-OUTPUT/FBCache", c_b0) if corr_b0 > corr_in else ("block0-INPUT/TeaCache", c_in)
    else:
        print(f"corr(input,out)={corr_in:.3f}")
        best_name, best = "block0-INPUT/TeaCache", c_in
    c, r2v, mono, pmin = best
    print(f"best signal: {best_name}  (deg {want_deg}: R^2={r2v:.4f}, mono↑={mono}, min={pmin:.3f})")
    if r2v < 0.3:
        print("VERDICT: no cheap signal predicts output drift (R^2 < 0.3) -> "
              "prediction-based caching is NOT viable for this model; prefer fixed late-skip.")
    print("COEFFS=" + ",".join(f"{v:.6e}" for v in c))


if __name__ == "__main__":
    main()
