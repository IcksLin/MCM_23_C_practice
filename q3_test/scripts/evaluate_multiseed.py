# -*- coding: utf-8 -*-
"""对冻结 Q3 主候选做多随机种子共同随机数样本外复评。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Q3_ROOT))

from algorithms import paths
from algorithms.dependency import BASE_LOADINGS, DependencyConfig
from algorithms.elasticity import ElasticityConfig
from algorithms.evaluate import (_load_q2_plan_dict, bootstrap_paired_difference,
                                 paired_compare)
from algorithms.io_data import load_inputs
from algorithms.preprocess import LEGUME_CODES, preprocess
from algorithms.scenarios import generate_raw_scenarios


def load_q3_plan(path: Path, data) -> dict:
    df = pd.read_csv(path)
    x = {}
    for row in df.itertuples(index=False):
        j = data.plot_idx[str(row.plot)]
        key = (j, int(row.crop_code), int(row.year), int(row.season))
        x[key] = float(row.area)
    y = {key: int(area > 1e-7) for key, area in x.items()}
    b = {}
    for yi, t in enumerate(data.years):
        for j in range(len(data.plot_names)):
            if yi == 0:
                val = int(any(data.bar_y.get((j, i, s), 0)
                              for s in data.plot_seasons[j]
                              for i in LEGUME_CODES))
            else:
                prev = data.years[yi - 1]
                val = int(any(y.get((j, i, prev, s), 0)
                              for s in data.plot_seasons[j]
                              for i in LEGUME_CODES))
            b[(j, t)] = val
    w = {key: area * b[(key[0], key[2])]
         for key, area in x.items() if key[1] not in LEGUME_CODES}
    return {"x": x, "y": y, "b": b, "w": w, "r": {}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="3024,3025,3026")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=Path,
                        default=paths.Q3_OUT_DIR / "multiseed_oos_q3.csv")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    data = preprocess(load_inputs())
    q3 = load_q3_plan(paths.SELECTED_PLAN_Q3, data)
    q2 = _load_q2_plan_dict(paths.Q2_SELECTED_PLAN, data)
    dep = DependencyConfig(5, 1.0, 0.5, BASE_LOADINGS)
    ela = ElasticityConfig(scale=1.0)
    rows = []
    for pos, seed in enumerate(seeds, 1):
        print(f"[{pos}/{len(seeds)}] seed={seed}, N={args.n}", flush=True)
        scenarios = generate_raw_scenarios(
            data, n=args.n, seed=seed, dependency_cfg=dep,
            elasticity_cfg=ela)
        report = paired_compare(q3, q2, scenarios, data, gamma=0.03)
        ci = bootstrap_paired_difference(report.delta_profit,
                                         n_boot=2000, seed=seed + 17)
        rows.append({
            "seed": seed, "n": args.n,
            "raw_kendall_max_error": scenarios.dependency_audit.max_kendall_error,
            "q3_mean": report.q3_frame.mean,
            "q3_cvar90": report.q3_frame.cvar,
            "q3_p10": report.q3_frame.p10,
            "q3_min": report.q3_frame.min_profit,
            "q2_mean": report.q2_frame.mean,
            "q2_cvar90": report.q2_frame.cvar,
            "mean_gain": report.mean_delta,
            "gain_pct": report.mean_delta / report.q2_frame.mean,
            "gain_ci95_low": ci.lower_bound,
            "gain_ci95_high": ci.upper_bound,
            "positive_pair_rate": float(np.mean(report.delta_profit > 0)),
        })
        print(f"  Q3={report.q3_frame.mean:,.0f}, Q2={report.q2_frame.mean:,.0f}, "
              f"gain={report.mean_delta:,.0f}", flush=True)
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False), flush=True)
    print(f"written: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
