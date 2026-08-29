# -*- coding: utf-8 -*-
"""固化 Q3 原始与缩减情景的 Kendall 门禁证据。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

Q3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Q3_ROOT))

from algorithms import paths
from algorithms.dependency import BASE_LOADINGS, DependencyConfig
from algorithms.elasticity import ElasticityConfig
from algorithms.io_data import load_inputs, load_q2_baseline
from algorithms.preprocess import preprocess
from algorithms.scenario_reduction import reduce_scenarios
from algorithms.scenarios import generate_raw_scenarios


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--out", type=Path,
                   default=paths.Q3_OUT_DIR / "reduction_audit_q3.csv")
    a = p.parse_args()
    data = preprocess(load_inputs())
    q2 = load_q2_baseline(paths.Q2_SELECTED_PLAN)
    s = generate_raw_scenarios(
        data, n=a.n, seed=a.seed,
        dependency_cfg=DependencyConfig(5, 1.0, 0.5, BASE_LOADINGS),
        elasticity_cfg=ElasticityConfig(1.0))
    _, audit = reduce_scenarios(data, s, k=a.k, baseline_plan=q2.area,
                                gamma=0.03, seed=a.seed)
    row = {
        "seed": a.seed, "raw_n": a.n, "reduced_k": a.k,
        "audit_columns": len(s.audit_labels or []),
        "audit_pairs": int(s.dependency_audit.pair_count),
        "raw_max_kendall_error": s.dependency_audit.max_kendall_error,
        "raw_threshold": 0.05,
        "raw_pass": s.dependency_audit.max_kendall_error <= 0.05,
        "reduced_max_kendall_error": audit.max_kendall_error,
        "reduced_threshold": 0.15,
        "reduced_direction_consistent": audit.kendall_direction_consistent,
        "reduced_pass": (audit.max_kendall_error <= 0.15
                         and audit.kendall_direction_consistent),
        "weight_sum": audit.sum_weights,
        "tail_representatives": audit.min_profit_layer_reps,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(a.out, index=False, encoding="utf-8-sig")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
