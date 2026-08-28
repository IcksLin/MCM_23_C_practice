# -*- coding: utf-8 -*-
"""Independent deterministic checks plus one real Q2 MILP vertical slice."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

Q2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Q2_ROOT))

from algorithms import io_data, paths, preprocess
from algorithms.export_ooxml import patch_result2
from algorithms.risk import compute_cvar, recompute_scenario_profits, select_unique_plan
from algorithms.scenario_reduction import reduce_scenarios
from algorithms.scenarios import generate_raw_scenarios
from algorithms.solve import solve_lexicographic
from algorithms.validate import validate_solution


def _assert_lhs_strata(values: np.ndarray, lo: float, hi: float) -> None:
    unit = (np.asarray(values) - lo) / (hi - lo)
    bins = np.floor(np.clip(unit, 0, 1 - np.finfo(float).eps) * len(unit)).astype(int)
    assert np.array_equal(np.sort(bins), np.arange(len(unit)))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths.ensure_dirs()
    data = preprocess.preprocess(io_data.load_inputs())
    # P1 uses all real plots/crops but only the first two planning years, as
    # required by the vertical-slice contract.
    data.years = [2024, 2025]
    allowed_years = {2023, *data.years}
    data.adj_pairs = [item for item in data.adj_pairs
                      if item[2][0] in allowed_years and item[3][0] in allowed_years]
    data.legume_windows = [item for item in data.legume_windows
                           if all(year in allowed_years for year in item[1])]
    raw = generate_raw_scenarios(data, n=10, seed=2024, distribution="uniform")

    wheat = raw.demand[(6, 2024, 1)] / data.D[(6, 1)]
    bean = raw.demand[(1, 2024, 1)] / data.D[(1, 1)]
    mushroom = raw.price[(38, 2024, 2)] / data.p[(38, 2)]
    assert wheat.min() >= 1.05 and wheat.max() <= 1.10
    assert bean.min() >= 0.95 and bean.max() <= 1.05
    assert mushroom.min() >= 0.95 and mushroom.max() <= 0.99
    _assert_lhs_strata(wheat - 1.0, 0.05, 0.10)

    baseline = {(j, i, year, season): area
                for (j, i, season), area in data.bar_x.items()
                for year in data.years}
    reduced = reduce_scenarios(data, raw, k=3, baseline_plan=baseline, seed=2024)
    assert reduced.k == 3 and len(set(reduced.indices.tolist())) == 3
    assert np.isclose(reduced.weights.sum(), 1.0) and np.all(reduced.weights > 0)

    lex = solve_lexicographic(data, reduced, beta=0.90, risk_lambda=1.0,
                              eta=0.5, time_limit=10, mip_gap=0.8,
                              eps_z=None, eps_e=None)
    assert lex.get("solution") is not None
    assert lex.get("lex_complete") and lex.get("final_stage") == 3
    plan = lex["solution"]
    profits = recompute_scenario_profits(plan["x"], reduced, data)
    expected = float(np.average(profits, weights=reduced.weights))
    cvar = compute_cvar(profits, reduced.weights, 0.90)
    max_sales_diff = 0.0
    for (omega, i, year, season), sold in plan["u"].items():
        if sold <= 1e-8:
            continue
        production = plan["Q"][(omega, i, year, season)]
        demand_arr = reduced.demand.get((i, year, season))
        demand = demand_arr[omega] if demand_arr is not None else 0.0
        # 约束检查: u <= Q 且 u <= D
        violation = max(0.0, sold - production, sold - demand)
        max_sales_diff = max(max_sales_diff, violation)
    assert max_sales_diff <= 1e-4, f"sales constraint violation: {max_sales_diff}"

    with tempfile.TemporaryDirectory(dir=paths.LOG_DIR) as tmp:
        workbook = Path(tmp) / "result2_test.xlsx"
        excel_audit = patch_result2(plan, data, paths.TEMPLATE2_PATH, workbook)
        assert excel_audit["changed_sheet_count"] == len(data.years)
        assert excel_audit["nonzero_cell_count"] > 0
        audit = validate_solution(
            plan, data, reduced, beta=0.90, risk_lambda=1.0,
            z_lambda=cvar, e_pi=expected, cvar_value=cvar, eta=0.5,
            excel_audit=excel_audit,
            solver_info={"status": lex["result"].status,
                         "dual_bound": lex["result"].dual_bound,
                         "mip_gap": lex["result"].mip_gap,
                         "time": lex["result"].time,
                         "certified": False})
    assert audit["feasible"], audit

    frontier = pd.DataFrame([
        {"lambda": 0.0, "expected_profit": 10.0, "lower_tail_cvar": 9.0,
         "n_activations": 2},
        {"lambda": 0.1, "expected_profit": 8.0, "lower_tail_cvar": 7.0,
         "n_activations": 2},
    ])
    assert select_unique_plan(frontier) == "0.0"
    stages = [(k, lex[f"result{k}"].status, lex[f"result{k}"].mip_gap)
              for k in (1, 2, 3)]
    print(f"P1 stages={stages} eps_z={lex['eps_z']:.6f} eps_e={lex['eps_e']:.6f}")
    print(f"P1 expected={expected:.6f} cvar={cvar:.6f} sales_diff={max_sales_diff:.3e}")
    print(f"P1 OOXML={excel_audit}")
    print("P1 PASS: parameters, LHS, K=3 reduction, lambda=1 three-stage MILP, audit and OOXML")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
