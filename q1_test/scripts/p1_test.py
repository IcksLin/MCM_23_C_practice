# -*- coding: utf-8 -*-
"""P1 minimal vertical chain (AGENT.md section 7):
   real Excel read -> clean -> build MILP -> solve -> audit -> test tables.

Uses real data but a reduced time limit to confirm feasibility and that
constraint violations are within tolerance.
"""
import sys, time, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from algorithms import io_data, preprocess, solve as solve_mod
from algorithms import validate, export_excel
from algorithms import paths

paths.ensure_dirs()
raw = io_data.load_inputs()
md = preprocess.preprocess(raw)
print("P1: data ready. Solving scenario 1 (short limit) ...")

m, res, z = solve_mod.solve_primary(
    md, scenario=1, eta=0.5, demand_scale=1.0,
    time_limit=60.0, mip_gap=0.05, seed=2024, disp=True,
    log_path=paths.LOG_DIR / "p1_scenario1.log")
print(f"\nstatus={res.status} msg={res.message}")
print(f"is_feasible={res.is_feasible}  Z1* = {-res.fun if res.is_feasible else 'nan'}")
print(f"mip_gap={res.mip_gap} dual={res.dual_bound} nodes={res.node_count} "
      f"time={res.time:.1f}s")

if not res.is_feasible:
    print("P1: no feasible incumbent -> FAIL")
    sys.exit(1)

# extract & full audit
sol = solve_mod.extract_solution(res, m, md)
print(f"\nx rows (nonzero area)={len(sol['x'])}")
print(f"profit_recomputed={sol['profit_recomputed']:.2f}  "
      f"solver_obj={sol['objective']:.2f}")
print(f"diff={abs(sol['profit_recomputed'] - sol['objective']):.4f}")

# total area per year-season (should == sum A_j for必种 slots)
xa = sol["x"]
g = xa.groupby(["year", "season"])["area"].sum()
print("\narea per (year,season):")
print(g.to_string())

audit = validate.validate_solution(sol, md)
for key, value in audit.to_dict().items():
    if key != "extra":
        print(f"{key}={value}")

# P1 uses an isolated temporary workbook and never touches canonical outputs.
# This also avoids false failures when the output directory is synchronized or
# protected by another desktop process.
with tempfile.TemporaryDirectory(prefix="q1_p1_") as tmp_dir:
    p1_workbook = Path(tmp_dir) / "p1_test.xlsx"
    export_excel.export_result_workbook(sol, md, paths.TEMPLATE1_PATH, p1_workbook)
    roundtrip = export_excel.reread_audit(p1_workbook, sol, md)
audit.excel_roundtrip_diff = roundtrip
print(f"excel_roundtrip_diff={roundtrip:.6e}")

print(f"\nNaN/Inf in x: {np.any(np.isnan(res.x))} / {np.any(np.isinf(res.x))}")
passed = (res.is_feasible and audit.feasible and roundtrip < 1e-4
          and not np.any(np.isnan(res.x)) and not np.any(np.isinf(res.x)))
print("\nP1: PASS" if passed else "P1: FAIL")
sys.exit(0 if passed else 1)
