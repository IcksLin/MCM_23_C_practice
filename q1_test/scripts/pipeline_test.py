# -*- coding: utf-8 -*-
"""End-to-end pipeline check on a short solve: solve -> validate -> export -> reread."""
import sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms import io_data, preprocess, model as model_mod, solve as solve_mod
from algorithms import validate, export_excel, paths

paths.ensure_dirs()
raw = io_data.load_inputs()
md = preprocess.preprocess(raw)

tmp_ctx = tempfile.TemporaryDirectory(prefix="q1_pipeline_")
tmp_dir = Path(tmp_ctx.name)

print("=== solve scenario 1 (90s) ===")
m, res, z = solve_mod.solve_primary(
    md, scenario=1, eta=0.5, demand_scale=1.0,
    time_limit=90.0, mip_gap=0.05, seed=2024, disp=False,
    log_path=tmp_dir / "pipeline_s1.log")
print(f"feasible={res.is_feasible} Z1={-res.fun if res.is_feasible else 'nan'} "
      f"gap={res.mip_gap:.3f} time={res.time:.1f}s")
if not res.is_feasible:
    sys.exit(1)

sol = solve_mod.extract_solution(res, m, md)
rep = validate.validate_solution(sol, md)
print("\n=== audit ===")
for k, v in rep.to_dict().items():
    if k != "extra":
        print(f"  {k}: {v}")

print("\n=== export scenario 1 ===")
out1 = export_excel.export_result_workbook(
    sol, md, paths.TEMPLATE1_PATH, tmp_dir / "result1_1.xlsx")
rt = export_excel.reread_audit(out1, sol, md)
print(f"exported -> {out1}")
print(f"excel_roundtrip_max_diff = {rt:.6e}")

# scenario 2
print("\n=== solve scenario 2 (90s) ===")
m2, res2, z2 = solve_mod.solve_primary(
    md, scenario=2, eta=0.5, demand_scale=1.0,
    time_limit=90.0, mip_gap=0.05, seed=2024, disp=False,
    log_path=tmp_dir / "pipeline_s2.log")
print(f"feasible={res2.is_feasible} Z2={-res2.fun if res2.is_feasible else 'nan'} "
      f"gap={res2.mip_gap:.3f}")
if res2.is_feasible:
    sol2 = solve_mod.extract_solution(res2, m2, md)
    rep2 = validate.validate_solution(sol2, md)
    print(f"  s2 audit feasible={rep2.feasible} "
          f"max_area_viol={rep2.max_area_conservation_violation:.2e}")
    out2 = export_excel.export_result_workbook(
        sol2, md, paths.TEMPLATE2_PATH, tmp_dir / "result1_2.xlsx")
    rt2 = export_excel.reread_audit(out2, sol2, md)
    print(f"  s2 excel_roundtrip_max_diff = {rt2:.6e}")
    print(f"  Z2 >= Z1?  {z2 >= z - 1e-6}   (Z1={z:.2f} Z2={z2:.2f})")

passed = (rep.feasible and rt < 1e-4 and res2.is_feasible
          and rep2.feasible and rt2 < 1e-4)
print("\nPIPELINE OK" if passed else "PIPELINE FAIL")
tmp_ctx.cleanup()
sys.exit(0 if passed else 1)
