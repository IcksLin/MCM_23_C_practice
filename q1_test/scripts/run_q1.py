# -*- coding: utf-8 -*-
"""Q1 main CLI runner (AGENT.md sections 8, 12, 14).

Workflow:
  1. verify inputs (SHA-256) & ensure dirs
  2. load + preprocess -> ModelData
  3. export cleaned data tables
  4. per scenario: solve primary -> solve lexicographic -> validate -> export
  5. yearly stats table
  6. (optional) sensitivity sweeps (eta / delta / demand)
  7. (optional) figures (with bound-history subsolves)
  8. (optional) text report + repro.json
  9. final summary block on stdout
"""
from __future__ import annotations
import sys
import time
import json
import hashlib
import platform
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# make q1_test root (parent of scripts/) importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import scipy
import matplotlib
import openpyxl

from algorithms import paths, io_data, preprocess, solve as solve_mod
from algorithms import validate, export_excel, plots
from algorithms.plots import yearly_economics


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return io_data.sha256_file(Path(path))


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(paths.PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "none"


def _dep_versions() -> dict:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "openpyxl": openpyxl.__version__,
        "solver": "HiGHS via scipy.optimize.milp",
    }


def _fmt_num(x):
    try:
        if x is None or (isinstance(x, float) and (x != x)):
            return "nan"
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def _gap_str(g):
    """Format a MIP gap; 'nan' for NaN/None (avoids nested f-strings)."""
    try:
        if g is None or (isinstance(g, float) and (g != g)):
            return "nan"
        return f"{float(g):.4f}"
    except Exception:
        return str(g)


def _progress(percent: int, label: str) -> None:
    """Render a deterministic terminal progress bar without extra packages."""
    percent = max(0, min(100, int(percent)))
    width = 32
    filled = round(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r[{bar}] {percent:3d}%  {label}", flush=True)


def _solve_certified(res, requested_gap: float) -> bool:
    """Whether a feasible solve meets the requested optimality tolerance."""
    if not res.is_feasible:
        return False
    if res.status == 0:
        return True
    return bool(res.mip_gap == res.mip_gap and
                res.mip_gap <= requested_gap + 1e-12)


# ---------------------------------------------------------------------------
# cleaned-data export (step 3)
# ---------------------------------------------------------------------------

def export_cleaned_data(md, raw) -> list:
    out_dir = paths.DATA_CLEAN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    # plots.csv
    plots_df = pd.DataFrame({
        "plot_idx": list(range(len(md.plot_names))),
        "name": md.plot_names,
        "type": md.plot_type,
        "area": md.plot_area,
        "seasons": [",".join(str(s) for s in ss) for ss in md.plot_seasons],
    })
    p = out_dir / "plots.csv"; plots_df.to_csv(p, index=False); made.append(p)

    # crops.csv
    rows = []
    for code in md.crop_codes:
        rows.append({"code": code, "name": md.crop_names.get(code, ""),
                     "is_legume": int(code in md.legume_set)})
    crops_df = pd.DataFrame(rows)
    p = out_dir / "crops.csv"; crops_df.to_csv(p, index=False); made.append(p)

    # planting_2023.csv
    rows = []
    for (j, i, s), area in md.bar_x.items():
        rows.append({"plot_idx": j, "plot": md.plot_names[j],
                     "crop_code": i, "crop": md.crop_names.get(i, ""),
                     "season": s, "area_2023": area})
    plant_df = pd.DataFrame(rows, columns=["plot_idx", "plot", "crop_code",
                                           "crop", "season", "area_2023"])
    p = out_dir / "planting_2023.csv"; plant_df.to_csv(p, index=False); made.append(p)

    # stats_2023.csv (dedup by plot_type, crop_code, season; includes inherited)
    stat_map = {}
    for (j, i, s) in md.suit:
        ptype = md.plot_type[j]
        key = (ptype, i, s)
        if key in stat_map:
            continue
        stat_map[key] = {
            "plot_type": ptype, "crop_code": i,
            "crop_name": md.crop_names.get(i, ""), "season": s,
            "yield": md.q.get((j, i, s), 0.0), "cost": md.c.get((j, i, s), 0.0),
            "price_mid": md.p.get((i, s), 0.0),
            "price_lo": md.p_lo.get((i, s), 0.0),
            "price_hi": md.p_hi.get((i, s), 0.0),
        }
    stats_df = pd.DataFrame(list(stat_map.values()))
    stats_df = stats_df.sort_values(["plot_type", "crop_code", "season"])
    p = out_dir / "stats_2023.csv"; stats_df.to_csv(p, index=False); made.append(p)

    # data_assertions.txt
    p = out_dir / "data_assertions.txt"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("data assertions\n==============\n")
        for k, v in md.assertions.items():
            fh.write(f"{k}: {v}\n")
    made.append(p)

    # clean_log.csv
    if md.clean_log is not None and len(md.clean_log):
        p = out_dir / "clean_log.csv"
        md.clean_log.to_csv(p, index=False)
        made.append(p)
    return made


# ---------------------------------------------------------------------------
# scenario runner (step 4)
# ---------------------------------------------------------------------------

def run_scenario(md, scenario, args, log_name):
    """Solve primary + lexicographic, validate, export. Robust to errors."""
    log_path = paths.LOG_DIR / log_name
    result = {
        "scenario": scenario, "ok": False, "error": None,
        "Z": float("nan"), "primary_Z": float("nan"),
        "status": None, "mip_gap": float("nan"),
        "dual_bound": float("nan"), "node_count": 0, "time": 0.0,
        "message": "", "lex_status": None, "lex_feasible": False,
        "primary_certified": False, "lex_certified": False,
        "certified": False,
        "sol": None, "audit": None, "out_path": None,
        "excel_roundtrip_diff": float("nan"),
    }
    try:
        m, res, z_star = solve_mod.solve_primary(
            md, scenario=scenario, eta=args.eta,
            demand_scale=args.demand_scale, time_limit=args.time_limit,
            mip_gap=args.mip_gap, seed=args.seed, disp=False,
            log_path=log_path)
        result["status"] = res.status
        result["message"] = res.message
        result["mip_gap"] = res.mip_gap
        result["dual_bound"] = res.dual_bound
        result["node_count"] = res.node_count
        result["time"] = res.time
        result["primary_certified"] = _solve_certified(res, args.mip_gap)
        if not res.is_feasible:
            result["error"] = f"primary infeasible/timeout: {res.message}"
            return result
        result["primary_Z"] = z_star
        sol_primary = solve_mod.extract_solution(res, m, md)
        sol_primary["objective"] = z_star

        # lexicographic stage 2
        mlex, reslex = solve_mod.solve_lexicographic(
            md, scenario=scenario, z_star=z_star, eta=args.eta,
            demand_scale=args.demand_scale, delta=args.delta,
            time_limit=args.time_limit, mip_gap=args.mip_gap,
            seed=args.seed, disp=False, log_path=log_path)
        result["lex_status"] = reslex.status
        result["lex_feasible"] = bool(reslex.is_feasible)
        result["lex_certified"] = _solve_certified(reslex, args.mip_gap)
        sol = sol_primary   # default: primary solution (max profit)
        if reslex.is_feasible:
            sol_lex = solve_mod.extract_solution(reslex, mlex, md)
            # Numerical safety net: the lexicographic Z-constraint should
            # guarantee profit >= (1-delta)*z_star, but tiny solver tolerances
            # can let the recomputed profit dip just under the floor.  Fall
            # back to the primary (max-profit) solution in that case so the
            # exported plan always respects the intended profit floor.
            lex_profit = sol_lex.get("profit_recomputed", float("nan"))
            floor = (1.0 - args.delta) * z_star
            if (lex_profit == lex_profit  # not nan
                    and lex_profit < floor - 1e-6):
                print(f"  [lex] 字典序解真实利润 {lex_profit:.2f} 低于 "
                      f"{floor:.2f}，回退到主解（数值容差内）")
                result["lex_feasible"] = False
            else:
                sol = sol_lex

        result["Z"] = sol.get("profit_recomputed", float("nan"))

        audit = validate.validate_solution(sol, md)
        template = (paths.TEMPLATE1_PATH if scenario == 1
                    else paths.TEMPLATE2_PATH)
        out_path = (paths.RESULT1_PATH if scenario == 1
                    else paths.RESULT2_PATH)
        if not audit.feasible:
            result["error"] = "solution audit failed before Excel export"
            return result

        candidate = out_path.with_name(out_path.stem + ".candidate" + out_path.suffix)
        candidate.unlink(missing_ok=True)
        export_excel.export_result_workbook(sol, md, template, candidate)
        rt = export_excel.reread_audit(candidate, sol, md)
        audit.excel_roundtrip_diff = rt
        if rt >= 1e-4:
            candidate.unlink(missing_ok=True)
            audit.feasible = False
            result["error"] = f"Excel round-trip diff too large: {rt}"
            result["audit"] = audit
            return result
        candidate.replace(out_path)
        certified = (result["primary_certified"] and
                     result["lex_feasible"] and result["lex_certified"])
        result.update({"ok": True, "certified": certified,
                       "sol": sol, "audit": audit,
                       "out_path": out_path,
                       "excel_roundtrip_diff": rt})
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    return result


# ---------------------------------------------------------------------------
# audit table (step 4 output)
# ---------------------------------------------------------------------------

def write_audit_row(rows, result):
    a = result["audit"]
    ad = a.to_dict() if a is not None else {}
    rows.append({
        "scenario": result["scenario"],
        "Z": result["Z"], "status": result["status"],
        "mip_gap": result["mip_gap"], "dual_bound": result["dual_bound"],
        "node_count": result["node_count"], "time": result["time"],
        "primary_certified": int(result.get("primary_certified", False)),
        "lex_status": result["lex_status"],
        "lex_feasible": int(result["lex_feasible"]),
        "lex_certified": int(result.get("lex_certified", False)),
        "certified": int(result.get("certified", False)),
        "max_area_viol": ad.get("max_area_conservation_violation", ""),
        "max_unsuitable_area": ad.get("max_unsuitable_area", ""),
        "max_upper_link_viol": ad.get("max_upper_link_violation", ""),
        "max_min_area_viol": ad.get("max_min_area_violation", ""),
        "max_Q_balance_viol": ad.get("max_production_balance_violation", ""),
        "max_irrigated_area_viol": ad.get("max_irrigated_area_violation", ""),
        "max_integrality_viol": ad.get("max_integrality_violation", ""),
        "monoculture_count": ad.get("monoculture_violation_count", ""),
        "legume_min_slack": ad.get("legume_min_slack", ""),
        "mode_conflicts": ad.get("irrigated_mode_conflict_count", ""),
        "max_u_exceeds_D": ad.get("max_u_exceeds_D", ""),
        "max_u_exceeds_Q": ad.get("max_u_exceeds_Q", ""),
        "profit_recompute_diff": ad.get("profit_recompute_diff", ""),
        "excel_roundtrip_diff": ad.get("excel_roundtrip_diff", result["excel_roundtrip_diff"]),
        "feasible": ad.get("feasible", ""),
    })


# ---------------------------------------------------------------------------
# sensitivity (step 7)
# ---------------------------------------------------------------------------

def run_sensitivity(md, z_star_s1, args):
    sens_eta, sens_delta, sens_demand = [], [], []
    eta_tl = min(args.time_limit, 120.0)

    # eta sweep (scenario 1 primary)
    for eta in (0.25, 0.5, 0.75, 1.0):
        try:
            _, res, z = solve_mod.solve_primary(
                md, scenario=1, eta=eta, demand_scale=args.demand_scale,
                time_limit=eta_tl, mip_gap=args.mip_gap, seed=args.seed,
                disp=False, log_path=paths.LOG_DIR / "sens_eta.log")
            certified = _solve_certified(res, args.mip_gap)
            sens_eta.append({"param": "eta", "value": eta,
                             "Z": z if certified else float("nan"),
                             "incumbent_Z": z, "certified": certified,
                             "gap": res.mip_gap, "time": res.time,
                             "status": res.status})
        except Exception as e:
            sens_eta.append({"param": "eta", "value": eta, "Z": float("nan"),
                             "gap": float("nan"), "time": 0.0,
                             "status": -1, "error": str(e)})

    # delta sweep (needs z_star from scenario-1 primary)
    if z_star_s1 == z_star_s1:  # not nan
        delta_tl = min(args.time_limit, 120.0)
        for delta in (0.0, 0.005, 0.01, 0.02):
            try:
                mlex, reslex = solve_mod.solve_lexicographic(
                    md, scenario=1, z_star=z_star_s1, eta=args.eta,
                    demand_scale=args.demand_scale, delta=delta,
                    time_limit=delta_tl, mip_gap=args.mip_gap,
                    seed=args.seed, disp=False,
                    log_path=paths.LOG_DIR / "sens_delta.log")
                Z = float("nan")
                incumbent_Z = float("nan")
                if reslex.is_feasible:
                    s = solve_mod.extract_solution(reslex, mlex, md)
                    incumbent_Z = s["profit_recomputed"]
                certified = _solve_certified(reslex, args.mip_gap)
                if certified:
                    Z = incumbent_Z
                sens_delta.append({"param": "delta", "value": delta, "Z": Z,
                                   "incumbent_Z": incumbent_Z,
                                   "certified": certified,
                                   "gap": reslex.mip_gap, "time": reslex.time,
                                   "status": reslex.status})
            except Exception as e:
                sens_delta.append({"param": "delta", "value": delta,
                                   "Z": float("nan"), "gap": float("nan"),
                                   "time": 0.0, "status": -1, "error": str(e)})
    else:
        print("[sensitivity] delta sweep skipped: no z_star from scenario 1")

    # demand sweep
    demand_tl = min(args.time_limit, 120.0)
    for lam in (0.9, 1.0, 1.1):
        try:
            _, res, z = solve_mod.solve_primary(
                md, scenario=1, eta=args.eta, demand_scale=lam,
                time_limit=demand_tl, mip_gap=args.mip_gap,
                seed=args.seed, disp=False,
                log_path=paths.LOG_DIR / "sens_demand.log")
            certified = _solve_certified(res, args.mip_gap)
            sens_demand.append({"param": "demand_scale", "value": lam,
                                "Z": z if certified else float("nan"),
                                "incumbent_Z": z, "certified": certified,
                                "gap": res.mip_gap, "time": res.time,
                                "status": res.status})
        except Exception as e:
            sens_demand.append({"param": "demand_scale", "value": lam,
                                "Z": float("nan"), "gap": float("nan"),
                                "time": 0.0, "status": -1, "error": str(e)})

    pd.DataFrame(sens_eta).to_csv(paths.SENS_ETA_PATH, index=False)
    pd.DataFrame(sens_delta).to_csv(paths.SENS_DELTA_PATH, index=False)
    pd.DataFrame(sens_demand).to_csv(paths.SENS_DEMAND_PATH, index=False)
    return sens_eta, sens_delta, sens_demand


# ---------------------------------------------------------------------------
# bound history for the convergence figure (step 8)
# ---------------------------------------------------------------------------

def build_bound_history(md, args):
    history = []
    for tl in (5.0, 15.0, 30.0):
        try:
            _, res, z = solve_mod.solve_primary(
                md, scenario=1, eta=args.eta, demand_scale=args.demand_scale,
                time_limit=tl, mip_gap=args.mip_gap, seed=args.seed,
                disp=False, log_path=paths.LOG_DIR / "bound_scan.log")
            if res.is_feasible:
                # primal = Z incumbent; dual reported by HiGHS is a lower bound
                # on the *minimisation* (-Z), so -dual_bound is the Z upper bound.
                history.append({
                    "time": res.time, "primal": z,
                    "dual": -res.dual_bound if res.dual_bound == res.dual_bound
                    else float("nan"),
                    "gap": res.mip_gap,
                })
        except Exception as e:
            print(f"[bound_scan] tl={tl} failed: {e}")
    return history


# ---------------------------------------------------------------------------
# report + repro.json (step 9)
# ---------------------------------------------------------------------------

def write_report(md, res1, res2, sens, args, generated, repro_extra):
    paths.DOC_DIR.mkdir(parents=True, exist_ok=True)
    md_path = paths.DOC_DIR / "Q1_建模实现报告.md"
    sens_eta, sens_delta, sens_demand = sens

    def _audit_table(result):
        a = result["audit"]
        if a is None:
            return "（无审计结果）"
        d = a.to_dict()
        return (
            f"| 指标 | 值 |\n|---|---|\n"
            f"| 面积守恒最大违反 | {d['max_area_conservation_violation']:.3e} |\n"
            f"| 非适种面积 | {d['max_unsuitable_area']:.3e} |\n"
            f"| x≤Ay 最大违反 | {d['max_upper_link_violation']:.3e} |\n"
            f"| 最小面积违反 | {d['max_min_area_violation']:.3e} |\n"
            f"| 产量平衡最大违反 | {d['max_production_balance_violation']:.3e} |\n"
            f"| 水浇地面积最大违反 | {d['max_irrigated_area_violation']:.3e} |\n"
            f"| 整数性最大违反 | {d['max_integrality_violation']:.3e} |\n"
            f"| 连作违反数 | {d['monoculture_violation_count']} |\n"
            f"| 豆类三年最小松弛 | {d['legume_min_slack']:.3e} |\n"
            f"| 水浇地模式冲突 | {d['irrigated_mode_conflict_count']} |\n"
            f"| u超过D最大值 | {d['max_u_exceeds_D']:.3e} |\n"
            f"| u超过Q最大值 | {d['max_u_exceeds_Q']:.3e} |\n"
            f"| 利润重算差 | {d['profit_recompute_diff']:.3e} |\n"
            f"| Excel回读差 | {d['excel_roundtrip_diff']:.3e} |\n"
            f"| 可行 | {d['feasible']} |\n"
        )

    def _sens_table(rows, name):
        if not rows:
            return f"**{name}**：无数据。\n"
        out = (f"**{name}**\n\n"
               "| 参数值 | 认证Z | incumbent Z | gap | 状态 | 已认证 |\n"
               "|---|---|---|---|---|---|\n")
        for r in rows:
            out += (f"| {r['value']} | {_fmt_num(r['Z'])} | "
                    f"{_fmt_num(r.get('incumbent_Z'))} | {_gap_str(r['gap'])} | "
                    f"{r['status']} | {r.get('certified', False)} |\n")
        return out + "\n"

    txt = f"""# 问题1 建模实现报告

> 由 `scripts/run_q1.py` 自动生成。时间：{repro_extra['end_iso']}

## 1. 数据与断言

输入文件 SHA-256 已校验通过（附件1/附件2/result1_1/result1_2 模板）。数据清洗关键断言：

| 断言项 | 值 |
|---|---|
| 地块数 | {md.assertions['n_plots']} |
| 作物数 | {md.assertions['n_crops']} |
| 2023 种植记录数 | {md.assertions['n_planting_2023']} |
| 2023 统计记录数 | {md.assertions['n_stats_2023']} |
| 地块总面积(亩) | {md.assertions['total_area']:.0f} |
| 智慧大棚第一季继承组合数 | {md.assertions['inherited_smart_s1']} |
| 智慧大棚 s1 继承产量(斤) | {md.assertions['smart_s1_inherited_yield']:.0f} |
| 智慧大棚 s1 继承成本(元) | {md.assertions['smart_s1_inherited_cost']:.0f} |

## 2. 模型概要

- 决策变量：$x_{{jits}}$（面积）、$y_{{jits}}$（激活）、$r_{{jt}}$（水浇地模式）、$Q_{{its}}$（产量）、$u_{{its}}$（正常销量）。
- 目标：情形1 $\\max Z_1=\\sum p u-\\sum c x$（超额浪费）；情形2 $\\max Z_2=\\sum[p u+0.5p(Q-u)]-\\sum c x$（超额半价）。
- 主要约束：面积-激活、面积守恒、水浇地模式互斥、重茬、豆类三年轮作、产量/销量定义。
- 二阶段字典序：先最大化利润得 $Z^*$，再在 $Z\\ge(1-\\delta)Z^*$ 下最小化 $\\sum y$。
- 求解器：HiGHS（经 scipy.optimize.milp），种子 {args.seed}，MIP gap {args.mip_gap}，时间限制 {args.time_limit}s，$\\eta$={args.eta}，$\\delta$={args.delta}，$\\lambda$={args.demand_scale}。

## 3. 求解结果

| 情形 | 导出方案利润(元) | 状态 | MIP gap | 利润上界 | 字典序可行 | 已认证 |
|---|---|---|---|---|---|---|
| 情形1 | {_fmt_num(res1['Z'])} | {res1['status']} | {_gap_str(res1['mip_gap'])} | {_fmt_num(-res1['dual_bound'])} | {res1['lex_feasible']} | {res1.get('certified', False)} |
| 情形2 | {_fmt_num(res2['Z'])} | {res2['status']} | {_gap_str(res2['mip_gap'])} | {_fmt_num(-res2['dual_bound'])} | {res2['lex_feasible']} | {res2.get('certified', False)} |

> 状态 0 表示最优；若因时间限制退出，则报告 incumbent 与对偶界，不宣称最优。

## 4. 约束审计

### 情形1
{_audit_table(res1)}

### 情形2
{_audit_table(res2)}

## 5. 敏感性分析

{_sens_table(sens_eta, "η 最小种植比例")}
{_sens_table(sens_delta, "δ 利润容许损失")}
{_sens_table(sens_demand, "λ 需求倍率")}

## 6. 交付物清单

- 求解代码：`algorithms/`（io_data/preprocess/model/solve/validate/export_excel/plots）+ `scripts/run_q1.py`
- 结果工作簿：`result1_1.xlsx`、`result1_2.xlsx`
- 求解日志：`outputs/q1/logs/`
- 约束审计：`audit.csv`
- 年度统计：`yearly_stats.csv`
- 敏感性表：`sensitivity_eta.csv`、`sensitivity_delta.csv`、`sensitivity_demand.csv`
- 图表（SVG+PNG，≥9 张）：`outputs/q1/figures/`
- 复现清单：`repro.json`
- 本报告：`Q1_建模实现报告.md`

## 7. 复现命令

```
{repro_extra['command']}
```
"""
    md_path.write_text(txt, encoding="utf-8")
    generated.append(md_path)
    return md_path


def write_repro(md, raw, res1, res2, args, generated, t0, t1):
    inputs = {
        "附件1.xlsx": {"path": str(paths.F1_PATH), "sha256": _sha(paths.F1_PATH)},
        "附件2.xlsx": {"path": str(paths.F2_PATH), "sha256": _sha(paths.F2_PATH)},
        "result1_1.xlsx (template)": {"path": str(paths.TEMPLATE1_PATH),
                                       "sha256": _sha(paths.TEMPLATE1_PATH)},
        "result1_2.xlsx (template)": {"path": str(paths.TEMPLATE2_PATH),
                                       "sha256": _sha(paths.TEMPLATE2_PATH)},
    }
    outputs = []
    for p in generated:
        try:
            outputs.append({"path": str(p), "sha256": _sha(p)})
        except Exception:
            outputs.append({"path": str(p), "sha256": "unreadable"})

    def _solv(r):
        return {
            "scenario": r["scenario"], "Z": r["Z"],
            "primary_Z": r.get("primary_Z"), "status": r["status"],
            "profit_upper_bound": (-r["dual_bound"]
                                     if r["dual_bound"] == r["dual_bound"]
                                     else float("nan")),
            "mip_gap": r["mip_gap"],
            "node_count": r["node_count"], "time": r["time"],
            "primary_certified": r.get("primary_certified", False),
            "lex_feasible": r["lex_feasible"],
            "lex_certified": r.get("lex_certified", False),
            "certified": r.get("certified", False), "error": r["error"],
        }

    command = (
        f'python "{Path(__file__).resolve()}" --scenario {args.scenario} '
        f'--eta {args.eta} --delta {args.delta} --demand-scale {args.demand_scale} '
        f'--mip-gap {args.mip_gap} --time-limit {args.time_limit} '
        f'--seed {args.seed}')
    if args.sensitivity:
        command += " --sensitivity"
    if args.figures:
        command += " --figures"
    if args.reports:
        command += " --reports"

    repro = {
        "inputs": inputs,
        "versions": _dep_versions(),
        "parameters": {
            "seed": args.seed, "eta": args.eta, "delta": args.delta,
            "lambda": args.demand_scale, "mip_gap": args.mip_gap,
            "time_limit": args.time_limit, "scenario": args.scenario,
        },
        "timestamps": {
            "start": datetime.fromtimestamp(t0).isoformat(),
            "end": datetime.fromtimestamp(t1).isoformat(),
            "duration_seconds": round(t1 - t0, 2),
        },
        "solves": {"scenario_1": _solv(res1), "scenario_2": _solv(res2)},
        "outputs": outputs,
        "git_commit": _git_commit(),
        "reproduction_command": command,
    }
    repro_path = paths.REPRO_PATH
    repro_path.write_text(json.dumps(repro, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    generated.append(repro_path)
    return repro_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2024 C题 问题1 求解主程序")
    parser.add_argument("--scenario", choices=["both", "1", "2"], default="both")
    parser.add_argument("--eta", type=float, default=0.5,
                        help="最小种植面积比例 (默认0.5)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="利润容忍度δ，0.001=允许0.1%%利润损失换取更少种植激活 (默认0.001)")
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--reports", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    generated = []
    _progress(0, "启动问题1求解流程")

    # 1. dirs + input verification
    paths.ensure_dirs()
    _progress(3, "校验输入文件哈希")
    ver = io_data.verify_inputs()
    print("=== 输入校验 ===")
    all_ok = True
    for p, (sha, ok, exp) in ver.items():
        print(f"  {'OK ' if ok else 'BAD'} {Path(p).name}  sha={sha[:16]}...")
        if not ok:
            all_ok = False
    if not all_ok:
        print("输入哈希不匹配，终止。")
        sys.exit(1)

    # 2. load + preprocess
    print("\n=== 读取与预处理 ===")
    raw = io_data.load_inputs()
    md = preprocess.preprocess(raw)
    _progress(10, "数据读取与预处理完成")
    print(f"  地块 {md.assertions['n_plots']}，作物 {md.assertions['n_crops']}，"
          f"总面积 {md.assertions['total_area']:.0f} 亩，"
          f"2023 种植 {md.assertions['n_planting_2023']} 条")

    # 3. cleaned-data export
    print("\n=== 导出清洗数据 ===")
    made = export_cleaned_data(md, raw)
    for p in made:
        print(f"  -> {p}")
        generated.append(p)
    _progress(15, "清洗数据已导出")

    # 4. scenarios
    scenarios = []
    if args.scenario in ("both", "1"):
        scenarios.append(1)
    if args.scenario in ("both", "2"):
        scenarios.append(2)

    results = {}
    z_star_s1 = float("nan")
    for sc in scenarios:
        solve_start = 20 if sc == 1 else 40
        _progress(solve_start, f"正在求解情形{sc}（主目标+字典序）")
        print(f"\n=== 求解情形 {sc} ===")
        r = run_scenario(md, sc, args, f"scenario{sc}.log")
        results[sc] = r
        if r["ok"]:
            print(f"  Z{sc} = {_fmt_num(r['Z'])}  status={r['status']}  "
                  f"gap={_gap_str(r['mip_gap'])}  "
                  f"dual={_fmt_num(r['dual_bound'])}  time={r['time']:.1f}s  "
                  f"lex_feasible={r['lex_feasible']}")
            print(f"  excel_roundtrip_diff = {r['excel_roundtrip_diff']:.3e}")
            if sc == 1:
                z_star_s1 = r["primary_Z"]
            if r.get("out_path") is not None:
                generated.append(r["out_path"])
        else:
            print(f"  失败: {r['error']}")
        _progress(solve_start + 18, f"情形{sc}求解与审计结束")

    # 5. yearly stats table
    print("\n=== 年度统计表 ===")
    stats_rows = []
    for sc in scenarios:
        r = results[sc]
        if r["ok"] and r["sol"] is not None:
            ye = yearly_economics(r["sol"], md)
            stats_rows.append(ye)
            print(f"  情形{sc}: 总利润 {ye['profit'].sum():.2f} 元, "
                  f"总产量 {ye['total_Q'].sum():.0f} 斤")
    if stats_rows:
        stats_df = pd.concat(stats_rows, ignore_index=True)
    else:
        stats_df = pd.DataFrame(columns=[
            "year", "scenario", "normal_revenue", "half_revenue",
            "cost", "profit", "total_Q", "total_u", "surplus"])
    stats_df.to_csv(paths.STATS_PATH, index=False)
    generated.append(paths.STATS_PATH)
    print(f"  -> {paths.STATS_PATH}")

    # 4b. audit table
    audit_rows = []
    for sc in scenarios:
        write_audit_row(audit_rows, results[sc])
    pd.DataFrame(audit_rows).to_csv(paths.AUDIT_PATH, index=False)
    generated.append(paths.AUDIT_PATH)
    print(f"  -> {paths.AUDIT_PATH}")
    _progress(60, "年度统计与约束审计完成")

    # 6. sensitivity
    sens = ([], [], [])
    if args.sensitivity:
        _progress(62, "正在运行敏感性分析")
        print("\n=== 敏感性分析 ===")
        sens = run_sensitivity(md, z_star_s1, args)
        print(f"  -> {paths.SENS_ETA_PATH} ({len(sens[0])} 行)")
        print(f"  -> {paths.SENS_DELTA_PATH} ({len(sens[1])} 行)")
        print(f"  -> {paths.SENS_DEMAND_PATH} ({len(sens[2])} 行)")
        generated += [paths.SENS_ETA_PATH, paths.SENS_DELTA_PATH,
                      paths.SENS_DEMAND_PATH]
        _progress(78, "敏感性分析完成")

    # 7. figures
    if args.figures:
        _progress(80, "正在生成三类图表")
        print("\n=== 生成图表 ===")
        sol1 = results.get(1, {}).get("sol") if results.get(1, {}).get("ok") else None
        sol2 = results.get(2, {}).get("sol") if results.get(2, {}).get("ok") else None
        audit1 = results.get(1, {}).get("audit")
        audit2 = results.get(2, {}).get("audit")
        print("  运行情形1定时子求解以构建上下界历史...")
        bh = build_bound_history(md, args)
        print(f"  bound_history 点数 = {len(bh)}")
        fps = plots.generate_figures(
            md, sol1=sol1, sol2=sol2, output_dir=paths.FIG_DIR,
            audit1=audit1, audit2=audit2, bound_history=bh,
            dpi=300, seed=args.seed)
        pngs = [p for p in fps if p.suffix == ".png"]
        svgs = [p for p in fps if p.suffix == ".svg"]
        print(f"  生成 {len(pngs)} 张 PNG, {len(svgs)} 张 SVG")
        generated += fps
        _progress(93, "图表生成完成")

    # 8. reports + repro
    if args.reports:
        _progress(95, "正在生成报告与复现清单")
        print("\n=== 生成报告与复现清单 ===")
        t1 = time.time()
        command = (
            f'python "{Path(__file__).resolve()}" --scenario {args.scenario} '
            f'--eta {args.eta} --delta {args.delta} '
            f'--demand-scale {args.demand_scale} --mip-gap {args.mip_gap} '
            f'--time-limit {args.time_limit} --seed {args.seed}')
        if args.sensitivity:
            command += " --sensitivity"
        if args.figures:
            command += " --figures"
        if args.reports:
            command += " --reports"
        repro_extra = {"end_iso": datetime.fromtimestamp(t1).isoformat(),
                       "command": command}
        _dummy = {"scenario": 0, "Z": float("nan"), "status": None,
                  "mip_gap": float("nan"), "dual_bound": float("nan"),
                  "node_count": 0, "time": 0.0, "lex_feasible": False,
                  "audit": None, "error": "not run"}
        write_report(md, results.get(1, _dummy),
                     results.get(2, _dummy), sens, args,
                     generated, repro_extra)
        # ensure res1/res2 always exist for repro
        r1 = results.get(1, {"scenario": 1, "Z": float("nan"),
                             "status": None, "mip_gap": float("nan"),
                             "dual_bound": float("nan"), "node_count": 0,
                             "time": 0.0, "lex_feasible": False,
                             "error": "scenario 1 not run"})
        r2 = results.get(2, {"scenario": 2, "Z": float("nan"),
                             "status": None, "mip_gap": float("nan"),
                             "dual_bound": float("nan"), "node_count": 0,
                             "time": 0.0, "lex_feasible": False,
                             "error": "scenario 2 not run"})
        rp = write_repro(md, raw, r1, r2, args, generated, t0, t1)
        print(f"  -> {paths.DOC_DIR / 'Q1_建模实现报告.md'}")
        print(f"  -> {rp}")
        _progress(99, "报告与复现清单完成")

    t1 = time.time()

    # 9. final summary
    r1 = results.get(1)
    r2 = results.get(2)

    def _viol(r):
        if r is None or r.get("audit") is None:
            return "未知"
        a = r["audit"].to_dict()
        v = []
        if a["max_area_conservation_violation"] > 1e-6:
            v.append("面积守恒")
        if a["max_unsuitable_area"] > 1e-6:
            v.append("非适种")
        if a["max_upper_link_violation"] > 1e-6:
            v.append("x-y联动")
        if a["max_min_area_violation"] > 1e-6:
            v.append("最小面积")
        if a["max_production_balance_violation"] > 1e-6:
            v.append("产量平衡")
        if a["max_irrigated_area_violation"] > 1e-6:
            v.append("水浇地面积")
        if a["max_integrality_violation"] > 1e-6:
            v.append("整数性")
        if a["monoculture_violation_count"] > 0:
            v.append("连作")
        if a["irrigated_mode_conflict_count"] > 0:
            v.append("水浇地模式")
        if a["max_u_exceeds_D"] > 1e-6:
            v.append("u>D")
        if a["max_u_exceeds_Q"] > 1e-6:
            v.append("u>Q")
        if a["legume_min_slack"] < -1e-6:
            v.append("豆类轮作")
        if a["profit_recompute_diff"] > 1e-6:
            v.append("利润重算")
        if a["excel_roundtrip_diff"] >= 1e-4:
            v.append("Excel回读")
        return "无" if not v else "、".join(v)

    local_ready = all(r is not None and r.get("ok") and r.get("certified")
                      for r in results.values()) and \
        len(results) == len(scenarios)

    print("\n" + "=" * 60)
    print("              Q1 求解完成 — 最终摘要")
    print("=" * 60)
    print("P1 (独立门禁)    : NOT_RECORDED（请先运行 scripts/p1_test.py）")
    print(f"本地交付就绪检查 : {'PASS' if local_ready else 'FAIL'}")
    print("P2 (独立门禁)    : NOT_RUN")
    if r1 is not None:
        print(f"情形1: Z1 = {_fmt_num(r1['Z'])}  status={r1['status']}  "
              f"gap={_gap_str(r1['mip_gap'])}  "
              f"dual={_fmt_num(r1['dual_bound'])}  time={r1['time']:.1f}s")
    if r2 is not None:
        print(f"情形2: Z2 = {_fmt_num(r2['Z'])}  status={r2['status']}  "
              f"gap={_gap_str(r2['mip_gap'])}  "
              f"dual={_fmt_num(r2['dual_bound'])}  time={r2['time']:.1f}s")
    if r1 is not None and r2 is not None and r1.get("ok") and r2.get("ok"):
        print(f"Z2 >= Z1 ? {r2['Z'] >= r1['Z'] - 1e-6}  "
              f"(Z1={_fmt_num(r1['Z'])}, Z2={_fmt_num(r2['Z'])})")
    print(f"约束违反: 情形1={_viol(r1) if r1 else '未知'}; "
          f"情形2={_viol(r2) if r2 else '未知'}")
    print("-" * 60)
    print("生成文件:")
    for p in generated:
        print(f"  {Path(p).resolve()}")
    print("-" * 60)
    print(f"总耗时: {t1 - t0:.1f}s")
    print("=" * 60)
    _progress(100, "流程结束" if local_ready else "流程结束，但未达到交付门槛")
    sys.exit(0 if local_ready else 2)


if __name__ == "__main__":
    main()
