# -*- coding: utf-8 -*-
"""Solving layer: HiGHS via scipy.optimize.milp.

  solve(m, ...)                 -> SolveResult (raw)
  solve_primary(...)            -> (MILPModel, SolveResult, z_star)
  solve_lexicographic(...)      -> (MILPModel, SolveResult)  (min F s.t. Z>= (1-d)*Z*)
  extract_solution(res, m, data)-> dict of pandas DataFrames

AGENT.md section 6: MIPGap <= 0.001, seed 2024, presolve on, time limit 600 s.
If time-limited, incumbent + best bound + gap are reported honestly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

from .model import MILPModel, build_model
from .preprocess import ModelData


@dataclass
class SolveResult:
    x: np.ndarray
    fun: float
    status: int                # 0 optimal, 1 time-limit, 2 infeasible, ...
    message: str
    mip_gap: float
    dual_bound: float
    node_count: int
    time: float
    is_feasible: bool


def solve(m: MILPModel, time_limit: float = 600.0, mip_gap: float = 0.001,
          seed: int = 2024, disp: bool = True, log_path: Path = None) -> SolveResult:
    """Call scipy.optimize.milp (HiGHS) on the assembled model."""
    constraints = []
    if m.A_ub.shape[0] > 0:
        constraints.append(LinearConstraint(m.A_ub, -np.inf, m.b_ub))
    if m.A_eq.shape[0] > 0:
        constraints.append(LinearConstraint(m.A_eq, m.b_eq, m.b_eq))
    options = {
        "disp": disp,
        "time_limit": time_limit,
        "mip_rel_gap": mip_gap,
        "presolve": True,
    }
    # scipy.optimize.milp does not expose HiGHS random_seed as a supported
    # option.  HiGHS' default MIP execution is deterministic for a fixed
    # model; keep `seed` in provenance without passing an unknown option.

    bounds = Bounds(m.lb, m.ub)
    t0 = time.time()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # scipy.optimize.milp does not stream a log file directly; we capture
        # via the returned message and stderr-like output if disp=True.
    res = milp(m.c, constraints=constraints or None,
               integrality=m.integrality, bounds=bounds, options=options)
    elapsed = time.time() - t0

    x = np.asarray(res.x) if res.x is not None else np.full(m.n, np.nan)
    fun = float(res.fun) if res.fun is not None else float("nan")
    status = int(getattr(res, "status", 0))
    message = str(getattr(res, "message", ""))
    gap_raw = getattr(res, "mip_gap", None)
    dual_raw = getattr(res, "mip_dual_bound", None)
    gap = float(gap_raw) if gap_raw is not None else float("nan")
    dual = float(dual_raw) if dual_raw is not None else float("nan")
    nodes = int(getattr(res, "mip_node_count", 0) or 0)
    has_incumbent = res.x is not None and np.all(np.isfinite(x)) and np.isfinite(fun)
    feas = bool(getattr(res, "success", False)) or has_incumbent

    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"=== solve scenario={m.scenario} stage={m.stage} "
                         f"eta={m.eta} delta={m.delta} scale={m.demand_scale} ===\n")
                fh.write(f"status={status} message={message}\n")
                fh.write(f"fun={fun} dual_bound={dual} mip_gap={gap} "
                         f"nodes={nodes} time={elapsed:.1f}s\n")
                fh.write(f"n_vars={m.n} n_ub={m.A_ub.shape[0]} "
                         f"n_eq={m.A_eq.shape[0]}\n\n")
        except Exception:
            pass

    return SolveResult(x=x, fun=fun, status=status, message=message,
                       mip_gap=gap, dual_bound=dual, node_count=nodes,
                       time=elapsed, is_feasible=feas)


def solve_primary(data: ModelData, scenario: int, eta: float = 0.5,
                  demand_scale: float = 1.0, time_limit: float = 600.0,
                  mip_gap: float = 0.001, seed: int = 2024,
                  disp: bool = True, log_path: Path = None):
    m = build_model(data, scenario=scenario, eta=eta,
                    demand_scale=demand_scale, stage="primary")
    res = solve(m, time_limit=time_limit, mip_gap=mip_gap, seed=seed,
                disp=disp, log_path=log_path)
    # primary objective is minimize -Z  ->  Z = -fun
    z_star = -res.fun if res.is_feasible else float("nan")
    return m, res, z_star


def solve_lexicographic(data: ModelData, scenario: int, z_star: float,
                        eta: float = 0.5, demand_scale: float = 1.0,
                        delta: float = 0.0, time_limit: float = 600.0,
                        mip_gap: float = 0.001, seed: int = 2024,
                        disp: bool = True, log_path: Path = None):
    m = build_model(data, scenario=scenario, eta=eta,
                    demand_scale=demand_scale, stage="lex",
                    z_star=z_star, delta=delta)
    res = solve(m, time_limit=time_limit, mip_gap=mip_gap, seed=seed,
                disp=disp, log_path=log_path)
    return m, res


# ---------------------------------------------------------------------------
# solution extraction
# ---------------------------------------------------------------------------

def extract_solution(res: SolveResult, m: MILPModel, data: ModelData) -> dict:
    """Return DataFrames / dicts for x, y, r, Q, u plus a tidy long table."""
    x = res.x
    # x area table: (plot, year, season, crop) -> area
    rows = []
    for (j, i, t, s), k in m.x_idx.items():
        area = float(x[k]) if k < len(x) and not np.isnan(x[k]) else 0.0
        if area > 1e-9:
            rows.append({
                "plot_idx": j, "plot": data.plot_names[j],
                "plot_type": data.plot_type[j],
                "crop_code": i, "crop": data.crop_names.get(i, ""),
                "year": t, "season": s, "area": area,
            })
    x_long = pd.DataFrame(rows, columns=[
        "plot_idx", "plot", "plot_type", "crop_code", "crop",
        "year", "season", "area"])

    y_long = pd.DataFrame([
        {"plot_idx": j, "plot": data.plot_names[j], "crop_code": i,
         "crop": data.crop_names.get(i, ""), "year": t, "season": s,
         "y": int(round(float(x[k])))}
        for (j, i, t, s), k in m.y_idx.items()
        if k < len(x) and float(x[k]) > 0.5
    ])

    r_long = pd.DataFrame([
        {"plot_idx": j, "plot": data.plot_names[j], "year": t,
         "r": int(round(float(x[k])))}
        for (j, t), k in m.r_idx.items()
        if k < len(x) and float(x[k]) > 0.5
    ])

    Q_long = pd.DataFrame([
        {"crop_code": i, "crop": data.crop_names.get(i, ""),
         "year": t, "season": s, "Q": float(x[k])}
        for (i, t, s), k in m.Q_idx.items()
        if k < len(x) and float(x[k]) > 1e-6
    ])

    u_long = pd.DataFrame([
        {"crop_code": i, "crop": data.crop_names.get(i, ""),
         "year": t, "season": s, "u": float(x[k])}
        for (i, t, s), k in m.u_idx.items()
        if k < len(x) and float(x[k]) > 1e-6
    ])

    # profit recomputation (independent of solver)
    profit = 0.0
    if res.is_feasible:
        # revenue
        for (i, t, s), k in m.u_idx.items():
            p = data.p.get((i, s), 0.0)
            u_val = float(x[k])
            if m.scenario == 2:
                qk = m.Q_idx[(i, t, s)]
                Q_val = float(x[qk])
                profit += p * u_val + 0.5 * p * (Q_val - u_val)
            else:
                profit += p * u_val
        # cost
        for (j, i, t, s), k in m.x_idx.items():
            profit -= data.c[(j, i, s)] * float(x[k])

    integrality_values = [float(x[k]) for k in list(m.y_idx.values()) + list(m.r_idx.values())
                          if k < len(x) and np.isfinite(x[k])]
    max_integrality_violation = max(
        (abs(v - round(v)) for v in integrality_values), default=0.0)
    achieved_profit = profit if res.is_feasible else float("nan")

    return {
        "x": x_long, "y": y_long, "r": r_long, "Q": Q_long, "u": u_long,
        "profit_recomputed": profit,
        # `objective` always uses the economic-profit convention.  The raw
        # solver objective differs in the lexicographic stage (it is sum y).
        "objective": achieved_profit if m.stage == "lex" else (
            -res.fun if res.is_feasible else float("nan")),
        "solver_objective": res.fun,
        "activation_count": int(round(res.fun)) if (
            m.stage == "lex" and np.isfinite(res.fun)) else len(y_long),
        "max_integrality_violation": max_integrality_violation,
        "mip_gap": res.mip_gap, "dual_bound": res.dual_bound,
        "node_count": res.node_count, "time": res.time,
        "status": res.status, "message": res.message,
        "scenario": m.scenario, "eta": m.eta,
        "demand_scale": m.demand_scale, "stage": m.stage, "delta": m.delta,
    }
