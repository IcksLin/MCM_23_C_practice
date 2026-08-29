# -*- coding: utf-8 -*-
"""Three-stage lexicographic solver for Q2 (AGENT.md section 8).

Stage 1: maximize Z_lambda (risk objective)
Stage 2: maximize E[Pi] subject to Z_lambda >= Z* - eps
Stage 3: minimize sum(y) subject to E[Pi] >= E* - eps

Uses scipy.optimize.milp (HiGHS backend).
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

from .preprocess import ModelData
from .scenario_reduction import ReducedScenarioSet
from .model import StochModel, build_q2_model


@dataclass
class SolveResult:
    is_feasible: bool
    status: int
    message: str
    fun: float            # objective value (minimization form)
    dual_bound: float
    mip_gap: float
    nodes: int
    time: float
    max_violation: float
    x: np.ndarray         # variable values
    # solver status classification (区分4类状态)
    has_incumbent: bool = False       # solver returned a valid x
    solver_status: str = "unknown"   # optimal | feasible_not_proven | time_limit_no_feasible | infeasible


def _solve_milp(model: StochModel, time_limit: float = 600.0,
                mip_gap: float = 0.001, disp: bool = False) -> SolveResult:
    """Run scipy.optimize.milp on the StochModel."""
    constraints = []
    if model.A_ub.shape[0] > 0:
        constraints.append(LinearConstraint(
            model.A_ub, ub=model.b_ub))
    if model.A_eq.shape[0] > 0:
        constraints.append(LinearConstraint(
            model.A_eq, lb=model.b_eq, ub=model.b_eq))

    bounds = Bounds(lb=model.lb, ub=model.ub)
    started = time.perf_counter()
    res = milp(
        c=model.c, constraints=constraints, bounds=bounds,
        integrality=model.integrality,
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap,
                 "disp": disp},
    )

    raw_x = getattr(res, "x", None)
    if raw_x is not None:
        x = np.asarray(raw_x, dtype=float)
    else:
        x = np.full(model.n, np.nan)

    _fun_raw = getattr(res, "fun", None)
    fun = float(_fun_raw) if _fun_raw is not None else np.nan
    # HiGHS status: 0=optimal, 1=time limit, 2=infeasible, etc.
    status = int(getattr(res, "status", -1))
    message = str(getattr(res, "message", ""))
    success = bool(getattr(res, "success", False))

    max_violation = np.inf
    if raw_x is not None and np.all(np.isfinite(x)):
        violations = [0.0]
        if model.A_ub.shape[0]:
            violations.append(float(np.max(np.maximum(model.A_ub @ x - model.b_ub, 0.0))))
        if model.A_eq.shape[0]:
            violations.append(float(np.max(np.abs(model.A_eq @ x - model.b_eq))))
        violations.append(float(np.max(np.maximum(model.lb - x, 0.0))))
        violations.append(float(np.max(np.maximum(x - model.ub, 0.0))))
        max_violation = max(violations)
    is_feasible = bool(raw_x is not None and np.all(np.isfinite(x))
                       and max_violation <= 1e-5)
    has_incumbent = bool(raw_x is not None and np.all(np.isfinite(x)))

    # classify solver status into 4 categories
    # HiGHS status: 0=optimal, 1=time limit, 2=infeasible, 3=unbounded
    if status == 0:
        solver_status = "optimal"
    elif status == 2:
        solver_status = "infeasible"
    elif status == 1:
        # time limit reached: did we get an incumbent?
        if has_incumbent:
            solver_status = "feasible_not_proven"
        else:
            solver_status = "time_limit_no_feasible"
    else:
        solver_status = "unknown"

    # extract HiGHS internals (scipy.milp uses mip_dual_bound / mip_node_count)
    _dual_raw = getattr(res, "mip_dual_bound", None)
    if _dual_raw is None:
        _dual_raw = getattr(res, "lower_bounds", None)
    if _dual_raw is None:
        _dual_raw = getattr(res, "upper_bounds", None)
    dual_bound = float(_dual_raw) if _dual_raw is not None else np.nan
    _gap_raw = getattr(res, "mip_gap", None)
    if _gap_raw is None:
        # scipy may not expose mip_gap; estimate from dual/primal if available
        _gap_raw = getattr(res, "mip_rel_gap", None)
    mip_gap_val = float(_gap_raw) if _gap_raw is not None else np.nan
    _nodes_raw = getattr(res, "mip_node_count", None)
    nodes = int(_nodes_raw) if _nodes_raw is not None else 0
    time_s = time.perf_counter() - started

    return SolveResult(
        is_feasible=is_feasible, status=status, message=message,
        fun=fun, dual_bound=dual_bound, mip_gap=mip_gap_val, nodes=nodes,
        time=time_s, max_violation=max_violation, x=x,
        has_incumbent=has_incumbent, solver_status=solver_status,
    )


def solve_risk_stage(data: ModelData, scenarios: ReducedScenarioSet,
                     beta: float, risk_lambda: float, eta: float,
                     time_limit: float, mip_gap: float,
                     disp: bool = False) -> tuple:
    """Stage 1: maximize Z_lambda."""
    model = build_q2_model(data, scenarios, beta=beta,
                           risk_lambda=risk_lambda, eta=eta, stage="risk")
    res = _solve_milp(model, time_limit, mip_gap, disp)
    z_star = -res.fun if res.is_feasible else None
    return model, res, z_star


def solve_expected_stage(data: ModelData, scenarios: ReducedScenarioSet,
                          beta: float, risk_lambda: float, eta: float,
                          z_star: float, eps_z: float = 1e-6,
                          time_limit: float = 600.0, mip_gap: float = 0.001,
                          disp: bool = False) -> tuple:
    """Stage 2: maximize E[Pi] subject to Z_lambda >= Z* - eps."""
    model = build_q2_model(data, scenarios, beta=beta,
                           risk_lambda=risk_lambda, eta=eta, stage="expected",
                           z_star=z_star, eps_z=eps_z)
    res = _solve_milp(model, time_limit, mip_gap, disp)
    e_star = -res.fun if res.is_feasible else None
    return model, res, e_star


def solve_fragment_stage(data: ModelData, scenarios: ReducedScenarioSet,
                           beta: float, risk_lambda: float, eta: float,
                           z_star: float, e_star: float,
                         eps_z: float | None = None,
                         eps_e: float | None = None,
                           time_limit: float = 600.0, mip_gap: float = 0.001,
                           disp: bool = False) -> tuple:
    """Stage 3: minimize sum(y) subject to E[Pi] >= E* - eps."""
    model = build_q2_model(data, scenarios, beta=beta,
                           risk_lambda=risk_lambda, eta=eta, stage="fragment",
                           z_star=z_star, e_star=e_star,
                           eps_z=eps_z, eps_e=eps_e)
    res = _solve_milp(model, time_limit, mip_gap, disp)
    return model, res


def solve_lexicographic(data: ModelData, scenarios: ReducedScenarioSet,
                        beta: float, risk_lambda: float, eta: float,
                        time_limit: float = 600.0, mip_gap: float = 0.001,
                        eps_z: float | None = None,
                        eps_e: float | None = None,
                        disp: bool = False) -> dict:
    """Run all three lexicographic stages.

    Returns dict with keys: z_star, e_star, n_activations, model, result, solution.
    """
    # Stage 1: risk
    m1, r1, z_star = solve_risk_stage(
        data, scenarios, beta, risk_lambda, eta, time_limit, mip_gap, disp)
    if z_star is None:
        return {"z_star": None, "e_star": None, "n_activations": None,
                "model": m1, "result": r1, "solution": None,
                "stage1_feasible": False}

    # Monetary lexicographic tolerances must scale with the objective.  A
    # sub-cent tolerance on a 10^7-yuan objective is numerically meaningless.
    eps_z = max(1.0, 1e-6 * abs(z_star)) if eps_z is None else float(eps_z)

    # Stage 2: expected profit
    m2, r2, e_star = solve_expected_stage(
        data, scenarios, beta, risk_lambda, eta, z_star, eps_z,
        time_limit, mip_gap, disp)
    if e_star is None:
        # Fallback: use stage 1 solution (satisfies Z_lambda >= z_star trivially)
        sol = extract_solution(r1, m1, data) if r1.is_feasible else None
        n_act = sol["n_activations"] if sol else None
        return {"z_star": z_star, "e_star": None, "n_activations": n_act,
                "model": m1, "result": r1, "solution": sol,
                "stage1_feasible": True, "stage2_feasible": False,
                "result1": r1, "result2": r2, "fallback": True,
                "final_stage": 1, "lex_complete": False,
                "eps_z": eps_z, "eps_e": None}

    eps_e = max(1.0, 1e-6 * abs(e_star)) if eps_e is None else float(eps_e)

    # Stage 3: fragmentation (minimize activations)
    m3, r3 = solve_fragment_stage(
        data, scenarios, beta, risk_lambda, eta, z_star, e_star,
        eps_z, eps_e, time_limit, mip_gap, disp)

    if r3.is_feasible:
        sol = extract_solution(r3, m3, data)
        final_model, final_result, final_stage = m3, r3, 3
    else:
        # Fallback: use stage 2 (or stage 1) solution
        if r2.is_feasible:
            sol = extract_solution(r2, m2, data)
            final_model, final_result, final_stage = m2, r2, 2
        else:
            sol = extract_solution(r1, m1, data) if r1.is_feasible else None
            final_model, final_result, final_stage = m1, r1, 1
    n_act = sol["n_activations"] if sol else None

    return {
        "z_star": z_star, "e_star": e_star, "n_activations": n_act,
        "model": final_model, "result": final_result, "solution": sol,
        "stage1_feasible": True, "stage2_feasible": True,
        "stage3_feasible": r3.is_feasible,
        "result1": r1, "result2": r2, "result3": r3,
        "final_stage": final_stage, "lex_complete": final_stage == 3,
        "eps_z": eps_z, "eps_e": eps_e,
    }


def extract_solution(res: SolveResult, model: StochModel,
                     data: ModelData) -> dict:
    """Extract variable values from solve result.

    Returns dict with:
      x: dict (j,i,t,s) -> area
      y: dict (j,i,t,s) -> 0/1
      r: dict (j,t) -> 0/1
      n_activations: int
    """
    x = {}
    y = {}
    r = {}
    raw_y = {}
    raw_r = {}
    Q = {}
    u = {}
    for (j, i, t, s), k in model.x_idx.items():
        val = res.x[k]
        if val > 1e-8:
            x[(j, i, t, s)] = float(val)
        else:
            x[(j, i, t, s)] = 0.0
    for (j, i, t, s), k in model.y_idx.items():
        val = res.x[k]
        raw_y[(j, i, t, s)] = float(val)
        y[(j, i, t, s)] = 1 if val > 0.5 else 0
    for (j, t), k in model.r_idx.items():
        val = res.x[k]
        raw_r[(j, t)] = float(val)
        r[(j, t)] = 1 if val > 0.5 else 0
    for key, k in model.Q_idx.items():
        Q[key] = float(res.x[k])
    for key, k in model.u_idx.items():
        u[key] = float(res.x[k])

    n_act = sum(y.values())
    return {"x": x, "y": y, "r": r, "raw_y": raw_y, "raw_r": raw_r,
            "Q": Q, "u": u, "n_activations": n_act}
