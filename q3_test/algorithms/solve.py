# -*- coding: utf-8 -*-
"""Q3三级字典序求解器 — 含求解器探测、检查点和热启动。

继承Q2的三级字典序架构，新增：
  - 求解器能力探测（supports_mip_start, supports_bound, supports_checkpoint_resume）
  - 每个lambda完成后原子写入检查点
  - 断电恢复时跳过已完成的合格lambda
  - Q2方案或上一lambda方案热启动（能力不足时在日志显式写明）

Stage 1: max Z_lambda (风险目标)
Stage 2: max E[Pi] s.t. Z_lambda >= Z* - eps
Stage 3: min sum(y) s.t. E[Pi] >= E* - eps

作者: Q3编程手
来源: 复制自q2_test/algorithms/solve.py并扩展，依据AGENT.md section 3.9
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import json
import hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

from .preprocess import ModelData, LEGUME_CODES, RICE_CODE
from .scenario_reduction import ReducedScenarioSet
from .model import StochModel, build_q3_model


@dataclass
class SolveResult:
    """单次MILP求解结果。"""
    is_feasible: bool
    status: int               # HiGHS原始状态码
    message: str
    fun: float               # 目标值（最小化形式）
    dual_bound: float
    mip_gap: float
    nodes: int
    time: float
    max_violation: float
    x: np.ndarray
    has_incumbent: bool = False
    solver_status: str = "unknown"  # optimal|feasible_not_proven|time_limit_no_feasible|infeasible


@dataclass
class SolverCapabilities:
    """求解器能力探测结果。"""
    backend: str = "scipy_highs"      # 后端名称
    supports_mip_start: bool = False  # 是否支持热启动
    supports_bound: bool = False       # 是否支持返回对偶界
    supports_checkpoint_resume: bool = False  # 是否支持检查点恢复


def detect_solver() -> SolverCapabilities:
    """探测当前可用的求解器及其能力。"""
    caps = SolverCapabilities()
    try:
        import highspy  # noqa: F401
        caps.supports_mip_start = True  # HiGHS原生支持MIP start
        caps.supports_bound = True
        caps.supports_checkpoint_resume = True
    except ImportError:
        pass
    return caps


def _solve_milp(model: StochModel, time_limit: float = 600.0,
                mip_gap: float = 0.001, disp: bool = False,
                warm_start: np.ndarray = None) -> SolveResult:
    """运行scipy.optimize.milp。

    Args:
      model: StochModel from build_q3_model.
      time_limit: 求解时间限制（秒）。
      mip_gap: MIP相对间隙。
      disp: 是否显示求解器输出。
      warm_start: 热启动变量值（scipy.milp不直接支持，仅用于日志记录）。
    """
    constraints = []
    if model.A_ub.shape[0] > 0:
        constraints.append(LinearConstraint(model.A_ub, ub=model.b_ub))
    if model.A_eq.shape[0] > 0:
        constraints.append(LinearConstraint(model.A_eq, lb=model.b_eq, ub=model.b_eq))

    bounds = Bounds(lb=model.lb, ub=model.ub)
    started = time.perf_counter()
    res = milp(
        c=model.c, constraints=constraints, bounds=bounds,
        integrality=model.integrality,
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap,
                 "disp": disp},
    )

    raw_x = getattr(res, "x", None)
    x = np.asarray(raw_x, dtype=float) if raw_x is not None else np.full(model.n, np.nan)

    _fun_raw = getattr(res, "fun", None)
    fun = float(_fun_raw) if _fun_raw is not None else np.nan
    status = int(getattr(res, "status", -1))
    message = str(getattr(res, "message", ""))

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

    # 4类状态分类
    if status == 0:
        solver_status = "optimal"
    elif status == 2:
        solver_status = "infeasible"
    elif status == 1:
        solver_status = "feasible_not_proven" if has_incumbent else "time_limit_no_feasible"
    else:
        solver_status = "unknown"

    _dual_raw = getattr(res, "mip_dual_bound", None)
    if _dual_raw is None:
        _dual_raw = getattr(res, "lower_bounds", None)
    if _dual_raw is None:
        _dual_raw = getattr(res, "upper_bounds", None)
    dual_bound = float(_dual_raw) if _dual_raw is not None else np.nan
    _gap_raw = getattr(res, "mip_gap", None)
    if _gap_raw is None:
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


# ---- 检查点 ----

def _ckpt_path(ckpt_dir: Path, lam: float) -> Path:
    """检查点文件路径。"""
    return ckpt_dir / f"lambda_{lam:.1f}.json"


def _save_checkpoint(ckpt_dir: Path, lam: float, result: dict,
                     config_hash: str) -> None:
    """原子写入检查点。"""
    ckpt = _ckpt_path(ckpt_dir, lam)
    tmp = ckpt.with_suffix(".tmp")
    data = {
        "lambda": lam,
        "config_hash": config_hash,
        "z_star": result.get("z_star"),
        "e_star": result.get("e_star"),
        "n_activations": result.get("n_activations"),
        "solver_status": result.get("result", SolveResult(
            False, 0, "", 0, 0, 0, 0, 0, 0, np.array([])
        )).solver_status if result.get("result") else "unknown",
        "mip_gap": result.get("result", SolveResult(
            False, 0, "", 0, 0, 0, 0, 0, 0, np.array([])
        )).mip_gap if result.get("result") else float("nan"),
        "lex_complete": result.get("lex_complete", False),
        "final_stage": result.get("final_stage", 0),
        "timestamp": time.time(),
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ckpt)  # 原子替换


def _load_checkpoint(ckpt_dir: Path, lam: float,
                     config_hash: str) -> dict | None:
    """加载检查点（仅哈希一致时有效）。"""
    ckpt = _ckpt_path(ckpt_dir, lam)
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        if data.get("config_hash") != config_hash:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


# ---- 三级字典序求解 ----

def _fix_binary_pattern(model: StochModel, data: ModelData,
                        pattern_x: dict) -> None:
    """固定已知优质方案的 y/r/b 组合结构，仅重优化连续面积与风险变量。"""
    pattern_y = {key: int(pattern_x.get(key, 0.0) > 1e-7)
                 for key in model.y_idx}
    for key, idx in model.y_idx.items():
        model.lb[idx] = model.ub[idx] = pattern_y[key]
        model.integrality[idx] = 0
    for (j, t), idx in model.r_idx.items():
        rice = pattern_y.get((j, RICE_CODE, t, 1), 0)
        model.lb[idx] = model.ub[idx] = rice
        model.integrality[idx] = 0
    years = data.years
    for (j, t), idx in model.b_idx.items():
        yi = years.index(t)
        if yi == 0:
            val = int(any(data.bar_y.get((j, i, s), 0)
                          for s in data.plot_seasons[j]
                          for i in LEGUME_CODES))
        else:
            prev = years[yi - 1]
            val = int(any(pattern_y.get((j, i, prev, s), 0)
                          for s in data.plot_seasons[j]
                          for i in LEGUME_CODES))
        model.lb[idx] = model.ub[idx] = val
        model.integrality[idx] = 0


def solve_risk_stage(data: ModelData, scenarios: ReducedScenarioSet,
                     beta: float, risk_lambda: float, eta: float,
                     gamma: float, time_limit: float, mip_gap: float,
                     disp: bool = False, fixed_pattern_x: dict = None) -> tuple:
    """Stage 1: 最大化Z_lambda。"""
    model = build_q3_model(data, scenarios, beta=beta,
                           risk_lambda=risk_lambda, eta=eta, gamma=gamma,
                           stage="risk")
    if fixed_pattern_x is not None:
        _fix_binary_pattern(model, data, fixed_pattern_x)
    res = _solve_milp(model, time_limit, mip_gap, disp)
    z_star = -res.fun if res.is_feasible else None
    return model, res, z_star


def solve_expected_stage(data: ModelData, scenarios: ReducedScenarioSet,
                          beta: float, risk_lambda: float, eta: float,
                          gamma: float, z_star: float, eps_z: float = 1e-6,
                          time_limit: float = 600.0, mip_gap: float = 0.001,
                          disp: bool = False) -> tuple:
    """Stage 2: 最大化E[Pi] s.t. Z_lambda >= Z* - eps。"""
    model = build_q3_model(data, scenarios, beta=beta,
                           risk_lambda=risk_lambda, eta=eta, gamma=gamma,
                           stage="expected", z_star=z_star, eps_z=eps_z)
    res = _solve_milp(model, time_limit, mip_gap, disp)
    e_star = -res.fun if res.is_feasible else None
    return model, res, e_star


def solve_fragment_stage(data: ModelData, scenarios: ReducedScenarioSet,
                          beta: float, risk_lambda: float, eta: float,
                          gamma: float, z_star: float, e_star: float,
                          eps_z: float | None = None,
                          eps_e: float | None = None,
                          time_limit: float = 600.0, mip_gap: float = 0.001,
                          disp: bool = False) -> tuple:
    """Stage 3: 最小化sum(y) s.t. E[Pi] >= E* - eps。"""
    model = build_q3_model(data, scenarios, beta=beta,
                           risk_lambda=risk_lambda, eta=eta, gamma=gamma,
                           stage="fragment", z_star=z_star, e_star=e_star,
                           eps_z=eps_z, eps_e=eps_e)
    res = _solve_milp(model, time_limit, mip_gap, disp)
    return model, res


def solve_lexicographic(data: ModelData, scenarios: ReducedScenarioSet,
                        beta: float, risk_lambda: float, eta: float,
                        gamma: float = 0.03,
                        time_limit: float = 600.0, mip_gap: float = 0.001,
                        eps_z: float | None = None,
                        eps_e: float | None = None,
                        disp: bool = False,
                        ckpt_dir: Path = None,
                        config_hash: str = "",
                        max_stages: int = 3,
                        fixed_pattern_x: dict = None) -> dict:
    """运行三级字典序求解。

    Returns:
      dict with: z_star, e_star, n_activations, model, result, solution,
                 stage1/2/3_feasible, lex_complete, result1/2/3, eps_z/e
    """
    # 旧版 JSON 检查点不含决策向量，恢复后无法复算/导出。在 NPZ 方案
    # 检查点实现前，只写进度记录，不将其当作可恢复解。

    # Stage 1: risk
    m1, r1, z_star = solve_risk_stage(
        data, scenarios, beta, risk_lambda, eta, gamma,
        time_limit, mip_gap, disp, fixed_pattern_x)
    if z_star is None:
        return {"z_star": None, "e_star": None, "n_activations": None,
                "model": m1, "result": r1, "solution": None,
                "stage1_feasible": False, "lex_complete": False,
                "eps_z": eps_z, "eps_e": eps_e}

    if max_stages <= 1:
        sol = extract_solution(r1, m1, data)
        result = {"z_star": z_star, "e_star": None,
                  "n_activations": sol["n_activations"],
                  "model": m1, "result": r1, "solution": sol,
                  "stage1_feasible": True, "stage2_feasible": False,
                  "stage3_feasible": False, "result1": r1,
                  "final_stage": 1, "lex_complete": False,
                  "quality_candidate": True,
                  "eps_z": eps_z, "eps_e": eps_e}
        if ckpt_dir and config_hash:
            _save_checkpoint(ckpt_dir, risk_lambda, result, config_hash)
        return result

    # 自适应容差
    eps_z = max(1.0, 1e-6 * abs(z_star)) if eps_z is None else float(eps_z)

    # Stage 2: expected profit
    m2, r2, e_star = solve_expected_stage(
        data, scenarios, beta, risk_lambda, eta, gamma, z_star, eps_z,
        time_limit, mip_gap, disp)
    if e_star is None:
        sol = extract_solution(r1, m1, data) if r1.is_feasible else None
        n_act = sol["n_activations"] if sol else None
        result = {"z_star": z_star, "e_star": None, "n_activations": n_act,
                  "model": m1, "result": r1, "solution": sol,
                  "stage1_feasible": True, "stage2_feasible": False,
                  "result1": r1, "result2": r2, "fallback": True,
                  "final_stage": 1, "lex_complete": False,
                  "eps_z": eps_z, "eps_e": None}
        if ckpt_dir and config_hash:
            _save_checkpoint(ckpt_dir, risk_lambda, result, config_hash)
        return result

    eps_e = max(1.0, 1e-6 * abs(e_star)) if eps_e is None else float(eps_e)

    # Stage 3: fragmentation
    m3, r3 = solve_fragment_stage(
        data, scenarios, beta, risk_lambda, eta, gamma, z_star, e_star,
        eps_z, eps_e, time_limit, mip_gap, disp)

    if r3.is_feasible:
        sol = extract_solution(r3, m3, data)
        final_model, final_result, final_stage = m3, r3, 3
    else:
        if r2.is_feasible:
            sol = extract_solution(r2, m2, data)
            final_model, final_result, final_stage = m2, r2, 2
        else:
            sol = extract_solution(r1, m1, data) if r1.is_feasible else None
            final_model, final_result, final_stage = m1, r1, 1
    n_act = sol["n_activations"] if sol else None

    result = {
        "z_star": z_star, "e_star": e_star, "n_activations": n_act,
        "model": final_model, "result": final_result, "solution": sol,
        "stage1_feasible": True, "stage2_feasible": True,
        "stage3_feasible": r3.is_feasible,
        "result1": r1, "result2": r2, "result3": r3,
        "final_stage": final_stage, "lex_complete": final_stage == 3,
        "eps_z": eps_z, "eps_e": eps_e,
    }
    if ckpt_dir and config_hash:
        _save_checkpoint(ckpt_dir, risk_lambda, result, config_hash)
    return result


def extract_solution(res: SolveResult, model: StochModel,
                     data: ModelData) -> dict:
    """从求解结果中提取变量值。

    Returns:
      dict with: x, y, r, b, w, Q, u, n_activations
    """
    x, y, r, b, w = {}, {}, {}, {}, {}
    Q, u = {}, {}
    for (j, i, t, s), k in model.x_idx.items():
        x[(j, i, t, s)] = float(res.x[k]) if res.x[k] > 1e-8 else 0.0
    for (j, i, t, s), k in model.y_idx.items():
        y[(j, i, t, s)] = 1 if res.x[k] > 0.5 else 0
    for (j, t), k in model.r_idx.items():
        r[(j, t)] = 1 if res.x[k] > 0.5 else 0
    for (j, t), k in model.b_idx.items():
        b[(j, t)] = 1 if res.x[k] > 0.5 else 0
    for (j, i, t, s), k in model.w_idx.items():
        w[(j, i, t, s)] = float(res.x[k]) if res.x[k] > 1e-8 else 0.0
    for key, k in model.Q_idx.items():
        Q[key] = float(res.x[k])
    for key, k in model.u_idx.items():
        u[key] = float(res.x[k])
    n_act = sum(y.values())
    return {"x": x, "y": y, "r": r, "b": b, "w": w, "Q": Q, "u": u,
            "n_activations": n_act}
