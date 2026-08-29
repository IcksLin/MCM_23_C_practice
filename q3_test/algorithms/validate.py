# -*- coding: utf-8 -*-
"""Q3约束审计 — 扩展Q2 validate.py的全面可行性检查。

继承Q2全部硬约束审计（面积守恒、重茬、豆科覆盖、水浇地模式、产量销量
平衡、CVaR复算、0-1整数性），新增Q3专属审计项（doc/Q3_编程手实现指导.md
section 3.12）：

  - min_latent_correlation_eigenvalue / max_kendall_error   (依赖模块)
  - marginal_range_violations                              (边际声明范围)
  - elasticity_sign_violations / elasticity_row_stability_violations
  - w_product_diff                                         (w = x*b 线性化精度)
  - complementarity_activation                             (b[j,t] = OR 豆类y)
  - paired_sample_count / q2_baseline_hash_match
  - frontier_complete / selected_certified

使用方法:
  from algorithms.validate import validate_solution
  report = validate_solution(plan, model, data, scenarios,
                             beta=0.90, gamma=0.03,
                             dependency_audit=dep,
                             elasticity_audit=ela,
                             frontier_complete=True,
                             selected_certified=True)

运行环境: Python 3.10+, numpy, 依赖 algorithms.preprocess /
         algorithms.scenario_reduction / algorithms.model / algorithms.risk。

作者: Q3编程手
来源: 复制自 q2_test/algorithms/validate.py 并扩展，依据
      doc/Q3_编程手实现指导.md section 3.12。
"""
from __future__ import annotations
import numpy as np

from .preprocess import (
    ModelData, LEGUME_CODES, RICE_CODE, VEG_CODES, ROOT_CODES,
    MUSHROOM_CODES, MOREL_CODE, GRAIN_CODES,
)
from .scenario_reduction import ReducedScenarioSet
from .model import StochModel
from .risk import recompute_scenario_profits, compute_cvar


# ---- Q3边际范围声明 (doc/Q3_编程手实现指导.md section 4.1) ----
_DEMAND_CEREAL_CODES = (6, 7)          # 小麦=6, 玉米=7
_DEMAND_GROWTH_LO, _DEMAND_GROWTH_HI = 0.05, 0.10      # 年增长，累积
_DEMAND_SHOCK_LO, _DEMAND_SHOCK_HI = -0.05, 0.05      # 非累积
_YIELD_SHOCK_LO, _YIELD_SHOCK_HI = -0.10, 0.10
_COST_GROWTH_LO, _COST_GROWTH_HI = 0.04, 0.06          # 累积
_PRICE_GRAIN_LO, _PRICE_GRAIN_HI = -0.01, 0.01         # 非累积
_PRICE_VEG_LO, _PRICE_VEG_HI = 0.04, 0.06              # 累积
_PRICE_MUSH_DECL_LO, _PRICE_MUSH_DECL_HI = 0.01, 0.05  # 下降率，累积
_PRICE_MOREL_DECL = 0.05                               # 固定年下降


def _price_group(i: int) -> str:
    """Q3作物价格分组（与 scenarios._crop_price_group_q3 一致）。"""
    if i in GRAIN_CODES or i == RICE_CODE:
        return "grain"
    if i == MOREL_CODE:
        return "morel"
    if i in MUSHROOM_CODES:
        return "mushroom"
    return "vegetable"


def _recompute_q3_profits(plan: dict, scenarios: ReducedScenarioSet,
                          data: ModelData, gamma: float) -> np.ndarray:
    """独立复算每情景利润，含豆类前茬互补增益。

    Pi_omega = sum p*u - sum c*x
    其中 u = min(Q, D)，Q = sum_j q_omega * (x + gamma_i * w)
    gamma_i = 0 对豆类，gamma 对非豆类；w 不产生直接成本。
    """
    x = plan["x"]; w = plan.get("w", {})
    K = scenarios.k
    profits = np.zeros(K)
    for omega in range(K):
        Q = {}
        for (j, i, t, s), area in x.items():
            if area <= 0:
                continue
            q_omega = scenarios.yield_.get((j, i, t, s), np.zeros(K))[omega]
            gamma_i = 0.0 if i in LEGUME_CODES else gamma
            w_area = w.get((j, i, t, s), 0.0)
            Q[(i, t, s)] = Q.get((i, t, s), 0.0) + q_omega * (area + gamma_i * w_area)
        revenue = 0.0
        for (i, t, s), q_tot in Q.items():
            d = scenarios.demand.get((i, t, s), np.zeros(K))[omega]
            u = min(q_tot, d)
            p = scenarios.price.get((i, t, s), np.zeros(K))[omega]
            revenue += p * u
        cost = 0.0
        for (j, i, t, s), area in x.items():
            if area <= 0:
                continue
            c = scenarios.cost.get((j, i, t, s), np.zeros(K))[omega]
            cost += c * area
        profits[omega] = revenue - cost
    return profits


def _count_marginal_range_violations(scenarios: ReducedScenarioSet,
                                     data: ModelData) -> int:
    """统计情景值超出Q3声明边际范围的样本数。

    范围依据 doc/Q3_编程手实现指导.md section 4.1：
      - 小麦/玉米销量: 年增长[5%,10%] 逐年累积
      - 其他作物销量: 相对2023年[-5%,5%] 不累积
      - 亩产: 相对2023年[-10%,10%]
      - 成本: 年增长[4%,6%] 逐年累积
      - 粮食价格: 当年趋势[-1%,1%] 不累积
      - 蔬菜价格: 年增长[4%,6%] 逐年累积
      - 食用菌价格: 年下降[1%,5%] 逐年累积
      - 羊肚菌价格: 固定年下降5%
    """
    viol = 0
    n = scenarios.k
    years = data.years
    is_pairs = sorted({(i, s) for (j, i, s) in data.suit.keys()})
    eps = 1e-6

    # ---- 需求 ----
    for (i, s) in is_pairs:
        d_base = data.D.get((i, s), 0.0)
        if d_base <= 0:
            continue
        is_cereal = i in _DEMAND_CEREAL_CODES
        for t_idx, t in enumerate(years):
            vals = scenarios.demand.get((i, t, s), np.zeros(n))
            if is_cereal:
                lo = d_base * (1.0 + _DEMAND_GROWTH_LO) ** (t_idx + 1)
                hi = d_base * (1.0 + _DEMAND_GROWTH_HI) ** (t_idx + 1)
            else:
                lo = d_base * (1.0 + _DEMAND_SHOCK_LO)
                hi = d_base * (1.0 + _DEMAND_SHOCK_HI)
            viol += int(np.sum((vals < lo - eps) | (vals > hi + eps)))

    # ---- 亩产 ----
    for (j, i, s) in data.suit.keys():
        q_base = data.q.get((j, i, s), 0.0)
        if q_base <= 0:
            continue
        for t in years:
            vals = scenarios.yield_.get((j, i, t, s), np.zeros(n))
            lo = q_base * (1.0 + _YIELD_SHOCK_LO)
            hi = q_base * (1.0 + _YIELD_SHOCK_HI)
            viol += int(np.sum((vals < lo - eps) | (vals > hi + eps)))

    # ---- 成本 (累积) ----
    for (j, i, s) in data.suit.keys():
        c_base = data.c.get((j, i, s), 0.0)
        if c_base <= 0:
            continue
        for t_idx, t in enumerate(years):
            vals = scenarios.cost.get((j, i, t, s), np.zeros(n))
            lo = c_base * (1.0 + _COST_GROWTH_LO) ** (t_idx + 1)
            hi = c_base * (1.0 + _COST_GROWTH_HI) ** (t_idx + 1)
            viol += int(np.sum((vals < lo - eps) | (vals > hi + eps)))

    # ---- 价格 ----
    for (i, s) in is_pairs:
        p_base = data.p.get((i, s), 0.0)
        if p_base <= 0:
            continue
        group = _price_group(i)
        for t_idx, t in enumerate(years):
            vals = scenarios.price.get((i, t, s), np.zeros(n))
            if group == "grain":
                lo = p_base * (1.0 + _PRICE_GRAIN_LO)
                hi = p_base * (1.0 + _PRICE_GRAIN_HI)
            elif group == "vegetable":
                lo = p_base * (1.0 + _PRICE_VEG_LO) ** (t_idx + 1)
                hi = p_base * (1.0 + _PRICE_VEG_HI) ** (t_idx + 1)
            elif group == "mushroom":
                lo = p_base * (1.0 - _PRICE_MUSH_DECL_HI) ** (t_idx + 1)
                hi = p_base * (1.0 - _PRICE_MUSH_DECL_LO) ** (t_idx + 1)
            elif group == "morel":
                target = p_base * (1.0 - _PRICE_MOREL_DECL) ** (t_idx + 1)
                lo = hi = target
            else:
                continue
            viol += int(np.sum((vals < lo - eps) | (vals > hi + eps)))
    return viol


def validate_solution(plan: dict, model: StochModel, data: ModelData,
                      scenarios: ReducedScenarioSet,
                      beta: float = 0.90, gamma: float = 0.03,
                      q2_baseline_hash_match: bool = True,
                      frontier_complete: bool = False,
                      selected_certified: bool = False,
                      dependency_audit=None,
                      elasticity_audit=None) -> dict:
    """Full Q3 constraint audit.

    Returns dict with:
      max_violation, feasible, certified,
      area_conservation, suitability, activation_bounds, min_area,
      rotation, rice_rotation, legume_coverage, irrigated_mode,
      production_sales_balance, cvar_recomputation, integrality,
      w_product_diff, complementarity_activation,
      min_latent_eigenvalue, max_kendall_error,
      marginal_range_violations, elasticity_sign_violations,
      elasticity_row_stability_violations,
      q2_baseline_hash_match, frontier_complete, selected_certified
    """
    x = plan["x"]; y = plan["y"]; r = plan.get("r", {})
    b = plan.get("b", {}); w = plan.get("w", {})
    eta = getattr(model, "eta", plan.get("eta", 0.5))
    risk_lambda = getattr(model, "risk_lambda", 0.5)
    out: dict = {}

    # === 继承自Q2的硬约束 ===

    # 1. 面积守恒（非水浇地）
    max_area_viol = 0.0
    for t in data.years:
        groups = {}
        for (j, i, s) in sorted(data.suit.keys()):
            if data.plot_type[j] == "水浇地":
                continue
            groups.setdefault((j, s), []).append(i)
        for (j, s), ilist in groups.items():
            total = sum(x.get((j, i, t, s), 0.0) for i in ilist)
            max_area_viol = max(max_area_viol, abs(total - data.A[j]))
    out["area_conservation"] = max_area_viol

    # 2. 不适宜地块
    max_unsuit = 0.0
    for (j, i, t, s), area in x.items():
        if area > 1e-8 and data.suit.get((j, i, s), 0) == 0:
            max_unsuit = max(max_unsuit, area)
    out["suitability"] = max_unsuit

    # 3. 面积-激活上界与下界: x <= A*y, x >= eta*A*y
    max_x_ub = 0.0; max_min_area = 0.0
    for (j, i, t, s), area in x.items():
        yk = y.get((j, i, t, s), 0)
        max_x_ub = max(max_x_ub, max(0.0, area - data.A[j] * yk))
        max_min_area = max(max_min_area, max(0.0, eta * data.A[j] * yk - area))
    out["activation_bounds"] = max_x_ub
    out["min_area"] = max_min_area

    # 4. 重茬邻接
    rotation_viol = 0
    for (j, i, (ta, sa), (tb, sb)) in data.adj_pairs:
        ya = (data.bar_y.get((j, i, sa), 0) if ta == 2023
              else y.get((j, i, ta, sa), 0))
        yb = (data.bar_y.get((j, i, sb), 0) if tb == 2023
              else y.get((j, i, tb, sb), 0))
        if ya + yb > 1 + 1e-6:
            rotation_viol += 1
    out["rotation"] = rotation_viol

    rice_rotation_viol = 0
    for j, ptype in enumerate(data.plot_type):
        if ptype != "水浇地":
            continue
        previous = data.r_2023.get(j, 0)
        for t in data.years:
            current = r.get((j, t), 0)
            if previous + current > 1 + 1e-6:
                rice_rotation_viol += 1
            previous = current
    out["rice_rotation"] = rice_rotation_viol

    # 5. 滚动三年豆科覆盖
    min_legume_slack = float("inf")
    for (j, window, hist) in data.legume_windows:
        legume_area = hist
        for tau in window:
            if tau == 2023:
                continue
            for s in data.plot_seasons[j]:
                for i in LEGUME_CODES:
                    legume_area += x.get((j, i, tau, s), 0.0)
        min_legume_slack = min(min_legume_slack, legume_area - data.A[j])
    out["legume_coverage"] = min_legume_slack

    # 6. 水浇地模式冲突
    mode_conflicts = 0; max_irrig_viol = 0.0; root_viol = 0
    for t in data.years:
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            rt = r.get((j, t), 0)
            root_count = sum(y.get((j, i, t, 2), 0) for i in ROOT_CODES)
            rice_area = x.get((j, RICE_CODE, t, 1), 0.0)
            veg_area = sum(x.get((j, i, t, 1), 0.0) for i in VEG_CODES)
            root_area = sum(x.get((j, i, t, 2), 0.0) for i in ROOT_CODES)
            max_irrig_viol = max(
                max_irrig_viol,
                abs(rice_area - data.A[j] * rt),
                abs(veg_area - data.A[j] * (1 - rt)),
                abs(root_area - data.A[j] * (1 - rt)),
            )
            expected = 1 - rt
            if abs(root_count - expected) > 1e-6:
                root_viol += 1
            if rt == 1 and root_count != 0:
                mode_conflicts += 1
            elif rt == 0 and root_count != 1:
                mode_conflicts += 1
    out["irrigated_mode"] = max(
        max_irrig_viol,
        float(mode_conflicts > 0),
        float(root_viol > 0),
    )

    # 7. 产量销量平衡（Q3: Q = sum_j q*(x + gamma*w)）
    K = scenarios.k
    solver_q = plan.get("Q", {}); solver_u = plan.get("u", {})
    max_q_bal = 0.0; max_u_q = 0.0; max_u_d = 0.0
    for omega in range(K):
        production = {}
        for (j, i, t, s), area in x.items():
            if area <= 0:
                continue
            q_omega = scenarios.yield_.get((j, i, t, s), np.zeros(K))[omega]
            gamma_i = 0.0 if i in LEGUME_CODES else gamma
            eff = area + gamma_i * w.get((j, i, t, s), 0.0)
            production[(i, t, s)] = production.get((i, t, s), 0.0) + q_omega * eff
        keys = set(production)
        keys.update((i, t, s) for (om, i, t, s) in solver_q if om == omega)
        keys.update((i, t, s) for (om, i, t, s) in solver_u if om == omega)
        for key in keys:
            expected_q = production.get(key, 0.0)
            q_val = solver_q.get((omega, *key), expected_q)
            demand = scenarios.demand.get(key, np.zeros(K))[omega]
            u_val = solver_u.get((omega, *key), min(expected_q, demand))
            max_q_bal = max(max_q_bal, abs(q_val - expected_q))
            max_u_q = max(max_u_q, max(0.0, u_val - q_val))
            max_u_d = max(max_u_d, max(0.0, u_val - demand))
    out["production_sales_balance"] = max(max_q_bal, max_u_q, max_u_d)

    # 8. CVaR复算（含互补增益）
    if w and gamma > 0:
        profits = _recompute_q3_profits(plan, scenarios, data, gamma)
    else:
        profits = recompute_scenario_profits(plan["x"], scenarios, data)
    weights = scenarios.weights
    e_pi_recomp = float(np.average(profits, weights=weights))
    cvar_recomp = compute_cvar(profits, weights, beta)
    e_pi = plan.get("e_pi"); cvar_value = plan.get("cvar_value")
    z_lambda = plan.get("z_lambda")
    z_recomp = (1.0 - risk_lambda) * e_pi_recomp + risk_lambda * cvar_recomp
    out["cvar_recomputation"] = max(
        abs(e_pi_recomp - e_pi) if e_pi is not None else 0.0,
        abs(cvar_recomp - cvar_value) if cvar_value is not None else 0.0,
        abs(z_recomp - z_lambda) if z_lambda is not None else 0.0,
    )

    # 9. 0-1整数性: y, r, b
    max_int_viol = 0.0
    for val in plan.get("raw_y", y).values():
        max_int_viol = max(max_int_viol, abs(val - round(val)))
    for val in plan.get("raw_r", r).values():
        max_int_viol = max(max_int_viol, abs(val - round(val)))
    for val in plan.get("raw_b", b).values():
        max_int_viol = max(max_int_viol, abs(val - round(val)))
    out["integrality"] = max_int_viol

    # === Q3专属审计 ===

    # 10. w = x*b 线性化精度（仅在b为0/1时校验）
    max_w_diff = 0.0
    for (j, i, t, s), w_val in w.items():
        bk = b.get((j, t), 0)
        b_round = round(bk)
        if abs(bk - b_round) <= 1e-6 and b_round in (0, 1):
            x_val = x.get((j, i, t, s), 0.0)
            max_w_diff = max(max_w_diff, abs(w_val - x_val * b_round))
    out["w_product_diff"] = max_w_diff

    # 11. 互补激活: b[j,t] = OR(上年豆类y)，2024用2023历史
    compl_viol = 0
    years = data.years
    for yi, t in enumerate(years):
        for j in range(len(data.plot_names)):
            bk = b.get((j, t), 0)
            if yi == 0:
                expected = 0
                for s in data.plot_seasons[j]:
                    for i in LEGUME_CODES:
                        if data.bar_y.get((j, i, s), 0) == 1:
                            expected = 1; break
                    if expected:
                        break
            else:
                t_prev = years[yi - 1]
                expected = 0
                for s in data.plot_seasons[j]:
                    for i in LEGUME_CODES:
                        if y.get((j, i, t_prev, s), 0) > 0.5:
                            expected = 1; break
                    if expected:
                        break
            if abs(bk - expected) > 1e-6:
                compl_viol += 1
    out["complementarity_activation"] = compl_viol

    # 12. 依赖模块审计（潜在相关最小特征值、Kendall误差）
    dep = dependency_audit
    out["min_latent_eigenvalue"] = (
        float(dep.min_eigenvalue) if dep is not None
        and hasattr(dep, "min_eigenvalue") else 1.0)
    out["max_kendall_error"] = (
        float(dep.max_kendall_error) if dep is not None
        and hasattr(dep, "max_kendall_error") else 0.0)

    # 13. 边际范围违反计数
    out["marginal_range_violations"] = _count_marginal_range_violations(
        scenarios, data)

    # 14. 弹性审计（符号违反、行稳定性违反）
    ela = elasticity_audit
    out["elasticity_sign_violations"] = int(
        getattr(ela, "sign_violation_count", 0) if ela is not None else 0)
    out["elasticity_row_stability_violations"] = int(
        getattr(ela, "row_stability_violation_count", 0) if ela is not None else 0)

    # 15. 成对样本计数与基线/前沿/认证状态
    out["paired_sample_count"] = int(plan.get("paired_sample_count", 0))
    out["q2_baseline_hash_match"] = bool(q2_baseline_hash_match)
    out["frontier_complete"] = bool(frontier_complete)
    out["selected_certified"] = bool(selected_certified)

    # === 汇总 ===
    out["max_violation"] = max(
        out["area_conservation"], out["suitability"],
        out["activation_bounds"], out["min_area"],
        float(out["rotation"] > 0),
        float(out["rice_rotation"] > 0),
        max(0.0, -out["legume_coverage"]),
        out["irrigated_mode"],
        out["production_sales_balance"],
        out["cvar_recomputation"],
        out["integrality"],
        out["w_product_diff"],
        float(out["complementarity_activation"] > 0),
        float(out["min_latent_eigenvalue"] < -1e-10),
        out["max_kendall_error"],
        float(out["marginal_range_violations"] > 0),
        float(out["elasticity_sign_violations"] > 0),
        float(out["elasticity_row_stability_violations"] > 0),
    )
    out["feasible"] = bool(out["max_violation"] <= 1e-4)
    out["certified"] = bool(
        out["feasible"] and out["selected_certified"]
        and out["q2_baseline_hash_match"] and out["frontier_complete"])
    return out
