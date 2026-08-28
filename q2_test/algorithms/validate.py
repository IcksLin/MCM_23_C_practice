# -*- coding: utf-8 -*-
"""Constraint audit for Q2 (AGENT.md section 16).

Validates all hard constraints and scenario-specific quantities
independently from the solver.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .preprocess import ModelData, LEGUME_CODES, RICE_CODE, VEG_CODES, ROOT_CODES


def validate_solution(plan: dict, data: ModelData, scenarios,
                      beta: float = 0.90, risk_lambda: float = 0.5,
                      z_lambda: float | None = None,
                      e_pi: float | None = None,
                      cvar_value: float | None = None,
                      eta: float = 0.5,
                      excel_audit: dict | None = None,
                      solver_info: dict | None = None) -> dict:
    """Run full audit. Returns dict of violation metrics.

    plan: dict with keys x, y, r from solve.extract_solution.
    """
    x = plan["x"]; y = plan["y"]; r = plan.get("r", {})
    out = {}
    # 1. area conservation (non-irrigated)
    max_area_viol = 0.0
    for t in data.years:
        groups = {}
        for (j, i, s) in sorted(data.suit.keys()):
            if data.plot_type[j] == "水浇地":
                continue
            groups.setdefault((j, s), []).append(i)
        for (j, s), ilist in groups.items():
            total = sum(x.get((j, i, t, s), 0.0) for i in ilist)
            viol = abs(total - data.A[j])
            max_area_viol = max(max_area_viol, viol)
    out["max_area_violation"] = max_area_viol

    # 2. unsuitability
    max_unsuit = 0.0
    for (j, i, t, s), area in x.items():
        if area > 1e-8 and data.suit.get((j, i, s), 0) == 0:
            max_unsuit = max(max_unsuit, area)
    out["max_unsuitable_area"] = max_unsuit

    # 3. area-activation upper: x <= A_j * y
    max_x_ub = 0.0
    for (j, i, t, s), area in x.items():
        yk = y.get((j, i, t, s), 0)
        viol = max(0.0, area - data.A[j] * yk)
        max_x_ub = max(max_x_ub, viol)
    out["max_area_activation_ub_violation"] = max_x_ub

    # 4. minimum area: x >= eta * A_j * y
    max_min_area = 0.0
    for (j, i, t, s), area in x.items():
        yk = y.get((j, i, t, s), 0)
        viol = max(0.0, eta * data.A[j] * yk - area)
        max_min_area = max(max_min_area, viol)
    out["max_min_area_violation"] = max_min_area

    # 5. 重茬 violation count
    rotation_viol = 0
    for (j, i, (ta, sa), (tb, sb)) in data.adj_pairs:
        ya = (data.bar_y.get((j, i, sa), 0) if ta == 2023
              else y.get((j, i, ta, sa), 0))
        yb = (data.bar_y.get((j, i, sb), 0) if tb == 2023
              else y.get((j, i, tb, sb), 0))
        if ya + yb > 1 + 1e-6:
            rotation_viol += 1
    out["rotation_violation_count"] = rotation_viol

    rice_rotation_viol = 0
    for j, ptype in enumerate(data.plot_type):
        if ptype != "水浇地":
            continue
        previous = data.r_2023.get(j, 0)
        for t in data.years:
            current = r.get((j, t), 0)
            if previous + current > 1:
                rice_rotation_viol += 1
            previous = current
    out["rice_rotation_violation_count"] = rice_rotation_viol

    # 6. rolling 3-year legume
    min_legume_slack = float("inf")
    for (j, window, hist) in data.legume_windows:
        legume_area = hist
        for tau in window:
            if tau == 2023:
                continue
            for s in data.plot_seasons[j]:
                for i in LEGUME_CODES:
                    legume_area += x.get((j, i, tau, s), 0.0)
        slack = legume_area - data.A[j]
        min_legume_slack = min(min_legume_slack, slack)
    out["min_legume_slack"] = min_legume_slack

    # 7. irrigated mode conflicts
    mode_conflicts = 0
    root_count_dev = 0.0
    root_count_viol = 0
    max_irrigated_area_viol = 0.0
    for t in data.years:
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            rt = r.get((j, t), 0)
            root_count = sum(y.get((j, i, t, 2), 0) for i in ROOT_CODES)
            rice_area = x.get((j, RICE_CODE, t, 1), 0.0)
            veg_area = sum(x.get((j, i, t, 1), 0.0) for i in VEG_CODES)
            root_area = sum(x.get((j, i, t, 2), 0.0) for i in ROOT_CODES)
            max_irrigated_area_viol = max(
                max_irrigated_area_viol,
                abs(rice_area - data.A[j] * rt),
                abs(veg_area - data.A[j] * (1 - rt)),
                abs(root_area - data.A[j] * (1 - rt)),
            )
            expected_root_count = 1 - rt
            dev = abs(root_count - expected_root_count)
            root_count_dev = max(root_count_dev, dev)
            if dev > 1e-6:
                root_count_viol += 1
            if rt == 1:
                if root_count != 0:
                    mode_conflicts += 1
            else:
                if root_count != 1:
                    mode_conflicts += 1
    out["irrigated_mode_conflicts"] = mode_conflicts
    out["max_irrigated_area_violation"] = max_irrigated_area_viol
    out["max_root_activation_count_deviation"] = root_count_dev
    out["root_activation_violation_count"] = root_count_viol

    # 8. Scenario production/sales variables, checked against an independent
    # recomputation from the common planting plan.
    K = scenarios.k if hasattr(scenarios, "k") else scenarios.n
    max_q_balance = 0.0
    max_u_exceeds_q = 0.0
    max_u_exceeds_d = 0.0
    solver_q = plan.get("Q", {})
    solver_u = plan.get("u", {})
    for omega in range(K):
        production = {}
        for (j, i, t, s), area in x.items():
            if area <= 0:
                continue
            q = scenarios.yield_.get((j, i, t, s), np.zeros(K))[omega]
            production[(i, t, s)] = production.get((i, t, s), 0.0) + q * area
        keys = set(production)
        keys.update((i, t, s) for (w, i, t, s) in solver_q if w == omega)
        keys.update((i, t, s) for (w, i, t, s) in solver_u if w == omega)
        for key in keys:
            expected_q = production.get(key, 0.0)
            q_value = solver_q.get((omega, *key), expected_q)
            u_value = solver_u.get((omega, *key), min(
                expected_q, scenarios.demand.get(key, np.zeros(K))[omega]))
            demand = scenarios.demand.get(key, np.zeros(K))[omega]
            max_q_balance = max(max_q_balance, abs(q_value - expected_q))
            max_u_exceeds_q = max(max_u_exceeds_q, max(0.0, u_value - q_value))
            max_u_exceeds_d = max(max_u_exceeds_d, max(0.0, u_value - demand))
    out["max_production_balance_diff"] = max_q_balance
    out["max_u_exceeds_Q"] = max_u_exceeds_q
    out["max_u_exceeds_D"] = max_u_exceeds_d

    # 9. scenario profit/CVaR recomputation
    from .risk import recompute_scenario_profits, compute_cvar
    profits = recompute_scenario_profits(plan["x"], scenarios, data)
    weights = scenarios.weights if hasattr(scenarios, "weights") else np.full(K, 1.0/K)
    e_pi_recomp = float(np.average(profits, weights=weights))
    cvar_recomp = compute_cvar(profits, weights, beta)

    out["expected_profit_recomputed"] = e_pi_recomp
    out["cvar_recomputed"] = cvar_recomp
    out["profit_recomputation_diff"] = (
        abs(e_pi_recomp - e_pi) if e_pi is not None else 0.0)
    out["cvar_recomputation_diff"] = (
        abs(cvar_recomp - cvar_value) if cvar_value is not None else 0.0)
    z_recomp = (1.0 - risk_lambda) * e_pi_recomp + risk_lambda * cvar_recomp
    out["z_lambda_recomputed"] = z_recomp
    out["z_lambda_diff"] = abs(z_recomp - z_lambda) if z_lambda is not None else 0.0

    # 9. 0-1 integrality
    max_int_viol = 0.0
    for val in plan.get("raw_y", y).values():
        frac = abs(val - round(val))
        max_int_viol = max(max_int_viol, frac)
    for val in plan.get("raw_r", r).values():
        frac = abs(val - round(val))
        max_int_viol = max(max_int_viol, frac)
    out["max_integrality_violation"] = max_int_viol

    # 11. Excel/OOXML and solver certification
    excel_audit = excel_audit or {}
    out["excel_roundtrip_diff"] = float(excel_audit.get("max_roundtrip_diff", 0.0))
    out["ooxml_non_target_diff_count"] = int(
        excel_audit.get("non_target_diff_count", 0))
    out["ooxml_structure_diff_count"] = int(
        excel_audit.get("structure_diff_count", 0))
    out["xlsx_changed_sheet_count"] = int(
        excel_audit.get("changed_sheet_count", 0))

    solver_info = solver_info or {}
    out["solver_status"] = solver_info.get("status", "unknown")
    out["solver_dual_bound"] = solver_info.get("dual_bound", np.nan)
    out["solver_mip_gap"] = solver_info.get("mip_gap", np.nan)
    out["solver_time_seconds"] = solver_info.get("time", np.nan)
    out["solver_certified"] = bool(solver_info.get("certified", False))

    # summary
    out["max_violation"] = max(
        out["max_area_violation"], out["max_unsuitable_area"],
        out["max_area_activation_ub_violation"], out["max_min_area_violation"],
        float(out["rotation_violation_count"] > 0),
        float(out["rice_rotation_violation_count"] > 0),
        max(0.0, -out["min_legume_slack"]),
        float(out["irrigated_mode_conflicts"] > 0),
        out["max_irrigated_area_violation"],
        out["max_root_activation_count_deviation"],
        out["max_production_balance_diff"], out["max_u_exceeds_Q"],
        out["max_u_exceeds_D"],
        out["max_integrality_violation"],
        out["excel_roundtrip_diff"],
        float(out["ooxml_non_target_diff_count"] > 0),
        float(out["ooxml_structure_diff_count"] > 0),
    )
    out["feasible"] = bool(
        out["max_violation"] <= 1e-6
        and out["profit_recomputation_diff"] <= 1e-4
        and out["cvar_recomputation_diff"] <= 1e-4
    )
    out["certified"] = bool(out["feasible"] and out["solver_certified"])

    return out
