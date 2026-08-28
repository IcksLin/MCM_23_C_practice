# -*- coding: utf-8 -*-
"""Constraint & numerical audit (AGENT.md section 10).

Tolerance: in-memory 1e-6, Excel round-trip 1e-4.

Audit metrics:
  - max_area_conservation_violation   |sum x - A_j| for non-irrigated (j,t,s)
  - max_unsuitable_area               x on combos with e=0
  - max_min_area_violation            eta*A_j*y - x  (when y==1)
  - monoculture_violation_count       重茬: y_a==y_b==1 on adjacent slots
  - legume_min_slack                  min over windows of (hist+sum legume x - A_j)
  - irrigated_mode_conflict_count     rice+veg both active, or root-y != 1-r
  - max_u_exceeds_D                   u - lambda*D
  - max_u_exceeds_Q                   u - Q
  - profit_recompute_diff             |recomputed - solver objective|
  - excel_roundtrip_diff              set by export_excel.reread_audit()
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import math
import numpy as np
import pandas as pd

from .preprocess import ModelData, LEGUME_CODES, RICE_CODE, VEG_CODES, ROOT_CODES


@dataclass
class AuditReport:
    max_area_conservation_violation: float = 0.0
    max_unsuitable_area: float = 0.0
    max_upper_link_violation: float = 0.0
    max_min_area_violation: float = 0.0
    max_production_balance_violation: float = 0.0
    max_irrigated_area_violation: float = 0.0
    max_integrality_violation: float = 0.0
    monoculture_violation_count: int = 0
    legume_min_slack: float = 0.0
    irrigated_mode_conflict_count: int = 0
    max_u_exceeds_D: float = 0.0
    max_u_exceeds_Q: float = 0.0
    profit_recompute_diff: float = 0.0
    excel_roundtrip_diff: float = 0.0
    feasible: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _x_map(sol_x: pd.DataFrame) -> dict:
    """(j,i,t,s) -> area from the long table."""
    return {
        (int(r["plot_idx"]), int(r["crop_code"]), int(r["year"]), int(r["season"])):
        float(r["area"])
        for _, r in sol_x.iterrows()
    }


def _y_map(sol_y: pd.DataFrame) -> dict:
    return {
        (int(r["plot_idx"]), int(r["crop_code"]), int(r["year"]), int(r["season"])):
        int(r["y"])
        for _, r in sol_y.iterrows()
    }


def _r_map(sol_r: pd.DataFrame) -> dict:
    return {(int(r["plot_idx"]), int(r["year"])): int(r["r"])
            for _, r in sol_r.iterrows()}


def _Q_map(sol_Q: pd.DataFrame) -> dict:
    return {(int(r["crop_code"]), int(r["year"]), int(r["season"])): float(r["Q"])
            for _, r in sol_Q.iterrows()}


def _u_map(sol_u: pd.DataFrame) -> dict:
    return {(int(r["crop_code"]), int(r["year"]), int(r["season"])): float(r["u"])
            for _, r in sol_u.iterrows()}


def validate_solution(sol: dict, data: ModelData, tol: float = 1e-6) -> AuditReport:
    """Run all audits. `sol` is the dict returned by solve.extract_solution."""
    rep = AuditReport()
    x_map = _x_map(sol["x"]) if sol.get("x") is not None else {}
    y_map = _y_map(sol["y"]) if sol.get("y") is not None else {}
    r_map = _r_map(sol["r"]) if sol.get("r") is not None else {}
    Q_map = _Q_map(sol["Q"]) if sol.get("Q") is not None else {}
    u_map = _u_map(sol["u"]) if sol.get("u") is not None else {}

    eta = sol.get("eta", 0.5)
    lam = sol.get("demand_scale", 1.0)
    scenario = sol.get("scenario", 1)

    # ---- 1. area conservation (non-irrigated): sum_i x = A_j per (j,t,s) ----
    acc = {}
    for (j, i, t, s), a in x_map.items():
        if data.plot_type[j] == "水浇地":
            continue
        acc[(j, t, s)] = acc.get((j, t, s), 0.0) + a
    max_ac = 0.0
    for (j, t, s), tot in acc.items():
        v = abs(tot - data.A[j])
        if v > max_ac:
            max_ac = v
    # also flag (j,t,s) with no planting but should be fully planted
    for t in data.years:
        for (j, i, s) in data.suit:
            if data.plot_type[j] == "水浇地":
                continue
            if (j, t, s) not in acc:
                v = abs(data.A[j])
                if v > max_ac:
                    max_ac = v
    rep.max_area_conservation_violation = max_ac

    # ---- 2. unsuitable area ----
    max_uns = 0.0
    for (j, i, t, s), a in x_map.items():
        if data.suit.get((j, i, s), 0) != 1:
            if a > max_uns:
                max_uns = a
    rep.max_unsuitable_area = max_uns

    # ---- 3. min-area violation  (eta*A_j*y - x > 0 when y==1) ----
    max_upper = 0.0
    for (j, i, t, s), a in x_map.items():
        yv = y_map.get((j, i, t, s), 0)
        max_upper = max(max_upper, a - data.A[j] * yv)
    rep.max_upper_link_violation = max(0.0, max_upper)

    max_min = 0.0
    for (j, i, t, s), yv in y_map.items():
        if yv != 1:
            continue
        a = x_map.get((j, i, t, s), 0.0)
        need = eta * data.A[j]
        if need - a > max_min:
            max_min = need - a
    rep.max_min_area_violation = max_min

    # ---- 3b. Q definition: Q = sum_j q*x ----
    expected_Q = {}
    for (j, i, t, s), a in x_map.items():
        expected_Q[(i, t, s)] = expected_Q.get((i, t, s), 0.0) + \
            data.q[(j, i, s)] * a
    max_q_balance = 0.0
    for key in set(expected_Q) | set(Q_map):
        max_q_balance = max(
            max_q_balance,
            abs(Q_map.get(key, 0.0) - expected_Q.get(key, 0.0)))
    rep.max_production_balance_violation = max_q_balance

    # ---- 4. monoculture (重茬) violation ----
    mv = 0
    for (j, i, (ta, sa), (tb, sb)) in data.adj_pairs:
        if ta == 2023:
            ya = data.bar_y.get((j, i, sa), 0)
        else:
            ya = y_map.get((j, i, ta, sa), 0)
        if tb == 2023:
            yb = data.bar_y.get((j, i, sb), 0)
        else:
            yb = y_map.get((j, i, tb, sb), 0)
        if ya == 1 and yb == 1:
            mv += 1
    rep.monoculture_violation_count = mv

    # ---- 5. legume 3-year min slack ----
    min_slack = math.inf
    for (j, window, hist) in data.legume_windows:
        s = hist
        for tau in window:
            if tau == 2023:
                continue
            for ss in data.plot_seasons[j]:
                for i in LEGUME_CODES:
                    s += x_map.get((j, i, tau, ss), 0.0)
        slack = s - data.A[j]
        if slack < min_slack:
            min_slack = slack
    rep.legume_min_slack = float(min_slack if min_slack != math.inf else 0.0)

    # ---- 6. irrigated mode conflicts ----
    conf = 0
    max_irrigated_area = 0.0
    for t in data.years:
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            r = r_map.get((j, t), 0)
            # rice + veg both active
            rice_y = y_map.get((j, RICE_CODE, t, 1), 0)
            veg_active = any(y_map.get((j, i, t, 1), 0) == 1 for i in VEG_CODES)
            if r == 1 and veg_active:
                conf += 1
            if r == 0 and rice_y == 1:
                conf += 1
            # root s2: exactly (1-r) active
            root_y_sum = sum(y_map.get((j, i, t, 2), 0) for i in ROOT_CODES)
            if root_y_sum != 1 - r:
                conf += 1
            # rice area consistency
            rice_x = x_map.get((j, RICE_CODE, t, 1), 0.0)
            if abs(rice_x - r * data.A[j]) > 1e-6:
                conf += 1
            veg_x = sum(x_map.get((j, i, t, 1), 0.0) for i in VEG_CODES)
            root_x = sum(x_map.get((j, i, t, 2), 0.0) for i in ROOT_CODES)
            expected_area = (1 - r) * data.A[j]
            max_irrigated_area = max(
                max_irrigated_area,
                abs(veg_x - expected_area), abs(root_x - expected_area),
                abs(rice_x - r * data.A[j]))
    rep.irrigated_mode_conflict_count = conf
    rep.max_irrigated_area_violation = max_irrigated_area
    rep.max_integrality_violation = float(
        sol.get("max_integrality_violation", float("inf")))

    # ---- 7. u <= lambda*D and u <= Q ----
    max_ud = 0.0
    max_uq = 0.0
    for (i, t, s), uv in u_map.items():
        D = lam * data.D.get((i, s), 0.0)
        if uv - D > max_ud:
            max_ud = uv - D
        Q = Q_map.get((i, t, s), 0.0)
        if uv - Q > max_uq:
            max_uq = uv - Q
    rep.max_u_exceeds_D = max_ud
    rep.max_u_exceeds_Q = max_uq

    # ---- 8. profit recompute diff ----
    rep.profit_recompute_diff = abs(
        sol.get("profit_recomputed", 0.0) - sol.get("objective", 0.0))

    # ---- overall feasibility ----
    objective = sol.get("objective", float("nan"))
    rep.feasible = (max_ac < tol and max_uns < tol
                    and rep.max_upper_link_violation < tol and max_min < tol
                    and rep.max_production_balance_violation < tol
                    and rep.max_irrigated_area_violation < tol
                    and rep.max_integrality_violation < tol
                    and mv == 0 and conf == 0 and max_ud < tol
                    and max_uq < tol and min_slack >= -tol
                    and rep.profit_recompute_diff < tol
                    and math.isfinite(objective))
    return rep


# ---------------------------------------------------------------------------
# unit-test style checks (AGENT.md section 10)
# ---------------------------------------------------------------------------

def income_for(Q: float, u: float, p: float, scenario: int) -> float:
    """Revenue for one (i,t,s) given Q, u, price, scenario."""
    if scenario == 1:
        return p * u                       # surplus wasted
    return p * u + 0.5 * p * (Q - u)         # surplus half-price


def check_revenue_relations() -> dict:
    """Unit-test the income formula for Q<D, Q=D, Q>D and the D->inf limit."""
    p = 4.0
    D = 100.0
    out = {}
    # Q < D
    out["Q<D s1"] = income_for(50, 50, p, 1)
    out["Q<D s2"] = income_for(50, 50, p, 2)
    # Q = D
    out["Q=D s1"] = income_for(100, 100, p, 1)
    out["Q=D s2"] = income_for(100, 100, p, 2)
    # Q > D  (u capped at D)
    out["Q>D s1"] = income_for(150, 100, p, 1)
    out["Q>D s2"] = income_for(150, 100, p, 2)
    # D large: u = Q, both scenarios equal
    out["D=inf s1"] = income_for(50, 50, p, 1)
    out["D=inf s2"] = income_for(50, 50, p, 2)
    return out
