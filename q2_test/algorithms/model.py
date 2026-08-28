# -*- coding: utf-8 -*-
"""Mean-CVaR stochastic MILP assembly for Q2 (AGENT.md sections 5-8).

Variable layout (all sparse-indexed):
  Common decisions (no scenario index):
    x[j,i,t,s] >= 0    planted area (mu)
    y[j,i,t,s] in {0,1} activation
    r[j,t]     in {0,1} irrigated-land mode
  Per-scenario variables (indexed by omega = 0..K-1):
    Q[omega,i,t,s] >= 0  total production (jin)
    u[omega,i,t,s] >= 0  full-price sales (jin)
    xi[omega]    >= 0     CVaR excess
  Global:
    zeta              CVaR VaR proxy (free)

Constraints:
  * Hard constraints 1-9 from Q1 (area, rotation, legume, etc.) on x,y,r
  * Q_omega = sum_j q_omega * x  (production balance, per scenario)
  * u_omega <= Q_omega            (sales ≤ production)
  * u_omega <= D_omega            (sales ≤ demand)
  * Pi_omega = sum p_omega*u_omega - sum c_omega*x  (profit, per scenario)
  * xi_omega >= zeta - Pi_omega   (CVaR excess)
  * xi_omega >= 0

Objective (minimization form, scipy.optimize.milp):
  Z_lambda = (1-lambda)*E[Pi] + lambda*LCVaR_beta
           = (1-lambda)*sum w_omega*Pi_omega + lambda*(zeta - 1/(1-beta)*sum w_omega*xi_omega)
  minimize  -Z_lambda

Three-stage lexicographic:
  stage 1 (risk):   min -Z_lambda
  stage 2 (profit): max E[Pi]  s.t.  Z_lambda >= Z* - eps
  stage 3 (fragment): min sum(y)  s.t.  E[Pi] >= E* - eps
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse

from .preprocess import ModelData, LEGUME_CODES, RICE_CODE, VEG_CODES, ROOT_CODES, GRAIN_CODES
from .scenario_reduction import ReducedScenarioSet


@dataclass
class StochModel:
    # variable index maps
    x_idx: dict           # (j,i,t,s) -> int
    y_idx: dict           # (j,i,t,s) -> int
    r_idx: dict           # (j,t) -> int
    Q_idx: dict           # (omega,i,t,s) -> int
    u_idx: dict           # (omega,i,t,s) -> int
    xi_idx: dict          # (omega,) -> int
    n: int
    # MILP data
    c: np.ndarray         # objective (minimization)
    A_ub: sparse.csr_matrix
    b_ub: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    # metadata
    eta: float
    beta: float
    risk_lambda: float
    stage: str             # "risk" | "expected" | "fragment"
    z_star: float = None
    e_star: float = None
    eps_z: float = 1e-6
    eps_e: float = 1e-6
    # -Z_lambda coefficients for lexicographic constraint
    neg_z_coefs: dict = field(default_factory=dict)
    neg_e_coefs: dict = field(default_factory=dict)


def _add(row, col, val, rows, cols, vals):
    rows.append(row); cols.append(col); vals.append(val)


def _build_indices(data: ModelData, scenarios: ReducedScenarioSet):
    """Build variable index maps. Order: x, y, r, Q, u, xi, zeta."""
    x_keys, y_keys = [], []
    for t in data.years:
        for (j, i, s) in sorted(data.suit.keys()):
            x_keys.append((j, i, t, s))
            y_keys.append((j, i, t, s))
    r_keys = []
    for t in data.years:
        for j, ptype in enumerate(data.plot_type):
            if ptype == "水浇地":
                r_keys.append((j, t))
    is_pairs = sorted({(i, s) for (j, i, s) in data.suit.keys()})
    Q_keys, u_keys = [], []
    for omega in range(scenarios.k):
        for t in data.years:
            for (i, s) in is_pairs:
                Q_keys.append((omega, i, t, s))
                u_keys.append((omega, i, t, s))
    xi_keys = [(omega,) for omega in range(scenarios.k)]

    x_idx = {k: i for i, k in enumerate(x_keys)}
    off = len(x_idx)
    y_idx = {k: off + i for i, k in enumerate(y_keys)}
    off += len(y_idx)
    r_idx = {k: off + i for i, k in enumerate(r_keys)}
    off += len(r_idx)
    Q_idx = {k: off + i for i, k in enumerate(Q_keys)}
    off += len(Q_idx)
    u_idx = {k: off + i for i, k in enumerate(u_keys)}
    off += len(u_idx)
    xi_idx = {k: off + i for i, k in enumerate(xi_keys)}
    off += len(xi_idx)
    zeta_idx = off  # scalar
    n = off + 1
    return (x_keys, y_keys, r_keys, Q_keys, u_keys, xi_keys,
            x_idx, y_idx, r_idx, Q_idx, u_idx, xi_idx, zeta_idx, n)


def build_q2_model(data: ModelData, scenarios: ReducedScenarioSet,
                   beta: float = 0.90, risk_lambda: float = 0.5,
                   eta: float = 0.5, stage: str = "risk",
                   z_star: float = None, e_star: float = None,
                   eps_z: float = 1e-6, eps_e: float = 1e-6) -> StochModel:
    """Assemble the sparse stochastic MILP.

    Args:
      data: ModelData from preprocess.
      scenarios: ReducedScenarioSet from scenario_reduction.
      beta: CVaR confidence level (default 0.90).
      risk_lambda: risk-aversion parameter in [0,1].
      eta: minimum area ratio.
      stage: "risk" | "expected" | "fragment".
      z_star: optimal Z_lambda from stage 1 (for stage 2).
      e_star: optimal E[Pi] from stage 2 (for stage 3).
      eps_z, eps_e: lexicographic tolerance.
    """
    (x_keys, y_keys, r_keys, Q_keys, u_keys, xi_keys,
     x_idx, y_idx, r_idx, Q_idx, u_idx, xi_idx, zeta_idx, n
     ) = _build_indices(data, scenarios)

    K = scenarios.k
    w = scenarios.weights  # (K,)

    c = np.zeros(n)
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    integrality = np.zeros(n, dtype=int)

    # ---- bounds ----
    for (j, i, t, s), k in x_idx.items():
        ub[k] = data.A[j]
    for k in y_idx.values():
        lb[k] = 0.0; ub[k] = 1.0; integrality[k] = 1
    for k in r_idx.values():
        lb[k] = 0.0; ub[k] = 1.0; integrality[k] = 1
    # zeta is free (can be negative for low-profit scenarios)
    lb[zeta_idx] = -np.inf

    # ---- objective coefficients ----
    # Z_lambda = (1-lam)*E[Pi] + lam*LCVaR
    # E[Pi] = sum_omega w_omega * (sum p*u - sum c*x)
    #       = sum_omega w_omega * sum_its p_omega*u_omega - sum_omega w_omega * sum_jits c_omega * x
    # But x is common: cost term = sum_{omega} w_omega * sum_{jits} c_omega * x_jits
    #                   = sum_{jits} [sum_omega w_omega * c_omega(jits)] * x_jits
    #
    # LCVaR = zeta - 1/(1-beta) * sum_omega w_omega * xi_omega

    neg_z = np.zeros(n)   # -Z_lambda coefficients (for lex constraint)
    neg_e = np.zeros(n)   # -E[Pi] coefficients (for lex constraint)

    lam = risk_lambda

    # cost on x: weighted average cost across scenarios
    for (j, i, t, s), k in x_idx.items():
        avg_cost = 0.0
        for omega in range(K):
            avg_cost += w[omega] * scenarios.cost.get((j, i, t, s), np.zeros(K))[omega]
        # in Z_lambda: -(1-lam) * w_avg_cost * x  (cost reduces profit)
        # minimize -> +(1-lam) * avg_cost
        c[k] += (1.0 - lam) * avg_cost
        # -Z_lambda: -(1-lam) * (-avg_cost) = (1-lam) * avg_cost  (same sign for -Z)
        neg_z[k] += (1.0 - lam) * avg_cost
        # -E[Pi]: -(-avg_cost) = avg_cost
        neg_e[k] += avg_cost

    # revenue on u: weighted average price across scenarios
    for (omega, i, t, s), uk in u_idx.items():
        p_omega = scenarios.price.get((i, t, s), np.zeros(K))[omega]
        # E[Pi] += w_omega * p_omega * u_omega  (per scenario)
        # Z_lambda: +(1-lam) * w_omega * p_omega * u_omega
        # minimize -> -(1-lam) * w_omega * p_omega
        c[uk] -= (1.0 - lam) * w[omega] * p_omega
        neg_z[uk] -= (1.0 - lam) * w[omega] * p_omega
        neg_e[uk] -= w[omega] * p_omega

    # CVaR terms: zeta and xi
    # LCVaR = zeta - 1/(1-beta) * sum w_omega * xi_omega
    # Z_lambda: +lam * zeta - lam/(1-beta) * sum w * xi
    # minimize: -lam * zeta + lam/(1-beta) * sum w * xi
    c[zeta_idx] -= lam
    neg_z[zeta_idx] -= lam
    for (omega,), xk in xi_idx.items():
        c[xk] += lam / (1.0 - beta) * w[omega]
        neg_z[xk] += lam / (1.0 - beta) * w[omega]

    # stage overrides
    if stage == "expected":
        # maximize E[Pi] = minimize -E[Pi] = minimize sum(neg_e * vars)
        c = neg_e.copy()
    elif stage == "fragment":
        # minimize sum(y)
        c = np.zeros(n)
        for k in y_idx.values():
            c[k] = 1.0

    # ---- constraints ----
    ub_rows, ub_cols, ub_vals = [], [], []
    eq_rows, eq_cols, eq_vals = [], [], []
    b_ub_list, b_eq_list = [], []
    r_ub = -1
    r_eq = -1

    # === HARD CONSTRAINTS (from Q1, on x/y/r only) ===

    # 1. area-activation: x - A_j*y <= 0, -x + eta*A_j*y <= 0
    for (j, i, t, s), k in x_idx.items():
        Aj = data.A[j]
        yk = y_idx[(j, i, t, s)]
        r_ub += 1; _add(r_ub, k, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, yk, -Aj, ub_rows, ub_cols, ub_vals); b_ub_list.append(0.0)
        r_ub += 1; _add(r_ub, k, -1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, yk, eta * Aj, ub_rows, ub_cols, ub_vals); b_ub_list.append(0.0)

    # 1b. cardinality cut
    max_y = int(np.floor(1.0 / eta + 1e-9)) if eta > 0 else 999
    for t in data.years:
        groups = {}
        for (j, i, s) in sorted(data.suit.keys()):
            groups.setdefault((j, s), []).append(i)
        for (j, s), ilist in groups.items():
            r_ub += 1
            for i in ilist:
                yk = y_idx.get((j, i, t, s))
                if yk is not None:
                    _add(r_ub, yk, 1.0, ub_rows, ub_cols, ub_vals)
            b_ub_list.append(float(max_y))

    # 2. area conservation (non-irrigated)
    for t in data.years:
        groups = {}
        for (j, i, s) in sorted(data.suit.keys()):
            if data.plot_type[j] == "水浇地":
                continue
            groups.setdefault((j, s), []).append(i)
        for (j, s), ilist in groups.items():
            r_eq += 1
            for i in ilist:
                _add(r_eq, x_idx[(j, i, t, s)], 1.0, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(data.A[j])

    # 3. irrigated mode constraints
    for t in data.years:
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            rk = r_idx[(j, t)]
            Aj = data.A[j]
            if (j, RICE_CODE, t, 1) in x_idx:
                r_eq += 1
                _add(r_eq, x_idx[(j, RICE_CODE, t, 1)], 1.0, eq_rows, eq_cols, eq_vals)
                _add(r_eq, rk, -Aj, eq_rows, eq_cols, eq_vals); b_eq_list.append(0.0)
            r_eq += 1
            for i in VEG_CODES:
                if (j, i, t, 1) in x_idx:
                    _add(r_eq, x_idx[(j, i, t, 1)], 1.0, eq_rows, eq_cols, eq_vals)
            _add(r_eq, rk, Aj, eq_rows, eq_cols, eq_vals); b_eq_list.append(Aj)
            r_eq += 1
            for i in ROOT_CODES:
                if (j, i, t, 2) in x_idx:
                    _add(r_eq, x_idx[(j, i, t, 2)], 1.0, eq_rows, eq_cols, eq_vals)
            _add(r_eq, rk, Aj, eq_rows, eq_cols, eq_vals); b_eq_list.append(Aj)
            r_eq += 1
            for i in ROOT_CODES:
                if (j, i, t, 2) in y_idx:
                    _add(r_eq, y_idx[(j, i, t, 2)], 1.0, eq_rows, eq_cols, eq_vals)
            _add(r_eq, rk, 1.0, eq_rows, eq_cols, eq_vals); b_eq_list.append(1.0)

    # 4. 重茬 adjacency
    for (j, i, (ta, sa), (tb, sb)) in data.adj_pairs:
        if ta == 2023 and tb == 2023:
            continue
        if ta == 2023:
            bar = data.bar_y.get((j, i, sa), 0)
            if bar == 1:
                kb = y_idx.get((j, i, tb, sb))
                if kb is not None:
                    ub[kb] = 0.0
            continue
        if tb == 2023:
            bar = data.bar_y.get((j, i, sb), 0)
            ka = y_idx.get((j, i, ta, sa))
            if ka is not None and bar == 1:
                ub[ka] = 0.0
            continue
        ka = y_idx.get((j, i, ta, sa))
        kb = y_idx.get((j, i, tb, sb))
        if ka is None or kb is None:
            continue
        r_ub += 1
        _add(r_ub, ka, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, kb, 1.0, ub_rows, ub_cols, ub_vals); b_ub_list.append(1.0)

    # 5. rice year adjacency
    for t_idx in range(len(data.years) - 1):
        t = data.years[t_idx]; t1 = data.years[t_idx + 1]
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            r_ub += 1
            _add(r_ub, r_idx[(j, t)], 1.0, ub_rows, ub_cols, ub_vals)
            _add(r_ub, r_idx[(j, t1)], 1.0, ub_rows, ub_cols, ub_vals); b_ub_list.append(1.0)
    for j, ptype in enumerate(data.plot_type):
        if ptype != "水浇地":
            continue
        if data.r_2023.get(j, 0) == 1:
            ub[r_idx[(j, data.years[0])]] = 0.0

    # 6. rolling 3-year legume coverage
    for (j, window, hist) in data.legume_windows:
        r_ub += 1
        for tau in window:
            if tau == 2023:
                continue
            for s in data.plot_seasons[j]:
                for i in LEGUME_CODES:
                    k = x_idx.get((j, i, tau, s))
                    if k is not None:
                        _add(r_ub, k, -1.0, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(hist - data.A[j])

    # === SCENARIO CONSTRAINTS ===

    # 7. production: Q_omega = sum_j q_omega * x
    prod_groups = {}
    for (j, i, t, s), k in x_idx.items():
        prod_groups.setdefault((i, t, s), []).append((j, k))
    for (i, t, s), jks in prod_groups.items():
        for omega in range(K):
            qk = Q_idx.get((omega, i, t, s))
            if qk is None:
                continue
            r_eq += 1
            _add(r_eq, qk, 1.0, eq_rows, eq_cols, eq_vals)
            for (j, k) in jks:
                q_omega = scenarios.yield_.get((j, i, t, s), np.zeros(K))[omega]
                _add(r_eq, k, -q_omega, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(0.0)

    # 8. sales: u_omega - Q_omega <= 0,  u_omega <= D_omega
    for (omega, i, t, s), uk in u_idx.items():
        qk = Q_idx.get((omega, i, t, s))
        if qk is not None:
            r_ub += 1
            _add(r_ub, uk, 1.0, ub_rows, ub_cols, ub_vals)
            _add(r_ub, qk, -1.0, ub_rows, ub_cols, ub_vals); b_ub_list.append(0.0)
        # demand bound
        d_omega = scenarios.demand.get((i, t, s), np.zeros(K))[omega]
        if d_omega > 0:
            ub[uk] = min(ub[uk], d_omega)
        else:
            ub[uk] = 0.0

    # 9. CVaR excess: xi_omega >= zeta - Pi_omega
    #    -> -zeta + xi_omega + Pi_omega_terms >= 0
    #    Actually: xi >= zeta - Pi
    #    -> zeta - xi - Pi <= 0
    #    Pi_omega = sum p_omega*u_omega - sum c_omega*x
    #    So: zeta - xi_omega - (sum p*u - sum c*x) <= 0
    #    -> zeta - xi_omega - sum p*u + sum c*x <= 0
    for (omega,), xk in xi_idx.items():
        r_ub += 1
        _add(r_ub, zeta_idx, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, xk, -1.0, ub_rows, ub_cols, ub_vals)
        # -Pi terms: -p*u (revenue reduces excess) and +c*x (cost increases excess)
        for (i, t, s) in sorted({(i, t, s) for t in data.years for (j, i, s) in data.suit.keys()}):
            uk = u_idx.get((omega, i, t, s))
            if uk is not None:
                p_omega = scenarios.price.get((i, t, s), np.zeros(K))[omega]
                _add(r_ub, uk, -p_omega, ub_rows, ub_cols, ub_vals)
        for (j, i, t, s), k in x_idx.items():
            c_omega = scenarios.cost.get((j, i, t, s), np.zeros(K))[omega]
            if c_omega != 0:
                _add(r_ub, k, c_omega, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(0.0)

    # === LEXICOGRAPHIC CONSTRAINTS ===

    # 10. stage 2: Z_lambda >= Z* - eps  ->  -Z_lambda <= -(Z* - eps)
    if stage in ("expected", "fragment") and z_star is not None:
        r_ub += 1
        for k, coef in enumerate(neg_z):
            if coef != 0.0:
                _add(r_ub, k, coef, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(-(z_star - eps_z))

    # 11. stage 3: E[Pi] >= E* - eps  ->  -E[Pi] <= -(E* - eps)
    if stage == "fragment" and e_star is not None:
        r_ub += 1
        for k, coef in enumerate(neg_e):
            if coef != 0.0:
                _add(r_ub, k, coef, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(-(e_star - eps_e))

    # ---- assemble matrices ----
    if ub_rows:
        A_ub = sparse.csr_matrix(
            (np.array(ub_vals, dtype=float),
             (np.array(ub_rows, dtype=np.int64),
              np.array(ub_cols, dtype=np.int64))),
            shape=(len(b_ub_list), n))
    else:
        A_ub = sparse.csr_matrix((0, n))
    if eq_rows:
        A_eq = sparse.csr_matrix(
            (np.array(eq_vals, dtype=float),
             (np.array(eq_rows, dtype=np.int64),
              np.array(eq_cols, dtype=np.int64))),
            shape=(len(b_eq_list), n))
    else:
        A_eq = sparse.csr_matrix((0, n))
    b_ub = np.array(b_ub_list, dtype=float)
    b_eq = np.array(b_eq_list, dtype=float)

    return StochModel(
        x_idx=x_idx, y_idx=y_idx, r_idx=r_idx, Q_idx=Q_idx, u_idx=u_idx,
        xi_idx=xi_idx, n=n, c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub, integrality=integrality,
        eta=eta, beta=beta, risk_lambda=risk_lambda, stage=stage,
        z_star=z_star, e_star=e_star, eps_z=eps_z, eps_e=eps_e,
        neg_z_coefs={k: float(v) for k, v in enumerate(neg_z) if v != 0.0},
        neg_e_coefs={k: float(v) for k, v in enumerate(neg_e) if v != 0.0},
    )
