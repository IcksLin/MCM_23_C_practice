# -*- coding: utf-8 -*-
"""MILP assembly for Q1 (scipy.optimize.milp / HiGHS backend).

Implements the priority-1 model contract from AGENT.md section 3 and
题目分析报告.md section 6:

  Variables
    x[j,i,t,s] >= 0   planted area (mu)         -- only suitable combos
    y[j,i,t,s] in {0,1} activation               -- same keys as x
    r[j,t]     in {0,1} irrigated-land mode (1=rice, 0=two-season veg)
    Q[i,t,s]   >= 0   total production (jin)
    u[i,t,s]   >= 0   full-price sales (jin)

  Constraints
    * area-activation:  x <= A_j y,  x >= eta * A_j y
    * area conservation per (j, season) for non-irrigated plots
    * irrigated mode互斥 (rice vs two-season veg, single root in s2)
    * 重茬 adjacency y_a + y_b <= 1 (incl. 2023 boundary)
    * rice year-to-year r_t + r_{t+1} <= 1 (incl. 2023 boundary)
    * rolling 3-year legume coverage >= A_j
    * Q = sum_j q * x,  0 <= u <= Q,  u <= lambda * D

  Objectives
    scenario 1 (waste):  max Z1 = sum p*u - sum c*x
    scenario 2 (half-price): max Z2 = sum [p*u + 0.5 p (Q-u)] - sum c*x
    lexicographic stage 2: min F = sum y  subject to  Z >= (1-delta) Z*
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse

from .preprocess import ModelData, LEGUME_CODES, RICE_CODE, VEG_CODES, ROOT_CODES, MUSHROOM_CODES, GRAIN_CODES


@dataclass
class MILPModel:
    # variable index maps
    x_idx: dict
    y_idx: dict
    r_idx: dict
    Q_idx: dict
    u_idx: dict
    n: int
    c: np.ndarray                  # objective (minimization)
    A_ub: sparse.csr_matrix
    b_ub: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray        # 1 = integer (binary via bounds), 0 = continuous
    # metadata
    scenario: int
    eta: float
    demand_scale: float
    stage: str
    z_star: float = None
    delta: float = 0.0
    # -Z coefficient per variable index (for lexicographic Z-constraint)
    neg_z_coefs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# variable indexing
# ---------------------------------------------------------------------------

def _build_indices(data: ModelData):
    """Return ordered key lists + index maps. Order: x, y, r, Q, u."""
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
    # (i, t, s) pairs that have at least one suitable plot
    is_pairs = sorted({(i, s) for (j, i, s) in data.suit.keys()})
    Q_keys, u_keys = [], []
    for t in data.years:
        for (i, s) in is_pairs:
            Q_keys.append((i, t, s))
            u_keys.append((i, t, s))
    x_idx = {k: k_idx for k_idx, k in enumerate(x_keys)}
    off = len(x_idx)
    y_idx = {k: off + k_idx for k_idx, k in enumerate(y_keys)}
    off += len(y_idx)
    r_idx = {k: off + k_idx for k_idx, k in enumerate(r_keys)}
    off += len(r_idx)
    Q_idx = {k: off + k_idx for k_idx, k in enumerate(Q_keys)}
    off += len(Q_idx)
    u_idx = {k: off + k_idx for k_idx, k in enumerate(u_keys)}
    n = off + len(u_idx)
    return (x_keys, y_keys, r_keys, Q_keys, u_keys,
            x_idx, y_idx, r_idx, Q_idx, u_idx, n)


# ---------------------------------------------------------------------------
# constraint builder
# ---------------------------------------------------------------------------

def _add(row, col, val, rows, cols, vals):
    rows.append(row); cols.append(col); vals.append(val)


def build_model(data: ModelData, scenario: int, eta: float = 0.5,
                demand_scale: float = 1.0, stage: str = "primary",
                z_star: float = None, delta: float = 0.0) -> MILPModel:
    """Assemble the sparse MILP. stage = 'primary' | 'lex'."""
    if scenario not in (1, 2):
        raise ValueError("scenario must be 1 or 2")
    if stage not in ("primary", "lex"):
        raise ValueError("stage must be 'primary' or 'lex'")

    (x_keys, y_keys, r_keys, Q_keys, u_keys,
     x_idx, y_idx, r_idx, Q_idx, u_idx, n) = _build_indices(data)

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
    # Q unbounded above
    # u upper bound = lambda * D
    for (i, t, s), k in u_idx.items():
        ub[k] = demand_scale * data.D.get((i, s), 0.0)

    # ---- objective & -Z coefficients ----
    neg_z = np.zeros(n)  # the -Z expression (for lex Z-constraint)
    # cost on x  (appears in -Z as +c, since Z = revenue - cost)
    for (j, i, t, s), k in x_idx.items():
        c[k] += data.c[(j, i, s)]
        neg_z[k] += data.c[(j, i, s)]
    # revenue on u
    # scenario 1:  Z = sum p*u - sum c*x                      -> u contributes p
    # scenario 2:  Z = sum [0.5p*u + 0.5p*Q] - sum c*x        -> u contributes 0.5p
    # (because surplus (Q-u) sells at 0.5p, so total = p*u + 0.5p*(Q-u) = 0.5p*u + 0.5p*Q)
    for (i, t, s), k in u_idx.items():
        p = data.p.get((i, s), 0.0)
        coef = p if scenario == 1 else 0.5 * p
        c[k] -= coef          # minimize -> negative of revenue
        neg_z[k] -= coef      # -Z uses same scenario-dependent coef
    # revenue on Q (scenario 2 half-price surplus)
    if scenario == 2:
        for (i, t, s), k in Q_idx.items():
            p = data.p.get((i, s), 0.0)
            c[k] -= 0.5 * p
            neg_z[k] -= 0.5 * p
    # lexicographic stage 2: objective becomes F = sum y
    if stage == "lex":
        c = np.zeros(n)
        for k in y_idx.values():
            c[k] = 1.0

    # ---- constraints (sparse COO accumulated then converted) ----
    ub_rows, ub_cols, ub_vals = [], [], []
    eq_rows, eq_cols, eq_vals = [], [], []
    b_ub_list, b_eq_list = [], []

    def _commit_ub():
        nonlocal ub_rows
        if not ub_rows:
            return sparse.csr_matrix((0, n))
        return sparse.csr_matrix(
            (np.array(ub_vals, dtype=float),
             (np.array(ub_rows, dtype=np.int64),
              np.array(ub_cols, dtype=np.int64))),
            shape=(len(b_ub_list), n))

    def _commit_eq():
        if not eq_rows:
            return sparse.csr_matrix((0, n))
        return sparse.csr_matrix(
            (np.array(eq_vals, dtype=float),
             (np.array(eq_rows, dtype=np.int64),
              np.array(eq_cols, dtype=np.int64))),
            shape=(len(b_eq_list), n))

    r_ub = -1  # current row counter (incremented before use)

    # 1. area-activation:  x - A_j y <= 0 ;  -x + eta A_j y <= 0
    for (j, i, t, s), k in x_idx.items():
        Aj = data.A[j]
        yk = y_idx[(j, i, t, s)]
        r_ub += 1
        _add(r_ub, k, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, yk, -Aj, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(0.0)
        r_ub += 1
        _add(r_ub, k, -1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, yk, eta * Aj, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(0.0)

    # 1b. implied cardinality cut:  sum_i y_{j,i,t,s} <= floor(1/eta)
    #     (follows from x >= eta*A_j*y and area conservation / mode total = A_j).
    #     Tightens the LP relaxation without changing the feasible region.
    max_y = int(np.floor(1.0 / eta + 1e-9)) if eta > 0 else 10 ** 9
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

    # 2. area conservation (non-irrigated) per (j, t, s): sum_i x = A_j
    r_eq = -1
    for t in data.years:
        groups = {}
        for (j, i, s) in sorted(data.suit.keys()):
            if data.plot_type[j] == "水浇地":
                continue   # handled by mode constraints
            groups.setdefault((j, s), []).append(i)
        for (j, s), ilist in groups.items():
            r_eq += 1
            for i in ilist:
                _add(r_eq, x_idx[(j, i, t, s)], 1.0, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(data.A[j])

    # 3. irrigated mode constraints (per (j,t) for 水浇地)
    for t in data.years:
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            rk = r_idx[(j, t)]
            Aj = data.A[j]
            # rice area: x[j,16,t,1] - A_j r = 0
            if (j, RICE_CODE, t, 1) in x_idx:
                r_eq += 1
                _add(r_eq, x_idx[(j, RICE_CODE, t, 1)], 1.0, eq_rows, eq_cols, eq_vals)
                _add(r_eq, rk, -Aj, eq_rows, eq_cols, eq_vals)
                b_eq_list.append(0.0)
            # veg s1 sum + A_j r = A_j
            r_eq += 1
            for i in VEG_CODES:
                if (j, i, t, 1) in x_idx:
                    _add(r_eq, x_idx[(j, i, t, 1)], 1.0, eq_rows, eq_cols, eq_vals)
            _add(r_eq, rk, Aj, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(Aj)
            # root s2 sum + A_j r = A_j
            r_eq += 1
            for i in ROOT_CODES:
                if (j, i, t, 2) in x_idx:
                    _add(r_eq, x_idx[(j, i, t, 2)], 1.0, eq_rows, eq_cols, eq_vals)
            _add(r_eq, rk, Aj, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(Aj)
            # exactly one root in s2 when veg mode: sum root y + r = 1
            r_eq += 1
            for i in ROOT_CODES:
                if (j, i, t, 2) in y_idx:
                    _add(r_eq, y_idx[(j, i, t, 2)], 1.0, eq_rows, eq_cols, eq_vals)
            _add(r_eq, rk, 1.0, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(1.0)

    # 4. 重茬 adjacency: y_a + y_b <= 1 (2023 boundary uses bar_y constant)
    for (j, i, (ta, sa), (tb, sb)) in data.adj_pairs:
        if ta == 2023 and tb == 2023:
            continue  # within 2023 (no decision variables) -- skip
        if ta == 2023:
            # y_a is bar_y_2023 constant
            bar = data.bar_y.get((j, i, sa), 0)
            if bar == 1:
                # y_b <= 0 -> fix y_b to 0 via ub
                kb = y_idx.get((j, i, tb, sb))
                if kb is not None:
                    ub[kb] = 0.0
            # bar == 0 -> no constraint (y_b <= 1 trivial)
            continue
        if tb == 2023:
            # symmetric boundary (shouldn't happen given ordering, but guard)
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
        _add(r_ub, kb, 1.0, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(1.0)

    # 5. rice year adjacency: r_t + r_{t+1} <= 1 (incl. 2023 boundary)
    for t_idx in range(len(data.years) - 1):
        t = data.years[t_idx]
        t1 = data.years[t_idx + 1]
        for j, ptype in enumerate(data.plot_type):
            if ptype != "水浇地":
                continue
            r_ub += 1
            _add(r_ub, r_idx[(j, t)], 1.0, ub_rows, ub_cols, ub_vals)
            _add(r_ub, r_idx[(j, t1)], 1.0, ub_rows, ub_cols, ub_vals)
            b_ub_list.append(1.0)
    # 2023 boundary for rice
    for j, ptype in enumerate(data.plot_type):
        if ptype != "水浇地":
            continue
        if data.r_2023.get(j, 0) == 1:
            # r_2023=1 -> r[2024] <= 0
            ub[r_idx[(j, data.years[0])]] = 0.0

    # 6. rolling 3-year legume coverage: sum x + hist >= A_j
    #    -> -sum x <= hist - A_j
    for (j, window, hist) in data.legume_windows:
        r_ub += 1
        for tau in window:
            if tau == 2023:
                continue  # history already in `hist`
            for s in data.plot_seasons[j]:
                for i in LEGUME_CODES:
                    k = x_idx.get((j, i, tau, s))
                    if k is not None:
                        _add(r_ub, k, -1.0, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(hist - data.A[j])

    # 7. production: Q = sum_j q * x
    # group x by (i, t, s)
    prod_groups = {}
    for (j, i, t, s), k in x_idx.items():
        prod_groups.setdefault((i, t, s), []).append((j, k))
    for (i, t, s), jks in prod_groups.items():
        qk = Q_idx.get((i, t, s))
        if qk is None:
            continue
        r_eq += 1
        _add(r_eq, qk, 1.0, eq_rows, eq_cols, eq_vals)
        for (j, k) in jks:
            _add(r_eq, k, -data.q[(j, i, s)], eq_rows, eq_cols, eq_vals)
        b_eq_list.append(0.0)

    # 8. sales: u - Q <= 0
    for (i, t, s), uk in u_idx.items():
        qk = Q_idx.get((i, t, s))
        if qk is None:
            continue
        r_ub += 1
        _add(r_ub, uk, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, qk, -1.0, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(0.0)

    # 9. lexicographic Z-constraint:  Z >= (1-delta) Z*  ->  -Z <= -(1-delta) Z*
    if stage == "lex" and z_star is not None:
        r_ub += 1
        for k, coef in enumerate(neg_z):
            if coef != 0.0:
                _add(r_ub, k, coef, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(-(1.0 - delta) * z_star)

    A_ub = _commit_ub()
    A_eq = _commit_eq()
    b_ub = np.array(b_ub_list, dtype=float)
    b_eq = np.array(b_eq_list, dtype=float)

    return MILPModel(
        x_idx=x_idx, y_idx=y_idx, r_idx=r_idx, Q_idx=Q_idx, u_idx=u_idx, n=n,
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub, integrality=integrality,
        scenario=scenario, eta=eta, demand_scale=demand_scale,
        stage=stage, z_star=z_star, delta=delta,
        neg_z_coefs={k: float(v) for k, v in enumerate(neg_z) if v != 0.0},
    )
