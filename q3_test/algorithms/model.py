# -*- coding: utf-8 -*-
"""Q3 随机MILP装配 — 含豆类前茬互补增益的均值-CVaR模型。

继承Q2全部变量和硬约束，新增：
  b[j,t] in {0,1}      上一年度是否种过豆类
  w[j,i,t,s] >= 0      互补面积线性化变量 (w = x * b)

产量约束修改：
  Q_omega = sum_j q_omega * (x + gamma_i * w)
  注意：gamma是每作物的增产率，乘在w上，不能写成 q*x + gamma*w

变量顺序: x, y, r, b, w, Q, u, xi, zeta

作者: Q3编程手
来源: 复制自q2_test/algorithms/model.py并扩展，依据AGENT.md section 3.8
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import sparse

from .preprocess import ModelData, LEGUME_CODES, RICE_CODE, VEG_CODES, ROOT_CODES
from .scenario_reduction import ReducedScenarioSet


@dataclass
class StochModel:
    """Q3随机MILP模型数据结构。"""
    # 变量索引映射
    x_idx: dict           # (j,i,t,s) -> int
    y_idx: dict           # (j,i,t,s) -> int
    r_idx: dict           # (j,t) -> int
    b_idx: dict           # (j,t) -> int  豆类前茬指示
    w_idx: dict           # (j,i,t,s) -> int  互补面积
    Q_idx: dict           # (omega,i,t,s) -> int
    u_idx: dict           # (omega,i,t,s) -> int
    xi_idx: dict          # (omega,) -> int
    n: int
    # MILP数据
    c: np.ndarray
    A_ub: sparse.csr_matrix
    b_ub: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integrality: np.ndarray
    # 元数据
    eta: float
    beta: float
    risk_lambda: float
    gamma: float              # 豆类前茬增产率
    stage: str
    z_star: float = None
    e_star: float = None
    eps_z: float = 1e-6
    eps_e: float = 1e-6
    neg_z_coefs: dict = field(default_factory=dict)
    neg_e_coefs: dict = field(default_factory=dict)


def _add(row, col, val, rows, cols, vals):
    """向稀疏矩阵三元组添加元素。"""
    rows.append(row); cols.append(col); vals.append(val)


def _build_indices(data: ModelData, scenarios: ReducedScenarioSet):
    """构建变量索引映射。顺序: x, y, r, b, w, Q, u, xi, zeta。"""
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
    # b: 每个地块每年一个（仅非豆类作物可用的地块）
    b_keys = []
    for t in data.years:
        for j in range(len(data.plot_names)):
            b_keys.append((j, t))
    # w: 与x同键（仅非豆类作物有互补增益）
    w_keys = []
    for t in data.years:
        for (j, i, s) in sorted(data.suit.keys()):
            if i not in LEGUME_CODES:
                w_keys.append((j, i, t, s))
    # Q和u: 情景索引
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
    b_idx = {k: off + i for i, k in enumerate(b_keys)}
    off += len(b_idx)
    w_idx = {k: off + i for i, k in enumerate(w_keys)}
    off += len(w_idx)
    Q_idx = {k: off + i for i, k in enumerate(Q_keys)}
    off += len(Q_idx)
    u_idx = {k: off + i for i, k in enumerate(u_keys)}
    off += len(u_idx)
    xi_idx = {k: off + i for i, k in enumerate(xi_keys)}
    off += len(xi_idx)
    zeta_idx = off
    n = off + 1
    return (x_keys, y_keys, r_keys, b_keys, w_keys, Q_keys, u_keys, xi_keys,
            x_idx, y_idx, r_idx, b_idx, w_idx, Q_idx, u_idx, xi_idx, zeta_idx, n)


def build_q3_model(data: ModelData, scenarios: ReducedScenarioSet,
                   beta: float = 0.90, risk_lambda: float = 0.5,
                   eta: float = 0.5, gamma: float = 0.03,
                   stage: str = "risk",
                   z_star: float = None, e_star: float = None,
                   eps_z: float = 1e-6, eps_e: float = 1e-6) -> StochModel:
    """装配Q3稀疏随机MILP。

    Args:
      data: ModelData from preprocess.
      scenarios: ReducedScenarioSet from scenario_reduction.
      beta: CVaR置信水平。
      risk_lambda: 风险厌恶参数 [0,1]。
      eta: 最小面积比例。
      gamma: 豆类前茬增产率（非豆类作物）。
      stage: "risk" | "expected" | "fragment"。
      z_star: Stage 1最优Z_lambda（用于Stage 2/3）。
      e_star: Stage 2最优E[Pi]（用于Stage 3）。
      eps_z, eps_e: 字典序容差。
    """
    (x_keys, y_keys, r_keys, b_keys, w_keys, Q_keys, u_keys, xi_keys,
     x_idx, y_idx, r_idx, b_idx, w_idx, Q_idx, u_idx, xi_idx, zeta_idx, n
     ) = _build_indices(data, scenarios)

    K = scenarios.k
    wts = scenarios.weights  # (K,) 情景权重

    c = np.zeros(n)
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    integrality = np.zeros(n, dtype=int)

    # ---- 变量边界 ----
    for (j, i, t, s), k in x_idx.items():
        ub[k] = data.A[j]
    for k in y_idx.values():
        lb[k] = 0.0; ub[k] = 1.0; integrality[k] = 1
    for k in r_idx.values():
        lb[k] = 0.0; ub[k] = 1.0; integrality[k] = 1
    for k in b_idx.values():
        lb[k] = 0.0; ub[k] = 1.0; integrality[k] = 1
    for (j, i, t, s), k in w_idx.items():
        ub[k] = data.A[j]  # w <= A_j
    lb[zeta_idx] = -np.inf  # zeta自由

    # ---- 目标系数 ----
    # Z_lambda = (1-lam)*E[Pi] + lam*LCVaR
    # E[Pi] = sum_omega w_omega * (sum p*u - sum c*x)
    # LCVaR = zeta - 1/(1-beta) * sum w * xi

    neg_z = np.zeros(n)   # -Z_lambda系数
    neg_e = np.zeros(n)   # -E[Pi]系数
    lam = risk_lambda

    # x上的成本：加权平均成本
    for (j, i, t, s), k in x_idx.items():
        avg_cost = 0.0
        for omega in range(K):
            avg_cost += wts[omega] * scenarios.cost.get((j, i, t, s), np.zeros(K))[omega]
        c[k] += (1.0 - lam) * avg_cost
        neg_z[k] += (1.0 - lam) * avg_cost
        neg_e[k] += avg_cost

    # w上的成本增量：gamma * q * w 增加产量，不直接产生成本
    # 但w影响Q（产量），进而影响u（销量）和利润
    # 产量约束中 Q = q*(x + gamma*w)，所以w的收益通过u间接体现
    # 在目标函数中，w不直接出现（通过Q→u→利润间接影响）

    # u上的收入：加权平均价格
    for (omega, i, t, s), uk in u_idx.items():
        p_omega = scenarios.price.get((i, t, s), np.zeros(K))[omega]
        c[uk] -= (1.0 - lam) * wts[omega] * p_omega
        neg_z[uk] -= (1.0 - lam) * wts[omega] * p_omega
        neg_e[uk] -= wts[omega] * p_omega

    # CVaR项
    c[zeta_idx] -= lam
    neg_z[zeta_idx] -= lam
    for (omega,), xk in xi_idx.items():
        c[xk] += lam / (1.0 - beta) * wts[omega]
        neg_z[xk] += lam / (1.0 - beta) * wts[omega]

    # 阶段覆盖
    if stage == "expected":
        c = neg_e.copy()
    elif stage == "fragment":
        c = np.zeros(n)
        for k in y_idx.values():
            c[k] = 1.0

    # ---- 约束 ----
    ub_rows, ub_cols, ub_vals = [], [], []
    eq_rows, eq_cols, eq_vals = [], [], []
    b_ub_list, b_eq_list = [], []
    r_ub = -1
    r_eq = -1

    # === 硬约束1-6（继承Q1/Q2，在x/y/r上）===

    # 1. 面积-激活: x - A*y <= 0, -x + eta*A*y <= 0
    for (j, i, t, s), k in x_idx.items():
        Aj = data.A[j]
        yk = y_idx[(j, i, t, s)]
        r_ub += 1; _add(r_ub, k, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, yk, -Aj, ub_rows, ub_cols, ub_vals); b_ub_list.append(0.0)
        r_ub += 1; _add(r_ub, k, -1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, yk, eta * Aj, ub_rows, ub_cols, ub_vals); b_ub_list.append(0.0)

    # 1b. 基数约束
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

    # 2. 面积守恒（非水浇地）
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

    # 3. 水浇地模式约束
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

    # 4. 重茬邻接
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

    # 5. 水稻年邻接
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

    # 6. 滚动三年豆科覆盖
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

    # === Q3新增约束: 豆类前茬互补 ===

    # 7. b[j,t] = OR(legume y's from year t-1)
    #    b[j,t] >= y[j,i,t-1,s] for each legume i, season s
    #    b[j,t] <= sum(y[j,i,t-1,s]) for legumes (using upper bound)
    #    For t=2024 (first year): use 2023 history
    years = data.years
    for yi, t in enumerate(years):
        for j in range(len(data.plot_names)):
            bk = b_idx[(j, t)]
            if yi == 0:
                # 2024: use 2023 history
                had_legume = 0
                for s in data.plot_seasons[j]:
                    for i in LEGUME_CODES:
                        if data.bar_y.get((j, i, s), 0) == 1:
                            had_legume = 1
                            break
                    if had_legume:
                        break
                # Fix b to historical value
                if had_legume:
                    lb[bk] = 1.0
                else:
                    ub[bk] = 0.0
            else:
                # t > 2024: b[j,t] >= y[j,i,t-1,s] for each legume
                t_prev = years[yi - 1]
                legume_y_keys = []
                for s in data.plot_seasons[j]:
                    for i in LEGUME_CODES:
                        yk = y_idx.get((j, i, t_prev, s))
                        if yk is not None:
                            legume_y_keys.append(yk)
                if legume_y_keys:
                    # b >= each legume y
                    for yk in legume_y_keys:
                        r_ub += 1
                        _add(r_ub, bk, 1.0, ub_rows, ub_cols, ub_vals)
                        _add(r_ub, yk, -1.0, ub_rows, ub_cols, ub_vals)
                        b_ub_list.append(0.0)
                    # b <= sum(legume y's): b - sum(y) <= 0
                    r_ub += 1
                    _add(r_ub, bk, 1.0, ub_rows, ub_cols, ub_vals)
                    for yk in legume_y_keys:
                        _add(r_ub, yk, -1.0, ub_rows, ub_cols, ub_vals)
                    b_ub_list.append(0.0)
                else:
                    # No legume suitable for this plot
                    ub[bk] = 0.0

    # 8. w线性化: w = x * b (McCormick envelope, exact for binary b)
    #    w <= x
    #    w <= A_j * b
    #    w >= x - A_j * (1 - b)
    #    w >= 0 (already in bounds)
    for (j, i, t, s), wk in w_idx.items():
        xk = x_idx[(j, i, t, s)]
        bk = b_idx[(j, t)]
        Aj = data.A[j]
        # w <= x
        r_ub += 1
        _add(r_ub, wk, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, xk, -1.0, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(0.0)
        # w <= A_j * b
        r_ub += 1
        _add(r_ub, wk, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, bk, -Aj, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(0.0)
        # w >= x - A_j * (1 - b) -> -w + x - A_j*b <= A_j
        r_ub += 1
        _add(r_ub, wk, -1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, xk, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, bk, -Aj, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(Aj)

    # === 情景约束 ===

    # 9. 产量: Q_omega = sum_j q_omega * (x + gamma_i * w)
    #    豆类作物gamma=0，所以只有非豆类有w项
    prod_groups = {}
    for (j, i, t, s), k in x_idx.items():
        prod_groups.setdefault((i, t, s), []).append((j, k))
    for (i, t, s), jks in prod_groups.items():
        # gamma_i: 非豆类用配置的gamma，豆类为0
        gamma_i = gamma if i not in LEGUME_CODES else 0.0
        for omega in range(K):
            qk = Q_idx.get((omega, i, t, s))
            if qk is None:
                continue
            r_eq += 1
            _add(r_eq, qk, 1.0, eq_rows, eq_cols, eq_vals)
            for (j, k) in jks:
                q_omega = scenarios.yield_.get((j, i, t, s), np.zeros(K))[omega]
                _add(r_eq, k, -q_omega, eq_rows, eq_cols, eq_vals)
                # gamma * q * w 项（仅非豆类）
                if gamma_i > 0:
                    wk = w_idx.get((j, i, t, s))
                    if wk is not None:
                        _add(r_eq, wk, -gamma_i * q_omega, eq_rows, eq_cols, eq_vals)
            b_eq_list.append(0.0)

    # 10. 销量: u <= Q, u <= D
    for (omega, i, t, s), uk in u_idx.items():
        qk = Q_idx.get((omega, i, t, s))
        if qk is not None:
            r_ub += 1
            _add(r_ub, uk, 1.0, ub_rows, ub_cols, ub_vals)
            _add(r_ub, qk, -1.0, ub_rows, ub_cols, ub_vals); b_ub_list.append(0.0)
        d_omega = scenarios.demand.get((i, t, s), np.zeros(K))[omega]
        if d_omega > 0:
            ub[uk] = min(ub[uk], d_omega)
        else:
            ub[uk] = 0.0

    # 11. CVaR excess: xi >= zeta - Pi
    #     zeta - xi - Pi <= 0
    #     Pi = sum p*u - sum c*x  (注意：Q3的Pi不直接包含w，w通过Q间接影响)
    for (omega,), xk in xi_idx.items():
        r_ub += 1
        _add(r_ub, zeta_idx, 1.0, ub_rows, ub_cols, ub_vals)
        _add(r_ub, xk, -1.0, ub_rows, ub_cols, ub_vals)
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

    # === 字典序约束 ===

    # 12. Stage 2: Z_lambda >= Z* - eps
    if stage in ("expected", "fragment") and z_star is not None:
        r_ub += 1
        for k, coef in enumerate(neg_z):
            if coef != 0.0:
                _add(r_ub, k, coef, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(-(z_star - eps_z))

    # 13. Stage 3: E[Pi] >= E* - eps
    if stage == "fragment" and e_star is not None:
        r_ub += 1
        for k, coef in enumerate(neg_e):
            if coef != 0.0:
                _add(r_ub, k, coef, ub_rows, ub_cols, ub_vals)
        b_ub_list.append(-(e_star - eps_e))

    # ---- 装配矩阵 ----
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
        x_idx=x_idx, y_idx=y_idx, r_idx=r_idx, b_idx=b_idx, w_idx=w_idx,
        Q_idx=Q_idx, u_idx=u_idx, xi_idx=xi_idx, n=n,
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        lb=lb, ub=ub, integrality=integrality,
        eta=eta, beta=beta, risk_lambda=risk_lambda, gamma=gamma,
        stage=stage, z_star=z_star, e_star=e_star, eps_z=eps_z, eps_e=eps_e,
        neg_z_coefs={k: float(v) for k, v in enumerate(neg_z) if v != 0.0},
        neg_e_coefs={k: float(v) for k, v in enumerate(neg_e) if v != 0.0},
    )
