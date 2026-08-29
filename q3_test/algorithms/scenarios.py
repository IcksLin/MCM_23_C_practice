# -*- coding: utf-8 -*-
"""Q3情景生成 — 边际LHS → t-Copula相关重排 → 交叉价格弹性修正。

流水线:
  1. generate_q2_marginals: LHS抽样生成原始边际（需求/产量/成本/价格）
  2. correlate_marginals: 用t-Copula秩分数重排LHS列，引入相关结构
  3. apply_market_interactions: 用交叉价格弹性修正需求

Q3边际范围（doc/Q3_尝试解答.md section 4.1）:
  小麦/玉米销量: 年增长率[5%,10%]逐年递推
  其他作物销量: 相对2023年[-5%,5%]不累积
  亩产量: 相对2023年[-10%,10%]
  成本: 年增长率[4%,6%]逐年递推
  粮食价格: 当年趋势基线附近[-1%,1%]不累积
  蔬菜价格: 年增长率[4%,6%]逐年递推
  食用菌价格: 年下降率[1%,5%]逐年递推
  羊肚菌价格: 固定年下降5%

作者: Q3编程手
来源: 复制自q2_test/algorithms/scenarios.py并扩展，依据AGENT.md section 3.6
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .preprocess import (
    ModelData, LEGUME_CODES, RICE_CODE, GRAIN_CODES,
    VEG_CODES, ROOT_CODES, MUSHROOM_CODES, MOREL_CODE,
)


@dataclass
class MarginalScenarioSet:
    """Q3边际LHS样本（尚未相关）。"""
    demand: dict          # (i,t,s) -> array(N,)
    yield_: dict          # (j,i,t,s) -> array(N,)
    cost: dict            # (j,i,t,s) -> array(N,)
    price: dict           # (i,t,s) -> array(N,)
    trend_price: dict     # (i,t,s) -> array(N,)  Q2确定趋势价格
    n: int
    factor_scores: np.ndarray | None = None
    dependency_audit: object | None = None
    audit_values: np.ndarray | None = None
    audit_target_kendall: np.ndarray | None = None
    audit_pairs: np.ndarray | None = None
    audit_labels: list | None = None


@dataclass
class Q3ScenarioSet:
    """Q3完整情景集（相关+弹性修正后）。"""
    weights: np.ndarray           # (N,) 均匀权重
    demand: dict                   # (i,t,s) -> array(N,)  弹性修正后
    demand_base: dict              # (i,t,s) -> array(N,)  弹性修正前
    yield_: dict                   # (j,i,t,s) -> array(N,)
    cost: dict                     # (j,i,t,s) -> array(N,)
    price: dict                    # (i,t,s) -> array(N,)
    trend_price: dict              # (i,t,s) -> array(N,)
    n: int
    # 因子分数（供情景缩减使用）
    factor_scores: np.ndarray | None = None  # (N, n_dims*7)
    macro_factor: np.ndarray | None = None    # (N, 7)
    climate_factor: np.ndarray | None = None  # (N, 7)
    dependency_audit: object | None = None
    audit_values: np.ndarray | None = None
    audit_target_kendall: np.ndarray | None = None
    audit_pairs: np.ndarray | None = None
    audit_labels: list | None = None

    @property
    def k(self) -> int:
        """兼容固定方案评估接口；原始样本外情景不经缩减。"""
        return self.n


@dataclass
class ScenarioDependencyAudit:
    """原始情景相关门禁（在预先固定的小型审计对集上计算）。"""
    min_eigenvalue: float
    max_kendall_error: float
    pair_count: int
    labels: list = field(default_factory=list)

def _lhs_unit(n: int, lo: float = 0.0, hi: float = 1.0,
              seed: int = 2024) -> np.ndarray:
    """标准LHS抽样 [0,1]^n。"""
    rng = np.random.default_rng(seed)
    result = np.empty(n)
    perm = rng.permutation(n)
    for i in range(n):
        result[perm[i]] = (i + rng.uniform(0, 1)) / n
    return lo + (hi - lo) * result


def _lhs_sample(n: int, lo: float, hi: float, seed: int = 2024,
                distribution: str = "uniform") -> np.ndarray:
    """LHS抽样，支持uniform/triangular。"""
    if distribution == "triangular":
        u = _lhs_unit(n, 0, 1, seed)
        mid = (lo + hi) / 2.0
        # triangular CDF逆变换
        fc = (mid - lo) / (hi - lo)
        result = np.where(u < fc,
                         lo + np.sqrt(u * (hi - lo) * (mid - lo)),
                         hi - np.sqrt((1 - u) * (hi - lo) * (hi - mid)))
        return result
    return _lhs_unit(n, lo, hi, seed)


def generate_q2_marginals(data: ModelData, n: int = 1000,
                          seed: int = 2024,
                          distribution: str = "uniform") -> MarginalScenarioSet:
    """生成Q3边际LHS样本。

    遵循Q3_尝试解答.md section 4.1的边际范围：
      - 小麦/玉米销量: 年增长率[5%,10%]逐年递推（累积）
      - 其他作物销量: 相对2023年[-5%,5%]不累积
      - 亩产量: 相对2023年[-10%,10%]
      - 成本: 年增长率[4%,6%]逐年递推
      - 粮食价格: 当年趋势附近[-1%,1%]不累积
      - 蔬菜价格: 年增长率[4%,6%]逐年递推
      - 食用菌价格: 年下降率[1%,5%]逐年递推
      - 羊肚菌价格: 固定年下降5%
    """
    rng = np.random.default_rng(seed)
    years = data.years
    is_pairs = sorted({(i, s) for (j, i, s) in data.suit.keys()})

    demand = {}
    yield_ = {}
    cost = {}
    price = {}
    trend_price = {}

    # ---- 需求生成 ----
    for (i, s) in is_pairs:
        d_base = data.D.get((i, s), 0.0)
        is_cereal = i in (6, 7)  # 小麦=6, 玉米=7
        cereal_path = np.full(n, d_base)
        for t_idx, t in enumerate(years):
            if is_cereal:
                # 年增长率5%-10%，逐年递推
                growth_rates = _lhs_sample(n, 0.05, 0.10, seed + i * 100 + t, distribution)
                cereal_path = cereal_path * (1.0 + growth_rates)
                demand[(i, t, s)] = cereal_path.copy()
            else:
                # 相对2023年[-5%,5%]不累积
                shock = _lhs_sample(n, -0.05, 0.05, seed + i * 100 + t, distribution)
                demand[(i, t, s)] = d_base * (1.0 + shock)

    # ---- 亩产生成 ----
    for (j, i, s) in sorted(data.suit.keys()):
        q_base = data.q.get((j, i, s), 0.0)
        for t in years:
            shock = _lhs_sample(n, -0.10, 0.10, seed + j * 1000 + i * 100 + t, distribution)
            yield_[(j, i, t, s)] = q_base * (1.0 + shock)

    # ---- 成本生成 ----
    for (j, i, s) in sorted(data.suit.keys()):
        c_base = data.c.get((j, i, s), 0.0)
        c_path = np.full(n, c_base)
        for t_idx, t in enumerate(years):
            growth = _lhs_sample(n, 0.04, 0.06, seed + j * 2000 + i * 100 + t, distribution)
            c_path = c_path * (1.0 + growth)
            cost[(j, i, t, s)] = c_path.copy()

    # ---- 价格生成 ----
    for (i, s) in is_pairs:
        p_base = data.p.get((i, s), 0.0)
        group = _crop_price_group_q3(i)
        p_path = np.full(n, p_base)
        for t_idx, t in enumerate(years):
            if group == "grain":
                # 粮食价格：[-1%,1%]不累积
                shock = _lhs_sample(n, -0.01, 0.01, seed + i * 300 + t, distribution)
                p_vals = p_base * (1.0 + shock)
            elif group == "vegetable":
                # 蔬菜价格：年增长率[4%,6%]逐年递推
                growth = _lhs_sample(n, 0.04, 0.06, seed + i * 300 + t, distribution)
                p_path = p_path * (1.0 + growth)
                p_vals = p_path.copy()
            elif group == "mushroom":
                # 食用菌价格：年下降率[1%,5%]逐年递推
                decline = _lhs_sample(n, 0.01, 0.05, seed + i * 300 + t, distribution)
                p_path = p_path * (1.0 - decline)
                p_vals = p_path.copy()
            elif group == "morel":
                # 羊肚菌：固定年下降5%
                p_vals = p_base * (0.95 ** (t_idx + 1))
                p_vals = np.full(n, p_vals)  # 确定值
            else:
                p_vals = np.full(n, p_base)
            price[(i, t, s)] = p_vals
            if group == "vegetable":
                p_trend = p_base * 1.05 ** (t_idx + 1)
            elif group == "mushroom":
                p_trend = p_base * 0.97 ** (t_idx + 1)
            elif group == "morel":
                p_trend = p_base * 0.95 ** (t_idx + 1)
            else:
                p_trend = p_base
            trend_price[(i, t, s)] = np.full(n, p_trend)

    return MarginalScenarioSet(
        demand=demand, yield_=yield_, cost=cost, price=price,
        trend_price=trend_price, n=n,
    )


def _crop_price_group_q3(crop_code: int) -> str:
    """Q3作物价格分组。"""
    if crop_code in GRAIN_CODES or crop_code == RICE_CODE:
        return "grain"
    if crop_code == MOREL_CODE:
        return "morel"
    if crop_code in MUSHROOM_CODES:
        return "mushroom"
    return "vegetable"


def correlate_marginals(marginals: MarginalScenarioSet,
                         dependency_cfg=None,
                         data: ModelData = None,
                         seed: int = 2024) -> MarginalScenarioSet:
    """用t-Copula秩分数重排LHS列，引入相关结构。

    依赖dependency模块的generate_t_rank_scores和reorder_lhs_by_ranks。
    如果dependency模块不可用，返回原始样本（在日志中记录）。

    Args:
      marginals: 边际LHS样本
      dependency_cfg: DependencyConfig (可选)
      data: ModelData (用于构建因子映射)
      seed: 随机种子
    """
    from .dependency import DependencyConfig, reorder_lhs_by_ranks, BASE_LOADINGS

    if dependency_cfg is None:
        dependency_cfg = DependencyConfig(
            df=5, correlation_scale=1.0, temporal_rho=0.5,
            loadings=BASE_LOADINGS,
        )
    if data is None:
        return marginals

    # 构建因子映射（每作物类别一个）
    crop_groups = getattr(data, "crop_group", {})
    if not crop_groups:
        return marginals

    years = data.years
    groups = sorted(set(crop_groups.values()))
    dim_for_dict = {
        "demand": "base_demand_change", "yield_": "yield_change",
        "cost": "cost_growth", "price": "price_change",
    }
    factors = ["macro", "global_climate"]
    for group in groups:
        factors.extend([f"{group}:category_demand",
                        f"{group}:category_climate"])
    rng = np.random.default_rng(seed)
    nf, nt, n = len(factors), len(years), marginals.n
    rho = float(dependency_cfg.temporal_rho)
    g = np.empty((nf, nt, n), dtype=float)
    g[:, 0, :] = rng.standard_normal((nf, n))
    for ti in range(1, nt):
        g[:, ti, :] = (rho * g[:, ti - 1, :]
                       + np.sqrt(1.0 - rho * rho)
                       * rng.standard_normal((nf, n)))
    if np.isinf(dependency_cfg.df) or dependency_cfg.df > 1e6:
        scale = np.ones(n)
    else:
        scale = np.sqrt(rng.chisquare(dependency_cfg.df, n)
                        / dependency_cfg.df)

    # 每个“作物组×随机维度×年份”冻结一个真实边际列作为代表。
    audit_by_tag = {}
    for dict_name, key_dict in (("demand", marginals.demand),
                                ("yield_", marginals.yield_),
                                ("cost", marginals.cost),
                                ("price", marginals.price)):
        dim = dim_for_dict[dict_name]
        for key in sorted(key_dict):
            if len(key) == 3:
                i, t, _ = key
            else:
                _, i, t, _ = key
            group = crop_groups[i]
            loads = dependency_cfg.loadings[dim]
            coeff = np.zeros(nf)
            coeff[factors.index("macro")] = loads["macro"]
            coeff[factors.index("global_climate")] = loads["global_climate"]
            coeff[factors.index(f"{group}:category_demand")] = loads["category_demand"]
            coeff[factors.index(f"{group}:category_climate")] = loads["category_climate"]
            coeff *= dependency_cfg.correlation_scale
            row_sq = float(coeff @ coeff)
            if row_sq >= 1.0 - 1e-12:
                raise ValueError("联合相关载荷行平方和必须小于1")
            ti = years.index(t)
            # 每个实际随机边际拥有独立 epsilon；只共享因子与卡方尺度。
            z = (coeff @ g[:, ti, :]
                 + np.sqrt(1.0 - row_sq) * rng.standard_normal(n)) / scale
            key_dict[key] = reorder_lhs_by_ranks(
                key_dict[key].reshape(-1, 1), z.reshape(-1, 1)).ravel()
            tag = (group, dim, int(t))
            if tag not in audit_by_tag:
                audit_by_tag[tag] = (key, key_dict[key].copy(), coeff.copy(), ti)

    # 真实样本 Kendall 审计，不得硬编码通过。
    from scipy.stats import kendalltau
    tags = sorted(audit_by_tag)
    audit_values = np.column_stack([audit_by_tag[tag][1] for tag in tags])
    audit_coeffs = [audit_by_tag[tag][2] for tag in tags]
    audit_years = [audit_by_tag[tag][3] for tag in tags]
    audit_labels = [(tag[0], tag[1], tag[2], audit_by_tag[tag][0])
                    for tag in tags]
    m = len(tags)
    target_r = np.eye(m)
    max_err = 0.0
    pair_count = 0
    for a in range(m):
        for b in range(a + 1, m):
            r_ab = float(audit_coeffs[a] @ audit_coeffs[b]) * rho ** abs(
                audit_years[a] - audit_years[b])
            target_r[a, b] = target_r[b, a] = r_ab
    target_tau = 2.0 / np.pi * np.arcsin(np.clip(target_r, -1.0, 1.0))
    audit_pair_set = set()
    for a in range(m):
        for b in range(a + 1, m):
            if abs(target_tau[a, b]) >= 0.10 - 1e-12:
                audit_pair_set.add((a, b))
            same_series = (tags[a][0] == tags[b][0]
                           and tags[a][1] == tags[b][1]
                           and abs(tags[a][2] - tags[b][2]) == 1)
            if same_series:
                audit_pair_set.add((a, b))
    audit_pairs = sorted(audit_pair_set)
    for a, b in audit_pairs:
        tau_target = target_tau[a, b]
        tau_sample = float(kendalltau(audit_values[:, a],
                                      audit_values[:, b]).statistic)
        max_err = max(max_err, abs(tau_sample - tau_target))
        pair_count += 1
    min_eig = float(np.linalg.eigvalsh(target_r).min()) if m else 1.0
    marginals.dependency_audit = ScenarioDependencyAudit(
        min_eigenvalue=min_eig, max_kendall_error=max_err,
        pair_count=pair_count, labels=audit_labels)
    marginals.audit_values = audit_values
    marginals.audit_target_kendall = target_tau
    marginals.audit_pairs = np.asarray(audit_pairs, dtype=int).reshape(-1, 2)
    marginals.audit_labels = audit_labels
    # 缩减特征仅保留公共因子，避免存储所有边际特异噪声。
    marginals.factor_scores = (g / scale[None, None, :]).transpose(2, 0, 1).reshape(n, -1)

    return marginals


def apply_market_interactions(marginals: MarginalScenarioSet,
                               data: ModelData,
                               elasticity_cfg=None) -> Q3ScenarioSet:
    """应用交叉价格弹性修正需求。

    D_its = D_base_its * exp(sum_h e_ih * ln(p_h / bar_p_h))

    Args:
      marginals: 相关后的边际样本
      data: ModelData
      elasticity_cfg: ElasticityConfig (可选，None则不修正)
    """
    if elasticity_cfg is not None:
        from .elasticity import (build_elasticity_matrix,
                                 apply_price_elasticity,
                                 build_crop_indices)
        matrices = build_elasticity_matrix(data, config=elasticity_cfg)
    else:
        matrices = None

    n = marginals.n
    weights = np.full(n, 1.0 / n)
    demand_base = {k: v.copy() for k, v in marginals.demand.items()}
    demand = {k: v.copy() for k, v in marginals.demand.items()}

    # 应用弹性修正
    if matrices is not None:
        crop_indices_by_season = build_crop_indices(data)
        for s, matrix in matrices.items():
            crop_indices = crop_indices_by_season[s]
            crops = [c for c, _ in sorted(crop_indices.items(), key=lambda z: z[1])]
            for t in data.years:
                base_mat = np.column_stack([demand_base[(i, t, s)] for i in crops])
                price_mat = np.column_stack([marginals.price[(i, t, s)] for i in crops])
                trend_mat = np.column_stack([marginals.trend_price[(i, t, s)] for i in crops])
                modified = apply_price_elasticity(
                    base_mat, price_mat, trend_mat, matrices, crop_indices, s)
                for col, i in enumerate(crops):
                    demand[(i, t, s)] = modified[:, col]

    return Q3ScenarioSet(
        weights=weights, demand=demand, demand_base=demand_base,
        yield_=marginals.yield_, cost=marginals.cost, price=marginals.price,
        trend_price=marginals.trend_price, n=n,
        factor_scores=marginals.factor_scores,
        dependency_audit=marginals.dependency_audit,
        audit_values=marginals.audit_values,
        audit_target_kendall=marginals.audit_target_kendall,
        audit_pairs=marginals.audit_pairs,
        audit_labels=marginals.audit_labels,
    )


def generate_raw_scenarios(data: ModelData, n: int = 1000, seed: int = 2024,
                           distribution: str = "uniform",
                           dependency_cfg=None,
                           elasticity_cfg=None) -> Q3ScenarioSet:
    """完整Q3情景生成流水线。

    1. 生成边际LHS
    2. t-Copula相关重排
    3. 交叉价格弹性修正
    """
    marginals = generate_q2_marginals(data, n, seed, distribution)
    marginals = correlate_marginals(marginals, dependency_cfg, data, seed)
    return apply_market_interactions(marginals, data, elasticity_cfg)


def compute_proxy_profit(scenarios: Q3ScenarioSet, data: ModelData,
                          plan_x: dict, scenario_idx: int = 0,
                          gamma: float = 0.03) -> float:
    """计算固定方案在特定情景下的代理利润。"""
    profit = 0.0
    # 收入 = sum p * min(Q, D)
    for (j, i, t, s), area in plan_x.items():
        if area <= 0:
            continue
        q = scenarios.yield_.get((j, i, t, s), np.zeros(scenarios.n))[scenario_idx]
        c = scenarios.cost.get((j, i, t, s), np.zeros(scenarios.n))[scenario_idx]
        p = scenarios.price.get((i, t, s), np.zeros(scenarios.n))[scenario_idx]
        d = scenarios.demand.get((i, t, s), np.zeros(scenarios.n))[scenario_idx]
        # 产量（含互补增益，简化版）
        Q = q * area
        u = min(Q, d)
        profit += p * u - c * area
    return profit


def scenario_to_dataframe(scenarios: Q3ScenarioSet) -> "pd.DataFrame":
    """将情景集转换为长表（供审计和导出）。"""
    import pandas as pd
    rows = []
    for (i, t, s), vals in scenarios.demand.items():
        for omega, v in enumerate(vals):
            rows.append({"scenario": omega, "crop": i, "year": t,
                        "season": s, "demand": v})
    return pd.DataFrame(rows)
