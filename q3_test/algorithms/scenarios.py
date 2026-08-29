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
        for t_idx, t in enumerate(years):
            if is_cereal:
                # 年增长率5%-10%，逐年递推
                growth_rates = _lhs_sample(n, 0.05, 0.10, seed + i * 100 + t, distribution)
                d_vals = np.full(n, d_base)
                for k in range(t_idx + 1):
                    d_vals = d_vals * (1.0 + growth_rates)
                demand[(i, t, s)] = d_vals
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
        for t_idx, t in enumerate(years):
            growth = _lhs_sample(n, 0.04, 0.06, seed + j * 2000 + i * 100 + t, distribution)
            c_vals = np.full(n, c_base)
            for k in range(t_idx + 1):
                c_vals = c_vals * (1.0 + growth)
            cost[(j, i, t, s)] = c_vals

    # ---- 价格生成 ----
    for (i, s) in is_pairs:
        p_base = data.p.get((i, s), 0.0)
        group = _crop_price_group_q3(i)
        for t_idx, t in enumerate(years):
            if group == "grain":
                # 粮食价格：[-1%,1%]不累积
                shock = _lhs_sample(n, -0.01, 0.01, seed + i * 300 + t, distribution)
                p_vals = p_base * (1.0 + shock)
            elif group == "vegetable":
                # 蔬菜价格：年增长率[4%,6%]逐年递推
                growth = _lhs_sample(n, 0.04, 0.06, seed + i * 300 + t, distribution)
                p_vals = np.full(n, p_base)
                for k in range(t_idx + 1):
                    p_vals = p_vals * (1.0 + growth)
            elif group == "mushroom":
                # 食用菌价格：年下降率[1%,5%]逐年递推
                decline = _lhs_sample(n, 0.01, 0.05, seed + i * 300 + t, distribution)
                p_vals = np.full(n, p_base)
                for k in range(t_idx + 1):
                    p_vals = p_vals * (1.0 - decline)
            elif group == "morel":
                # 羊肚菌：固定年下降5%
                p_vals = p_base * (0.95 ** (t_idx + 1))
                p_vals = np.full(n, p_vals)  # 确定值
            else:
                p_vals = np.full(n, p_base)
            price[(i, t, s)] = p_vals
            trend_price[(i, t, s)] = np.full(n, p_base)  # Q2确定趋势基线

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
    try:
        from .dependency import (
            DependencyConfig, build_factor_map, build_latent_correlation,
            generate_t_rank_scores, reorder_lhs_by_ranks, BASE_LOADINGS,
        )
    except ImportError:
        return marginals  # 降级：返回原始样本

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

    factor_maps = build_factor_map(crop_groups, dependency_cfg)
    years = data.years

    # 为每组生成t分数并重排边际样本
    for group, fmap in factor_maps.items():
        r_lat = build_latent_correlation(fmap, years)
        scores = generate_t_rank_scores(marginals.n, fmap, years,
                                         seed, dependency_cfg.df)
        # scores shape: (N, n_dims * n_years)
        # 对该组内所有作物的边际样本按秩重排
        # 使用第一维分数重排该组内所有参数
        score_col = 0  # 用第一个维度的分数
        for key_dict in (marginals.demand, marginals.yield_,
                         marginals.cost, marginals.price):
            for key in list(key_dict.keys()):
                if len(key) == 3:  # (i, t, s)
                    i, t, s = key
                elif len(key) == 4:  # (j, i, t, s)
                    _, i, t, s = key
                else:
                    continue
                if crop_groups.get(i) == group:
                    vals = key_dict[key].copy()
                    rank_order = np.argsort(scores[:, score_col])
                    key_dict[key] = vals[np.argsort(rank_order)]

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
        try:
            from .elasticity import build_elasticity_matrix, apply_price_elasticity
            matrices = build_elasticity_matrix(data, elasticity_cfg.scale)
        except (ImportError, Exception):
            matrices = None
    else:
        matrices = None

    n = marginals.n
    weights = np.full(n, 1.0 / n)
    demand_base = {k: v.copy() for k, v in marginals.demand.items()}
    demand = {k: v.copy() for k, v in marginals.demand.items()}

    # 应用弹性修正
    if matrices is not None:
        is_pairs = sorted({(i, s) for (j, i, s) in data.suit.keys()})
        crop_to_idx = {i: idx for idx, i in enumerate(sorted(
            {i for (i, s) in is_pairs}))}
        for s, matrix in matrices.items():
            for (i, s2) in is_pairs:
                if s2 != s:
                    continue
                for t in data.years:
                    base = demand_base.get((i, t, s))
                    scen_p = marginals.price.get((i, t, s))
                    trend_p = marginals.trend_price.get((i, t, s))
                    if base is None or scen_p is None or trend_p is None:
                        continue
                    idx = crop_to_idx.get(i)
                    if idx is None or idx >= matrix.shape[0]:
                        continue
                    # 获取该季次所有作物价格
                    crop_indices = {ci: ci_idx for ci, ci_idx in crop_to_idx.items()
                                   if (ci, s) in is_pairs}
                    p_ratio = np.ones((n, len(crop_indices)))
                    for ci, ci_idx in crop_indices.items():
                        sp = marginals.price.get((ci, t, s))
                        tp = marginals.trend_price.get((ci, t, s))
                        if sp is not None and tp is not None:
                            p_ratio[:, ci_idx] = sp / np.maximum(tp, 1e-10)
                    # 应用弹性
                    modified = apply_price_elasticity(
                        base.reshape(1, -1), p_ratio,
                        trend_p[:1] if trend_p is not None else np.ones(1),
                        matrices, crop_indices, s)
                    demand[(i, t, s)] = modified.ravel()

    return Q3ScenarioSet(
        weights=weights, demand=demand, demand_base=demand_base,
        yield_=marginals.yield_, cost=marginals.cost, price=marginals.price,
        trend_price=marginals.trend_price, n=n,
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
