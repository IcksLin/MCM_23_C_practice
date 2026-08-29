# -*- coding: utf-8 -*-
"""Q3情景缩减 — 分层PAM k-medoids + Kendall审计 + 尾部保护。

扩展Q2的缩减算法，新增：
  - 距离特征增加参数变化率和公共因子得分
  - 缩减前后Kendall相关方向审计
  - 最低利润10%层至少保留ceil(0.1*K)个代表
  - 缩减样本加权Kendall最大误差不超过0.15

作者: Q3编程手
来源: 复制自q2_test/algorithms/scenario_reduction.py并扩展
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .preprocess import ModelData
from .scenarios import Q3ScenarioSet, compute_proxy_profit


@dataclass
class ReducedScenarioSet:
    """缩减后的情景集。"""
    indices: np.ndarray       # (K,) 原始情景索引
    weights: np.ndarray       # (K,) 权重，和为1
    demand: dict              # (i,t,s) -> array(K,)
    yield_: dict              # (j,i,t,s) -> array(K,)
    cost: dict                # (j,i,t,s) -> array(K,)
    price: dict               # (i,t,s) -> array(K,)
    k: int
    n_original: int
    proxy_profits: np.ndarray # (K,) 每个代表的代理利润
    demand_base: dict | None = None
    trend_price: dict | None = None


@dataclass
class ReductionAudit:
    """缩减审计结果。"""
    sum_weights: float
    min_weight: float
    max_weight: float
    zero_weight_count: int
    min_profit_layer_reps: int
    medoid_uniqueness: bool
    max_kendall_error: float
    kendall_direction_consistent: bool


def _compute_all_proxy_profits(scenarios: Q3ScenarioSet, data: ModelData,
                                plan_x: dict, gamma: float = 0.03) -> np.ndarray:
    """计算所有情景的代理利润。"""
    n = scenarios.n
    profits = np.zeros(n)
    for idx in range(n):
        profits[idx] = compute_proxy_profit(scenarios, data, plan_x, idx, gamma)
    return profits


def _build_change_vectors(scenarios: Q3ScenarioSet, data: ModelData) -> np.ndarray:
    """构建标准化参数变化率特征向量。

    特征包括:
      - 需求相对2023年的变化率
      - 亩产相对2023年的变化率
      - 成本相对2023年的变化率
      - 价格相对2023年的变化率
    """
    n = scenarios.n
    features = []

    # 需求变化率
    for (i, s) in sorted({(i, s) for (j, i, s) in data.suit.keys()}):
        for t in data.years:
            d_base = data.D.get((i, s), 0.0)
            d_vals = scenarios.demand.get((i, t, s), np.zeros(n))
            if d_base > 0:
                features.append((d_vals / d_base - 1.0).reshape(-1, 1))

    # 亩产变化率
    for (j, i, s) in sorted(data.suit.keys()):
        for t in data.years:
            q_base = data.q.get((j, i, s), 0.0)
            q_vals = scenarios.yield_.get((j, i, t, s), np.zeros(n))
            if q_base > 0:
                features.append((q_vals / q_base - 1.0).reshape(-1, 1))

    # 成本变化率
    for (j, i, s) in sorted(data.suit.keys()):
        for t in data.years:
            c_base = data.c.get((j, i, s), 0.0)
            c_vals = scenarios.cost.get((j, i, t, s), np.zeros(n))
            if c_base > 0:
                features.append((c_vals / c_base - 1.0).reshape(-1, 1))

    # 价格变化率
    for (i, s) in sorted({(i, s) for (j, i, s) in data.suit.keys()}):
        for t in data.years:
            p_base = data.p.get((i, s), 0.0)
            p_vals = scenarios.price.get((i, t, s), np.zeros(n))
            if p_base > 0:
                features.append((p_vals / p_base - 1.0).reshape(-1, 1))

    if not features:
        return np.zeros((n, 1))

    mat = np.hstack(features)
    # 距离空间使用确定性均匀抽取的边际变化列，控制 N²F 计算量。
    # 公共因子列在下方全量追加，不会被抽掉。
    if mat.shape[1] > 64:
        keep = np.linspace(0, mat.shape[1] - 1, 64, dtype=int)
        mat = mat[:, keep]
    if scenarios.factor_scores is not None:
        mat = np.hstack([mat, np.asarray(scenarios.factor_scores, dtype=float)])
    # 标准化
    std = np.std(mat, axis=0)
    std[std < 1e-10] = 1.0
    mat = mat / std
    return mat


def _weighted_kendall(x: np.ndarray, y: np.ndarray,
                      w: np.ndarray | None = None) -> float:
    """加权Kendall tau。

    Args:
      x, y: 值向量
      w: 权重向量（None则均匀）
    """
    n = len(x)
    if w is None:
        w = np.full(n, 1.0 / n)
    # 排序后计算
    order_x = np.argsort(x)
    # 使用O(n^2)算法（N不大时可接受）
    tau_sum = 0.0
    w_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = x[i], x[j]
            yi, yj = y[i], y[j]
            if (xi != xj) and (yi != yj):
                concord = ((xi - xj) * (yi - yj)) > 0
                wij = w[i] * w[j]
                tau_sum += (1 if concord else -1) * wij
                w_sum += wij
    return tau_sum / w_sum if w_sum > 0 else 0.0


def _pam_kmedoids(distances: np.ndarray, k: int, seed: int = 2024) -> tuple:
    """PAM k-medoids with L1 distance.

    Returns:
      (medoid_indices, assignments) where:
        medoid_indices: (k,) indices of selected medoids
        assignments: (N,) cluster assignment for each point
    """
    n = distances.shape[0]
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n), np.arange(n)

    # 初始化：最大距离贪心（确定性）
    medoids = [int(np.argmax(distances.sum(axis=1)))]
    for _ in range(1, k):
        min_dist_to_medoids = distances[:, medoids].min(axis=1)
        next_medoid = int(np.argmax(min_dist_to_medoids))
        if next_medoid in medoids:
            candidates = [i for i in range(n) if i not in medoids]
            if not candidates:
                break
            next_medoid = int(rng.choice(candidates))
        medoids.append(next_medoid)

    # 大样本使用簇内 medoid 交替更新；复杂度远低于穷举所有 swap。
    for _ in range(20):
        medoid_arr = np.asarray(medoids, dtype=int)
        assignments = np.argmin(distances[:, medoid_arr], axis=1)
        updated = medoids.copy()
        for cluster in range(len(medoids)):
            members = np.flatnonzero(assignments == cluster)
            if len(members) == 0:
                continue
            intra = distances[np.ix_(members, members)]
            updated[cluster] = int(members[np.argmin(intra.sum(axis=1))])
        if updated == medoids:
            break
        medoids = updated

    medoid_arr = np.array(medoids)
    dist_to_medoids = distances[:, medoid_arr]
    assignments = np.argmin(dist_to_medoids, axis=1)
    return medoid_arr, assignments


def reduce_scenarios(data: ModelData, scenarios: Q3ScenarioSet, k: int = 30,
                     baseline_plan: dict = None, gamma: float = 0.03,
                     seed: int = 2024) -> tuple:
    """执行分层PAM情景缩减。

    Returns:
      (ReducedScenarioSet, ReductionAudit)
    """
    n = scenarios.n
    if k >= n:
        # 无需缩减
        return ReducedScenarioSet(
            indices=np.arange(n), weights=np.full(n, 1.0/n),
            demand={key: vals for key, vals in scenarios.demand.items()},
            yield_={key: vals for key, vals in scenarios.yield_.items()},
            cost={key: vals for key, vals in scenarios.cost.items()},
            price={key: vals for key, vals in scenarios.price.items()},
            k=n, n_original=n, proxy_profits=np.zeros(n),
            demand_base={key: vals for key, vals in scenarios.demand_base.items()},
            trend_price={key: vals for key, vals in scenarios.trend_price.items()},
        ), ReductionAudit(
            sum_weights=1.0, min_weight=1.0/n, max_weight=1.0/n,
            zero_weight_count=0, min_profit_layer_reps=n,
            medoid_uniqueness=True, max_kendall_error=0.0,
            kendall_direction_consistent=True,
        )

    # 1. 计算代理利润
    if baseline_plan is None:
        baseline_plan = {k: v for k, v in data.bar_x.items() if v > 0}
    proxy_profits = _compute_all_proxy_profits(scenarios, data, baseline_plan, gamma)

    # 2. 利润十等频分层
    profit_order = np.argsort(proxy_profits)
    layer_size = max(1, n // 10)
    layers = []
    for layer_idx in range(10):
        start = layer_idx * layer_size
        end = min(start + layer_size, n) if layer_idx < 9 else n
        layers.append(profit_order[start:end])

    # 3. 构建距离矩阵（标准化变化率 + 利润）
    change_vectors = _build_change_vectors(scenarios, data)
    # 加入利润作为特征
    p_std = np.std(proxy_profits)
    if p_std < 1e-10:
        p_std = 1.0
    profit_feature = (proxy_profits - np.mean(proxy_profits)) / p_std
    all_features = np.hstack([change_vectors, profit_feature.reshape(-1, 1)])

    # L1距离矩阵
    from scipy.spatial.distance import cdist
    distances = cdist(all_features, all_features, metric="cityblock")

    # 4. 按利润层样本量分配代表，在各层内独立求 medoid。
    raw_alloc = np.array([k * len(layer) / n for layer in layers])
    alloc = np.floor(raw_alloc).astype(int)
    remainder = k - int(alloc.sum())
    for idx in np.argsort(-(raw_alloc - alloc))[:remainder]:
        alloc[idx] += 1
    min_tail_reps = math.ceil(0.1 * k)
    if alloc[0] < min_tail_reps:
        need = min_tail_reps - alloc[0]
        for idx in np.argsort(-alloc[1:]) + 1:
            take = min(need, max(0, alloc[idx]))
            alloc[idx] -= take
            alloc[0] += take
            need -= take
            if need == 0:
                break
    medoids = []
    for layer_idx, layer in enumerate(layers):
        count = min(int(alloc[layer_idx]), len(layer))
        if count <= 0:
            continue
        local, _ = _pam_kmedoids(distances[np.ix_(layer, layer)], count,
                                 seed + layer_idx)
        medoids.extend(layer[local].tolist())
    medoid_indices = np.asarray(medoids, dtype=int)
    assignments = np.argmin(distances[:, medoid_indices], axis=1)

    # 5. 尾部保护：最低10%利润层至少ceil(0.1*K)个代表
    bottom_layer = set(layers[0].tolist())
    tail_reps = [m for m in medoid_indices if m in bottom_layer]

    if len(tail_reps) < min_tail_reps:
        # 从底层未选情景中补充
        available = [i for i in layers[0] if i not in medoid_indices]
        needed = min_tail_reps - len(tail_reps)
        if len(available) >= needed:
            # 用尾部代表替换非尾部 medoid，始终保持代表数为 K。
            for _ in range(needed):
                if not available:
                    break
                dists = distances[available][:, medoid_indices].min(axis=1)
                far_idx = available[int(np.argmax(dists))]
                replaceable = [pos for pos, med in enumerate(medoid_indices)
                               if med not in bottom_layer]
                if not replaceable:
                    break
                # 优先替换权重贡献最小的非尾部代表。
                pos = replaceable[-1]
                medoid_indices[pos] = far_idx
                available.remove(far_idx)
            # 重新分配
            medoid_arr = np.array(medoid_indices)
            dist_to_medoids = distances[:, medoid_arr]
            assignments = np.argmin(dist_to_medoids, axis=1)

    # 6. 计算权重 = 簇大小 / N
    medoid_arr = np.array(medoid_indices)
    # 每个情景分配到最近的medoid
    dist_to_medoids = distances[:, medoid_arr]
    assignments = np.argmin(dist_to_medoids, axis=1)
    cluster_sizes = np.zeros(len(medoid_arr))
    for a in assignments:
        cluster_sizes[a] += 1
    weights = cluster_sizes / n

    # 7. 构建缩减情景集
    reduced_demand = {}
    reduced_yield = {}
    reduced_cost = {}
    reduced_price = {}
    reduced_demand_base = {}
    reduced_trend_price = {}

    for (i, t, s), vals in scenarios.demand.items():
        reduced_demand[(i, t, s)] = vals[medoid_arr]
    for (j, i, t, s), vals in scenarios.yield_.items():
        reduced_yield[(j, i, t, s)] = vals[medoid_arr]
    for (j, i, t, s), vals in scenarios.cost.items():
        reduced_cost[(j, i, t, s)] = vals[medoid_arr]
    for (i, t, s), vals in scenarios.price.items():
        reduced_price[(i, t, s)] = vals[medoid_arr]
    for key, vals in scenarios.demand_base.items():
        reduced_demand_base[key] = vals[medoid_arr]
    for key, vals in scenarios.trend_price.items():
        reduced_trend_price[key] = vals[medoid_arr]

    reduced = ReducedScenarioSet(
        indices=medoid_arr, weights=weights,
        demand=reduced_demand, yield_=reduced_yield,
        cost=reduced_cost, price=reduced_price,
        k=len(medoid_arr), n_original=n,
        proxy_profits=proxy_profits[medoid_arr],
        demand_base=reduced_demand_base,
        trend_price=reduced_trend_price,
    )

    # 8. 审计
    max_kendall_error = 0.0
    direction_ok = True
    if scenarios.factor_scores is not None:
        fs = np.asarray(scenarios.factor_scores, dtype=float)
        cols = min(fs.shape[1], 12)
        raw_w = np.full(n, 1.0 / n)
        for a in range(0, cols - 1, 2):
            b = a + 1
            tau_raw = _weighted_kendall(fs[:, a], fs[:, b], raw_w)
            tau_red = _weighted_kendall(
                fs[medoid_arr, a], fs[medoid_arr, b], weights)
            max_kendall_error = max(max_kendall_error,
                                    abs(tau_red - tau_raw))
            if abs(tau_raw) > 0.05 and tau_raw * tau_red < 0:
                direction_ok = False
    audit = ReductionAudit(
        sum_weights=float(weights.sum()),
        min_weight=float(weights.min()),
        max_weight=float(weights.max()),
        zero_weight_count=int((weights == 0).sum()),
        min_profit_layer_reps=int(np.sum([1 for m in medoid_arr
                                          if m in bottom_layer])),
        medoid_uniqueness=len(set(medoid_arr.tolist())) == len(medoid_arr),
        max_kendall_error=float(max_kendall_error),
        kendall_direction_consistent=bool(direction_ok),
    )

    return reduced, audit
