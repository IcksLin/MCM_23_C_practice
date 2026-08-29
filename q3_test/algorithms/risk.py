# -*- coding: utf-8 -*-
"""Q3 风险指标、前沿资格判定与唯一方案选择。

功能
----
1. recompute_scenario_profits — 独立复算固定方案在每个情景下的利润。
   采用 Q3 产量公式 Q = sum_j q * (x + gamma_i * w)，gamma_i 对豆类为 0。
2. compute_cvar — 下尾 CVaR_beta（最差 (1-beta) 分位加权均值，处理边界分权重）。
3. pareto_nondominated — 剔除前沿中弱支配点。
4. select_unique_plan — Q3 唯一方案选择（膝点法 + 99% 下尾覆盖回退 + 字典序破同分）。
5. check_frontier_complete — 判定 11 个 lambda 是否全部有资格（frontier_complete）。
6. check_recommended_conditions — 推荐条件（非强制）：平均利润损失<=2%、下尾改善>0、
   成对 bootstrap 95% CI 下界>=0，并输出 1%/2%/3%/5% 门槛敏感性。

使用方法
--------
本模块为 q3_test.algorithms 子包的一部分，供 run_q3.py / evaluate.py 调用：

    from algorithms.risk import (
        recompute_scenario_profits, compute_cvar, select_unique_plan,
        check_frontier_complete, check_recommended_conditions,
    )

    profits = recompute_scenario_profits(plan, data, reduced, gamma=0.03)
    cvar = compute_cvar(profits, reduced.weights, beta=0.90)
    selected = select_unique_plan(frontier_points)

frontier_points 为字典列表，每个字典至少包含:
    lambda, z_lambda, expected_profit, cvar, status, lex_complete, n_activations

运行环境
--------
- Python 3.10+
- numpy（仅依赖 numpy，不依赖 pandas）
- 所属包: q3_test.algorithms（相对导入 preprocess / scenario_reduction）

参考: doc/Q3_编程手实现指导.md section 3.11；题目分析报告.md section 12.6。

作者: Q3编程手
来源: 扩展自 q2_test/algorithms/risk.py，依据 Q3 新增豆类前茬互补与前沿资格规则。
"""
from __future__ import annotations
import numpy as np

from .preprocess import ModelData, LEGUME_CODES
from .scenario_reduction import ReducedScenarioSet


# ---- 全局常量 ----
# 11 个风险厌恶参数网格（doc/Q3_编程手实现指导.md section 4 配置合同）
LAMBDA_GRID: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# 膝点判定阈值：端点连线最大垂直距离 >= 0.02 才视为有膝点（根报告 12.6）
KNEE_DISTANCE_THRESHOLD: float = 0.02
# 无膝点回退：下尾利润需达到全前沿最大 CVaR 的 99%
CVAR_COVERAGE_RATIO: float = 0.99
# 距离并列容差
DISTANCE_TIE_TOL: float = 1e-9
# 帕累托弱支配容差
DOMINANCE_TOL: float = 1e-8
# 归一化分母保护
_NORM_EPS: float = 1e-15


# =============================================================================
# 利润复算
# =============================================================================

def recompute_scenario_profits(plan: dict, data: ModelData,
                               scenarios: ReducedScenarioSet,
                               gamma: float = 0.03) -> np.ndarray:
    """独立复算固定方案在每个情景下的利润（不依赖求解器目标值）。

    Q3 产量公式: Q_omega = sum_j q_omega * (x + gamma_i * w)
      - gamma_i = gamma（非豆类作物）, 0（豆类作物）
      - w 仅对非豆类作物有定义（来自 extract_solution 的 w_idx）
    利润: Pi_omega = sum_{i,t,s} p * min(Q, D) - sum_{j,i,t,s} c * x
      - 成本只计 x（种植面积），w 不产生额外成本（仅提升产量）

    Args:
      plan: dict，来自 solve.extract_solution，含 x/y/r/b/w/Q/u/n_activations。
            兼容 Q2 方案（无 w 键时按 w=0 处理）。
      data: ModelData（保留用于 API 一致性与未来校验；本函数实际不读取其字段，
            情景数据全部来自 scenarios）。
      scenarios: ReducedScenarioSet，提供 demand/yield_/cost/price 与权重。
      gamma: 豆类前茬增产率（默认 0.03，与配置合同一致）。

    Returns:
      (K,) np.ndarray，每个情景的利润（元）。
    """
    K = scenarios.k                          # 缩减情景数
    profits = np.zeros(K, dtype=float)
    x = plan["x"]                            # (j,i,t,s) -> 种植面积
    w = plan.get("w", {})                    # (j,i,t,s) -> 互补面积（仅非豆类）

    # 预取数组字典引用，避免循环内重复属性查找
    yield_ = scenarios.yield_
    demand = scenarios.demand
    cost = scenarios.cost
    price = scenarios.price

    for omega in range(K):
        # ---- 1. 产量 Q[(i,t,s)] = sum_j q_omega * (x + gamma_i * w) ----
        Q: dict = {}
        for (j, i, t, s), area in x.items():
            if area <= 0.0:
                continue
            q_omega = yield_.get((j, i, t, s), np.zeros(K))[omega]
            Q[(i, t, s)] = Q.get((i, t, s), 0.0) + q_omega * area
        # 互补增益项：gamma * q * w（仅非豆类）
        for (j, i, t, s), w_area in w.items():
            if w_area <= 0.0:
                continue
            if i in LEGUME_CODES:            # 豆类无互补增益（防御性检查）
                continue
            q_omega = yield_.get((j, i, t, s), np.zeros(K))[omega]
            Q[(i, t, s)] = Q.get((i, t, s), 0.0) + gamma * q_omega * w_area

        # ---- 2. 收入 = sum p * min(Q, D) ----
        revenue = 0.0
        for (i, t, s), q_tot in Q.items():
            d_omega = demand.get((i, t, s), np.zeros(K))[omega]
            u = min(q_tot, d_omega)          # 销量 = min(产量, 需求)
            p_omega = price.get((i, t, s), np.zeros(K))[omega]
            revenue += p_omega * u

        # ---- 3. 成本 = sum c * x（仅种植面积）----
        cost_val = 0.0
        for (j, i, t, s), area in x.items():
            if area <= 0.0:
                continue
            c_omega = cost.get((j, i, t, s), np.zeros(K))[omega]
            cost_val += c_omega * area

        profits[omega] = revenue - cost_val

    return profits


# =============================================================================
# CVaR
# =============================================================================

def compute_cvar(profits: np.ndarray, weights: np.ndarray,
                 beta: float = 0.90) -> float:
    """下尾 CVaR_beta = E[Pi | Pi <= VaR_beta]（最差 (1-beta) 分位加权均值）。

    离散加权情景算法:
      1. 按利润升序排列。
      2. 累计权重达到 (1-beta) 时定位 VaR 边界。
      3. 边界点按需取部分权重，使尾部总权重恰好为 (1-beta)。
      4. CVaR = 尾部利润的加权均值。

    Args:
      profits: (K,) 情景利润。
      weights: (K,) 情景权重（和为 1）。
      beta: 置信水平（默认 0.90，即关注最差 10%）。

    Returns:
      下尾 CVaR（float）。利润越高 CVaR 越高（下尾越好）。
    """
    K = len(profits)
    if K == 0:
        return 0.0

    # 按利润升序排列（稳定排序保证同值时权重确定性）
    order = np.argsort(profits, kind="stable")
    sorted_p = np.asarray(profits, dtype=float)[order]
    sorted_w = np.asarray(weights, dtype=float)[order]
    cum_w = np.cumsum(sorted_w)

    tail_mass = 1.0 - beta                   # 尾部目标权重
    idx = int(np.searchsorted(cum_w, tail_mass, side="left"))
    if idx >= K:
        idx = K - 1

    # 边界点部分权重处理：使尾部总权重恰好等于 (1-beta)
    if cum_w[idx] > tail_mass and idx > 0:
        partial = tail_mass - cum_w[idx - 1]   # 边界点需取的部分权重
        tail_weights = np.append(sorted_w[:idx], partial)
        tail_profits = sorted_p[:idx + 1]
    else:
        # 边界点恰好对齐或 idx==0（尾部只有最差一个点）
        tail_weights = sorted_w[:idx + 1]
        tail_profits = sorted_p[:idx + 1]

    total_tail = float(tail_weights.sum())
    if total_tail > 0.0:
        return float(np.average(tail_profits, weights=tail_weights))
    return float(sorted_p[0]) if K > 0 else 0.0


# =============================================================================
# 前沿几何辅助
# =============================================================================

def _perpendicular_distances(points_norm: list) -> np.ndarray:
    """计算各归一化点到首末点连线的垂直距离。

    Args:
      points_norm: 归一化点列表，每项为 (x, y)（期望利润 x，CVaR y）。

    Returns:
      (n,) 距离数组；首末点距离为 0。
    """
    n = len(points_norm)
    dists = np.zeros(n, dtype=float)
    if n <= 2:
        return dists

    p0 = np.asarray(points_norm[0], dtype=float)
    p1 = np.asarray(points_norm[-1], dtype=float)
    line_vec = p1 - p0
    line_len = float(np.linalg.norm(line_vec))
    if line_len < _NORM_EPS:
        return dists                         # 首末重合，距离全 0

    for i in range(n):
        pi = np.asarray(points_norm[i], dtype=float)
        # 2D 叉积返回标量，其绝对值 / 线长 = 点到直线距离
        dists[i] = abs(float(np.cross(line_vec, pi - p0))) / line_len
    return dists


def _max_perpendicular_distance(points_norm: list) -> tuple:
    """端点连线的最大垂直距离及其索引。

    Args:
      points_norm: 归一化点列表（同 _perpendicular_distances）。

    Returns:
      (max_dist, index_of_max)。点数<=2 或首末重合时返回 (0.0, 0)。
    """
    n = len(points_norm)
    if n <= 2:
        return (0.0, 0)
    dists = _perpendicular_distances(points_norm)
    max_idx = int(np.argmax(dists))
    return (float(dists[max_idx]), max_idx)


# =============================================================================
# 前沿资格与帕累托
# =============================================================================

def _qualifies_for_frontier(point: dict) -> bool:
    """判定单个 lambda 点是否有资格进入风险前沿。

    资格条件（doc/Q3_编程手实现指导.md section 3.11）:
      - 存在有限 incumbent（求解器返回有效 x，status 非 infeasible/time_limit_no_feasible）
      - 三级字典序全部完成（lex_complete=True）
    注意: 可行但未证明最优（feasible_not_proven）仍算有资格。
    """
    if not point.get("lex_complete", False):
        return False
    status = point.get("status", "unknown")
    # 只有 status 为 optimal 或 feasible_not_proven 才算有有限 incumbent
    if status not in ("optimal", "feasible_not_proven"):
        return False
    # 期望利润与 CVaR 必须有限（incumbent 有效性的代理校验）
    if not np.isfinite(point.get("expected_profit", np.nan)):
        return False
    if not np.isfinite(point.get("cvar", np.nan)):
        return False
    return True


def pareto_nondominated(frontier_points: list) -> list:
    """剔除前沿中弱支配点（同时最大化期望利润与下尾 CVaR）。

    Args:
      frontier_points: 字典列表，每项含 expected_profit、cvar（及 lambda 等）。

    Returns:
      非支配点列表，按 lambda 升序排列。
    """
    # 仅保留有限点
    pts = [p for p in frontier_points
           if np.isfinite(p.get("expected_profit", np.nan))
           and np.isfinite(p.get("cvar", np.nan))]
    keep: list = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            # q 弱支配 p: q.ep>=p.ep 且 q.cvar>=p.cvar 且至少一项严格更优
            if (q["expected_profit"] >= p["expected_profit"] - DOMINANCE_TOL
                    and q["cvar"] >= p["cvar"] - DOMINANCE_TOL
                    and (q["expected_profit"] > p["expected_profit"] + DOMINANCE_TOL
                         or q["cvar"] > p["cvar"] + DOMINANCE_TOL)):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    keep.sort(key=lambda p: p["lambda"])
    return keep


def check_frontier_complete(frontier_points: list,
                            lambda_grid: tuple = LAMBDA_GRID) -> bool:
    """判定风险前沿是否完整。

    frontier_complete=True 要求 lambda_grid 中全部点均有资格
    （doc/Q3_编程手实现指导.md section 3.11）。

    Args:
      frontier_points: 全部 lambda 点字典列表。
      lambda_grid: lambda 网格（默认 11 点 0.0..1.0）。

    Returns:
      bool，全部有资格为 True。
    """
    # 以 lambda 值为键建索引（兼容 float 与字符串形式的 lambda）
    by_lam: dict = {}
    for p in frontier_points:
        lam = p.get("lambda")
        if lam is None:
            continue
        by_lam[float(lam)] = p
    for lam in lambda_grid:
        p = by_lam.get(float(lam))
        if p is None or not _qualifies_for_frontier(p):
            return False
    return True


# =============================================================================
# 唯一方案选择
# =============================================================================

def select_unique_plan(frontier_points: list) -> dict:
    """Q3 风险前沿唯一方案选择（根报告 12.6 / 编程指导 3.11）。

    选择规则:
      1. 过滤到有资格点（有限 incumbent + lex_complete）。
      2. 帕累托非支配过滤。
      3. 期望利润(x)与 CVaR(y) 分别归一化到 [0,1]。
      4. 计算各点到首末连线的垂直距离。
      5. 若最大距离 >= 0.02: 选膝点（最大距离点）。
         距离并列(1e-9 内)按 (lambda 升序, 期望利润降序, 启用次数升序) 破同分。
      6. 否则: 在 CVaR >= 99% 最大 CVaR 的候选中，按
         (lambda 升序, 期望利润降序, 启用次数升序) 选择。
      7. 破同分统一采用固定字段字典序:
         lambda 升序 → 期望利润降序 → 启用次数升序。

    Args:
      frontier_points: 字典列表，每项含 lambda, z_lambda, expected_profit,
                       cvar, status, lex_complete, n_activations。

    Returns:
      选中点字典（副本），新增 "selected_lambda" 键。
      若无有资格点，返回 None。
    """
    # 1. 资格过滤
    qualified = [p for p in frontier_points if _qualifies_for_frontier(p)]
    if not qualified:
        return None

    # 2. 帕累托非支配
    frontier = pareto_nondominated(qualified)
    n = len(frontier)
    if n == 0:
        return None
    if n == 1:
        result = dict(frontier[0])
        result["selected_lambda"] = frontier[0]["lambda"]
        return result

    # 3. 归一化到 [0,1]（期望利润为 x 轴，CVaR 为 y 轴）
    ep = np.array([p["expected_profit"] for p in frontier], dtype=float)
    cv = np.array([p["cvar"] for p in frontier], dtype=float)
    ep_n = (ep - ep.min()) / (ep.max() - ep.min() + _NORM_EPS)
    cv_n = (cv - cv.min()) / (cv.max() - cv.min() + _NORM_EPS)
    points_norm = list(zip(ep_n.tolist(), cv_n.tolist()))

    # 4. 垂直距离
    dists = _perpendicular_distances(points_norm)
    max_dist = float(dists.max())
    max_idx = int(np.argmax(dists))

    # 统一破同分键: lambda 升序, 期望利润降序, 启用次数升序
    def _sort_key(p: dict):
        return (p["lambda"], -p.get("expected_profit", 0.0),
                p.get("n_activations", 0))

    # 5. 膝点分支
    if max_dist >= KNEE_DISTANCE_THRESHOLD:
        tied = np.where(dists >= max_dist - DISTANCE_TIE_TOL)[0]
        if len(tied) <= 1:
            selected = frontier[max_idx]
        else:
            tied_pts = [frontier[i] for i in tied]
            tied_pts.sort(key=_sort_key)
            selected = tied_pts[0]
    else:
        # 6. 无明显膝点: CVaR >= 99% 最大 CVaR 的候选
        cvar_max = float(cv.max())
        threshold = CVAR_COVERAGE_RATIO * cvar_max
        candidates = [p for p in frontier if p["cvar"] >= threshold - DOMINANCE_TOL]
        if not candidates:
            # 退化: 取最大 CVaR 点
            candidates = [frontier[int(np.argmax(cv))]]
        candidates.sort(key=_sort_key)
        selected = candidates[0]

    result = dict(selected)
    result["selected_lambda"] = selected["lambda"]
    return result


# =============================================================================
# 推荐条件检查（非强制）
# =============================================================================

def check_recommended_conditions(q3_profit: np.ndarray, q2_profit: np.ndarray,
                                 weights: np.ndarray, beta: float = 0.90,
                                 n_boot: int = 2000,
                                 seed: int = 2024) -> dict:
    """检查 Q3 推荐条件（仅输出，不强制）。

    推荐条件（doc/Q3_编程手实现指导.md section 3.11）:
      1. Q3 相对 Q2 平均利润损失 <= 2%
      2. 下尾改善 > 0（CVaR_q3 - CVaR_q2 > 0）
      3. 成对 bootstrap 95% CI 下界 >= 0
    另输出 1%/2%/3%/5% 平均利润损失门槛敏感性。

    成对 bootstrap 对逐情景差值 delta = q3 - q2 重采样，计算每次的加权均值，
    取 2.5 百分位为 95% CI 下界；要求 Q3 不统计上劣于 Q2。

    Args:
      q3_profit: (K,) Q3 方案各情景利润（与 Q2 配对、共享情景）。
      q2_profit: (K,) Q2 基线各情景利润。
      weights: (K,) 情景权重。
      beta: CVaR 置信水平（默认 0.90）。
      n_boot: bootstrap 抽样次数（默认 2000）。
      seed: 随机种子（默认 2024）。

    Returns:
      含各项指标与 pass 标志的字典。
    """
    q3_profit = np.asarray(q3_profit, dtype=float)
    q2_profit = np.asarray(q2_profit, dtype=float)
    weights = np.asarray(weights, dtype=float)
    K = len(q3_profit)

    # ---- 加权均值 ----
    q3_mean = float(np.average(q3_profit, weights=weights)) if K else 0.0
    q2_mean = float(np.average(q2_profit, weights=weights)) if K else 0.0

    # ---- 平均利润损失比率 ----
    if abs(q2_mean) > _NORM_EPS:
        avg_loss_ratio = (q2_mean - q3_mean) / abs(q2_mean)
    else:
        avg_loss_ratio = 0.0 if q3_mean >= q2_mean else float("inf")

    # ---- 下尾改善 ----
    q3_cvar = compute_cvar(q3_profit, weights, beta)
    q2_cvar = compute_cvar(q2_profit, weights, beta)
    tail_improvement = q3_cvar - q2_cvar

    # ---- 成对 bootstrap 95% CI（逐情景差值 delta = q3 - q2 的加权均值）----
    delta = q3_profit - q2_profit            # 配对差值（正=Q3更优）
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, K, size=K)
        boot_means[b] = float(np.average(delta[idx], weights=weights[idx]))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # ---- 门槛敏感性（1%/2%/3%/5%）----
    thresholds = (0.01, 0.02, 0.03, 0.05)
    sensitivity = {
        f"avg_loss_pass_{int(t * 100)}pct": bool(avg_loss_ratio <= t)
        for t in thresholds
    }

    # ---- pass 标志 ----
    avg_loss_pass = bool(avg_loss_ratio <= 0.02)
    tail_pass = bool(tail_improvement > 0.0)
    ci_pass = bool(ci_lower >= 0.0)

    return {
        "q3_mean_profit": q3_mean,
        "q2_mean_profit": q2_mean,
        "avg_loss_ratio": float(avg_loss_ratio),
        "avg_loss_pass_2pct": avg_loss_pass,
        "q3_cvar": q3_cvar,
        "q2_cvar": q2_cvar,
        "tail_improvement": float(tail_improvement),
        "tail_improvement_pass": tail_pass,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "bootstrap_ci_pass": ci_pass,
        "n_bootstrap": int(n_boot),
        "n_paired": int(K),
        **sensitivity,
        "all_recommended_pass": bool(avg_loss_pass and tail_pass and ci_pass),
    }
