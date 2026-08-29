# -*- coding: utf-8 -*-
"""Q2/Q3 配对比较与四组消融 — doc/Q3_编程手实现指导.md section 3.10、
doc/Q3_尝试解答.md section 9。

功能介绍
    本模块实现 Q3 相对 Q2 的共同随机数配对比较和四组消融实验：
      1. evaluate_fixed_plan   — 在一批情景上复算固定方案的逐情景利润，
          返回均值/标准差/10%分位/最差10%平均(CVaR)/最低/亏损概率。
      2. paired_compare         — Q3 与 Q2 方案在同一批情景上配对比较，
          返回逐情景利润差、面积 L1 距离、激活 Jaccard 相似度。
      3. bootstrap_paired_difference — 对配对利润差做 bootstrap 95% CI。
      4. run_ablation           — 四组消融（Q2基线/仅相关/仅弹性互补/完整Q3），
          所有组共享边际样本和随机秩流（共同随机数），Q2组只复算不重优化。

    利润公式（与 model.py 一致）：
        Q_{i,t,s}^omega = sum_j q_omega * (x + gamma_i * w)
        u_{i,t,s}^omega = min(Q, D)
        Pi^omega = sum_{i,t,s} p*u - sum_{j,i,t,s} c*x
    其中 gamma_i = gamma（非豆类）或 0（豆类 LEGUME_CODES），w 仅对非豆类定义。

使用方法
    from algorithms.evaluate import (
        evaluate_fixed_plan,
        paired_compare,
        bootstrap_paired_difference,
        run_ablation,
        ScenarioProfitFrame,
        ComparisonReport,
        BootstrapCI,
    )

    # 逐情景利润复算
    frame = evaluate_fixed_plan(plan, scenarios, data, gamma=0.03)

    # Q3 vs Q2 配对比较（同一批情景）
    report = paired_compare(plan_q3, plan_q2, scenarios, data, gamma=0.03)

    # 配对利润差的 bootstrap 95% CI
    ci = bootstrap_paired_difference(report.delta_profit, n_boot=10000, seed=2024)

    # 四组消融
    df = run_ablation(configs, common_random_stream, data)

    plan 字典格式（来自 solve.extract_solution）：
        {"x": {(j,i,t,s): area}, "y": {(j,i,t,s): 0/1},
         "w": {(j,i,t,s): comp_area}, "r": {}, "b": {}}
    Q2 冻结方案通过 _load_q2_plan_dict 从 selected_plan.csv 转换而来。

运行环境
    Python 3.10+，依赖 numpy、pandas。run_ablation 额外依赖
    scenarios / scenario_reduction / solve / io_data 子模块。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .preprocess import ModelData, LEGUME_CODES
from .scenarios import Q3ScenarioSet, MarginalScenarioSet


# ================================================================== #
# 数据结构
# ================================================================== #

@dataclass
class ScenarioProfitFrame:
    """固定方案在一批情景上的利润统计框。

    Attributes:
        profits: (N,) 逐情景利润。
        weights: (N,) 情景权重（均匀或缩减权重）。
        mean: 加权平均利润。
        std: 加权标准差。
        p10: 10% 分位数（下尾）。
        cvar: 最差 10% 平均利润（LCVaR_{0.90}）。
        min_profit: 最低单情景利润。
        loss_prob: 亏损概率 P(profit < 0)。
    """
    profits: np.ndarray
    weights: np.ndarray
    mean: float
    std: float
    p10: float
    cvar: float
    min_profit: float
    loss_prob: float


@dataclass
class ComparisonReport:
    """Q3 vs Q2 配对比较报告（同一批情景，共同随机数）。

    Attributes:
        q3_frame: Q3 方案的 ScenarioProfitFrame。
        q2_frame: Q2 方案的 ScenarioProfitFrame。
        delta_profit: (N,) 逐情景 Q3-Q2 利润差。
        mean_delta: 加权平均利润差。
        area_l1_distance: 面积 L1 距离 sum|x_q3 - x_q2|。
        activation_jaccard: 激活 Jaccard |y_q3 ∩ y_q2| / |y_q3 ∪ y_q2|。
    """
    q3_frame: ScenarioProfitFrame
    q2_frame: ScenarioProfitFrame
    delta_profit: np.ndarray
    mean_delta: float
    area_l1_distance: float
    activation_jaccard: float


@dataclass
class BootstrapCI:
    """配对利润差均值的 bootstrap 95% 置信区间。

    Attributes:
        point_estimate: 原始样本均值。
        lower_bound: 95% CI 下界（2.5% 分位）。
        upper_bound: 95% CI 上界（97.5% 分位）。
        n_boot: bootstrap 重抽样次数。
    """
    point_estimate: float
    lower_bound: float
    upper_bound: float
    n_boot: int


# ================================================================== #
# 辅助函数
# ================================================================== #

def _scenario_count(scenarios) -> int:
    """获取情景数量（兼容 Q3ScenarioSet.n 和 ReducedScenarioSet.k）。"""
    if hasattr(scenarios, "k"):
        return int(scenarios.k)
    if hasattr(scenarios, "n"):
        return int(scenarios.n)
    return int(len(scenarios.weights))


def _get_weights(scenarios, n: int) -> np.ndarray:
    """获取情景权重，缺失时返回均匀权重。"""
    w = getattr(scenarios, "weights", None)
    if w is not None:
        arr = np.asarray(w, dtype=float)
        if arr.size == n:
            return arr
    return np.full(n, 1.0 / n)


def _weighted_percentile(profits: np.ndarray, weights: np.ndarray,
                         p: float) -> float:
    """加权分位数（p ∈ [0, 100]）。

    按利润升序排列，累计权重达到 p% 时对应的利润值。
    """
    order = np.argsort(profits)
    sorted_p = profits[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    target = (p / 100.0) * cum_w[-1]
    idx = int(np.searchsorted(cum_w, target))
    if idx >= len(sorted_p):
        idx = len(sorted_p) - 1
    return float(sorted_p[idx])


def _weighted_cvar(profits: np.ndarray, weights: np.ndarray,
                    beta: float = 0.90) -> float:
    """加权 LCVaR_beta（最差 (1-beta) 部分的加权均值）。

    排序后取累计权重达到 (1-beta) 的尾部，边界点按比例拆分权重。
    """
    order = np.argsort(profits)
    sorted_p = profits[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    tail_mass = (1.0 - beta) * total_w
    idx = int(np.searchsorted(cum_w, tail_mass))
    if idx >= len(sorted_p):
        idx = len(sorted_p) - 1
    # 边界权重拆分：超出 tail_mass 的部分从边界点扣除
    tail_w = sorted_w[:idx + 1].astype(float).copy()
    if cum_w[idx] > tail_mass and idx > 0:
        excess = cum_w[idx] - tail_mass
        tail_w[idx] = max(0.0, sorted_w[idx] - excess)
    total_tail = tail_w.sum()
    if total_tail > 0:
        return float(np.average(sorted_p[:idx + 1], weights=tail_w))
    return float(sorted_p[0])


def _compute_scenario_profits(plan: dict, scenarios, data: ModelData,
                              gamma: float = 0.03) -> np.ndarray:
    """向量化计算固定方案在所有情景上的利润。

    Q_{i,t,s}^omega = sum_j q_omega[j,i,t,s] * (x[j,i,t,s] + gamma_i * w[j,i,t,s])
    u = min(Q, D)
    Pi = sum p*u - sum c*x

    gamma_i = gamma（非豆类）或 0（豆类）；w 仅对非豆类有定义。
    """
    n = _scenario_count(scenarios)
    x = plan.get("x", {})
    w = plan.get("w", {})

    # 按 (i, t, s) 分组聚合产量
    prod_groups = {}  # (i,t,s) -> [(j, x_area, gamma_i*w_area), ...]
    for (j, i, t, s), area in x.items():
        if area <= 0:
            continue
        gamma_i = gamma if i not in LEGUME_CODES else 0.0
        gw_area = float(w.get((j, i, t, s), 0.0)) * gamma_i
        prod_groups.setdefault((i, t, s), []).append((j, float(area), gw_area))

    revenue = np.zeros(n)
    for (i, t, s), entries in prod_groups.items():
        q_sum = np.zeros(n)
        for (j, x_area, gw_area) in entries:
            q_arr = scenarios.yield_.get((j, i, t, s))
            if q_arr is None:
                continue
            q_sum = q_sum + q_arr * (x_area + gw_area)
        d_arr = scenarios.demand.get((i, t, s), np.zeros(n))
        u = np.minimum(q_sum, d_arr)
        p_arr = scenarios.price.get((i, t, s), np.zeros(n))
        revenue = revenue + p_arr * u

    cost = np.zeros(n)
    for (j, i, t, s), area in x.items():
        if area <= 0:
            continue
        c_arr = scenarios.cost.get((j, i, t, s))
        if c_arr is None:
            continue
        cost = cost + c_arr * float(area)

    return revenue - cost


def _build_frame(profits: np.ndarray, weights: np.ndarray) -> ScenarioProfitFrame:
    """从利润数组和权重构建 ScenarioProfitFrame。"""
    n = len(profits)
    if n == 0:
        return ScenarioProfitFrame(
            profits=np.array([]), weights=np.array([]),
            mean=0.0, std=0.0, p10=0.0, cvar=0.0,
            min_profit=0.0, loss_prob=0.0)

    mean = float(np.average(profits, weights=weights))
    var = float(np.average((profits - mean) ** 2, weights=weights))
    std = float(np.sqrt(max(var, 0.0)))
    p10 = _weighted_percentile(profits, weights, 10.0)
    cvar = _weighted_cvar(profits, weights, beta=0.90)
    min_profit = float(np.min(profits))
    loss_prob = float(np.average(profits < 0, weights=weights))

    return ScenarioProfitFrame(
        profits=profits, weights=weights, mean=mean, std=std,
        p10=p10, cvar=cvar, min_profit=min_profit, loss_prob=loss_prob)


def _deep_copy_marginals(marginals: MarginalScenarioSet) -> MarginalScenarioSet:
    """深拷贝 MarginalScenarioSet（共享边际样本，各组独立修改）。"""
    return MarginalScenarioSet(
        demand={k: v.copy() for k, v in marginals.demand.items()},
        yield_={k: v.copy() for k, v in marginals.yield_.items()},
        cost={k: v.copy() for k, v in marginals.cost.items()},
        price={k: v.copy() for k, v in marginals.price.items()},
        trend_price={k: v.copy() for k, v in marginals.trend_price.items()},
        n=marginals.n,
    )


def _derive_scenarios(marginals: MarginalScenarioSet, data: ModelData,
                      use_corr: bool, use_elas: bool,
                      dep_cfg, elas_cfg, seed: int) -> Q3ScenarioSet:
    """从共享边际样本派生情景（按开关施加相关重排和弹性修正）。

    所有组共享同一份边际样本（深拷贝后修改），保证共同随机数。
    """
    from .scenarios import correlate_marginals, apply_market_interactions
    m = _deep_copy_marginals(marginals)
    if use_corr:
        m = correlate_marginals(m, dep_cfg, data, seed)
    return apply_market_interactions(m, data, elas_cfg if use_elas else None)


def _load_q2_plan_dict(q2_plan_path, data: ModelData) -> dict:
    """从 selected_plan.csv 读取冻结 Q2 方案，转为 plan 字典。

    CSV 的 plot_idx 按 CSV 出现顺序编号，需重映射到 ModelData 的 plot_idx。
    Q2 无 w/r/b 变量，置为空字典。
    """
    from .io_data import load_q2_baseline
    bp = load_q2_baseline(q2_plan_path)
    # CSV 顺序 j -> 地块名 -> ModelData 顺序 j
    csv_j_to_name = {j: name for name, j in bp.plot_idx.items()}
    x, y = {}, {}
    for (j_csv, i, t, s), area in bp.area.items():
        name = csv_j_to_name.get(j_csv)
        if name is None:
            continue
        j_md = data.plot_idx.get(name)
        if j_md is None:
            continue
        x[(j_md, i, t, s)] = area
        y[(j_md, i, t, s)] = 1 if area > 0 else 0
    return {"x": x, "y": y, "w": {}, "r": {}, "b": {}}


def _count_activations(plan: dict) -> int:
    """统计方案中激活的 (j,i,t,s) 组合数。"""
    y = plan.get("y", {})
    if y:
        return int(sum(1 for v in y.values() if v >= 0.5))
    x = plan.get("x", {})
    return int(sum(1 for v in x.values() if v > 0))


def _solve_and_select(data: ModelData, reduced, beta: float,
                      lambda_grid, eta: float, gamma: float,
                      time_limit: float, mip_gap: float,
                      selection_rule: str,
                      ckpt_dir=None, config_hash: str = "") -> dict | None:
    """对 lambda 网格逐点求解三级字典序，按选择规则选出唯一方案。

    选择规则（与 risk.py 一致，此处为简化内联版）：
      - "expected": 选缩减情景上平均利润最高的方案。
      - "knee": 简化为选平均利润最高（前沿完整时应计算膝点距离）。

    所有组使用相同的选择规则，保证消融可比性。
    """
    from pathlib import Path
    from .solve import solve_lexicographic

    ckpt = Path(ckpt_dir) if ckpt_dir else None
    solutions = []
    for lam in lambda_grid:
        result = solve_lexicographic(
            data, reduced, beta=beta, risk_lambda=float(lam), eta=eta,
            gamma=gamma, time_limit=time_limit, mip_gap=mip_gap,
            ckpt_dir=ckpt, config_hash=config_hash)
        sol = result.get("solution")
        if sol is not None:
            solutions.append((float(lam), result, sol))

    if not solutions:
        return None

    # 在缩减情景上评估各方案，选最优
    best_plan = None
    best_score = -np.inf
    weights = reduced.weights
    for lam, result, sol in solutions:
        profits = _compute_scenario_profits(sol, reduced, data, gamma)
        score = float(np.average(profits, weights=weights))
        if score > best_score:
            best_score = score
            best_plan = sol

    return best_plan


# ================================================================== #
# 主接口
# ================================================================== #

def evaluate_fixed_plan(plan: dict, scenarios, data: ModelData,
                        gamma: float = 0.03) -> ScenarioProfitFrame:
    """在情景集上复算固定方案的逐情景利润及统计指标。

    Args:
        plan: 方案字典，至少含 "x" {(j,i,t,s): area} 和 "w" {(j,i,t,s): comp_area}。
              "y" 可选（用于配对比较的 Jaccard）。
        scenarios: Q3ScenarioSet（.n）或 ReducedScenarioSet（.k）。
        data: ModelData（用于 LEGUME_CODES 判定）。
        gamma: 豆类前茬互补增产率（非豆类作物）。

    Returns:
        ScenarioProfitFrame 含均值/标准差/p10/CVaR/min/亏损概率。
    """
    n = _scenario_count(scenarios)
    weights = _get_weights(scenarios, n)
    profits = _compute_scenario_profits(plan, scenarios, data, gamma)
    return _build_frame(profits, weights)


def paired_compare(plan_q3: dict, plan_q2: dict, scenarios,
                   data: ModelData, gamma: float = 0.03) -> ComparisonReport:
    """Q3 与 Q2 方案在同一批情景上配对比较（共同随机数）。

    两方案在完全相同的情景集上逐情景复算利润，保证一一对应。
    逐情景差 delta = Pi_q3 - Pi_q2，消除共同随机数方差。

    Args:
        plan_q3: Q3 方案字典（含 x, y, w）。
        plan_q2: Q2 方案字典（含 x, y；w 为空）。
        scenarios: 共同情景集（Q3ScenarioSet 或 ReducedScenarioSet）。
        data: ModelData。
        gamma: 互补增产率。

    Returns:
        ComparisonReport 含两方案的 ScenarioProfitFrame、逐情景差、
        面积 L1 距离和激活 Jaccard 相似度。
    """
    n = _scenario_count(scenarios)
    weights = _get_weights(scenarios, n)

    q3_profits = _compute_scenario_profits(plan_q3, scenarios, data, gamma)
    q2_profits = _compute_scenario_profits(plan_q2, scenarios, data, gamma)

    q3_frame = _build_frame(q3_profits, weights)
    q2_frame = _build_frame(q2_profits, weights)

    delta = q3_profits - q2_profits
    mean_delta = float(np.average(delta, weights=weights))

    # 面积 L1 距离
    x_q3 = plan_q3.get("x", {})
    x_q2 = plan_q2.get("x", {})
    all_keys = set(x_q3.keys()) | set(x_q2.keys())
    area_l1 = float(sum(
        abs(float(x_q3.get(k, 0.0)) - float(x_q2.get(k, 0.0)))
        for k in all_keys
    ))

    # 激活 Jaccard：y 集合的交并比
    y_q3 = plan_q3.get("y", {})
    y_q2 = plan_q2.get("y", {})
    act_q3 = {k for k, v in y_q3.items() if v >= 0.5}
    act_q2 = {k for k, v in y_q2.items() if v >= 0.5}
    # y 为空时从 x 推导激活集
    if not act_q3:
        act_q3 = {k for k, v in x_q3.items() if v > 0}
    if not act_q2:
        act_q2 = {k for k, v in x_q2.items() if v > 0}
    union = act_q3 | act_q2
    inter = act_q3 & act_q2
    jaccard = float(len(inter) / len(union)) if union else 1.0

    return ComparisonReport(
        q3_frame=q3_frame, q2_frame=q2_frame, delta_profit=delta,
        mean_delta=mean_delta, area_l1_distance=area_l1,
        activation_jaccard=jaccard)


def bootstrap_paired_difference(delta_profit: np.ndarray,
                                 n_boot: int = 10000,
                                 seed: int = 2024) -> BootstrapCI:
    """配对利润差均值的 bootstrap 95% 置信区间（百分位法）。

    对 delta_profit 做 n_boot 次有放回重抽样，每次计算均值，
    取 2.5% 和 97.5% 分位作为 95% CI 的上下界。

    Args:
        delta_profit: (N,) 逐情景配对利润差。
        n_boot: 重抽样次数。
        seed: 随机种子。

    Returns:
        BootstrapCI 含点估计和 95% CI 上下界。
    """
    delta = np.asarray(delta_profit, dtype=float).ravel()
    n = len(delta)
    if n == 0:
        return BootstrapCI(
            point_estimate=0.0, lower_bound=0.0, upper_bound=0.0,
            n_boot=n_boot)

    point = float(np.mean(delta))
    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(delta[idx]))

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return BootstrapCI(
        point_estimate=point, lower_bound=lower, upper_bound=upper,
        n_boot=n_boot)


def run_ablation(configs: dict, common_random_stream, data: ModelData
                 ) -> "pd.DataFrame":
    """四组消融实验 — Q2基线/仅相关/仅弹性互补/完整Q3。

    所有组共享同一份边际样本和随机秩流（共同随机数），Q2 组读取冻结方案
    只复算不重优化；其余三组只开关相关性、弹性/互补组件，保持 beta、lambda
    网格、情景数、求解时限、MIP gap 和选择规则一致。

    configs 期望字段:
        beta (float): CVaR 置信水平，默认 0.90。
        lambda_grid (list): 风险厌恶参数网格。
        eta (float): 最小面积比例，默认 0.5。
        gamma (float): 互补增产率，默认 0.03。
        time_limit (float): 求解时限（秒），默认 600。
        mip_gap (float): MIP 相对间隙，默认 0.001。
        n_reduced (int): 缩减情景数，默认 30。
        selection_rule (str): 方案选择规则，默认 "expected"。
        q2_plan_path: Q2 selected_plan.csv 路径。
        dependency_cfg: DependencyConfig（相关性开关，None=关闭）。
        elasticity_cfg: ElasticityConfig（弹性开关，None=关闭）。
        ckpt_dir: 检查点目录（可选）。
        config_hash (str): 配置哈希（用于检查点恢复）。

    common_random_stream 期望字段:
        marginals (MarginalScenarioSet): 共享边际 LHS 样本。
        eval_scenarios (Q3ScenarioSet): 共同样本外评估情景（完整 Q3）。
        seed (int): 共享随机种子。

    Args:
        configs: 消融配置字典。
        common_random_stream: 共同随机数流（边际样本+评估情景+种子）。
        data: ModelData。

    Returns:
        DataFrame，列: group, mean_profit, std, p10, cvar, min, loss_prob,
        n_activations。求解失败的组以 NaN 填充。
    """
    from .scenario_reduction import reduce_scenarios

    # ---- 提取公共配置 ----
    beta = float(configs.get("beta", 0.90))
    eta = float(configs.get("eta", 0.5))
    gamma_full = float(configs.get("gamma", 0.03))
    lambda_grid = configs.get("lambda_grid", [0.0, 0.5, 1.0])
    time_limit = float(configs.get("time_limit", 600.0))
    mip_gap = float(configs.get("mip_gap", 0.001))
    n_reduced = int(configs.get("n_reduced", 30))
    selection_rule = configs.get("selection_rule", "expected")
    q2_plan_path = configs.get("q2_plan_path")
    dep_cfg = configs.get("dependency_cfg")
    elas_cfg = configs.get("elasticity_cfg")
    ckpt_dir = configs.get("ckpt_dir")
    config_hash = configs.get("config_hash", "")

    # ---- 共同随机数流 ----
    marginals = common_random_stream["marginals"]
    eval_scenarios = common_random_stream["eval_scenarios"]
    seed = int(common_random_stream.get("seed", 2024))

    # ---- 四组定义: (名称, 开相关, 开弹性, 开互补) ----
    group_defs = [
        ("q2_baseline", False, False, False),                 # Q2 冻结方案只复算
        ("correlation_only", True, False, False),            # 仅加相关性
        ("elasticity_complementarity_only", False, True, True),  # 仅加弹性+互补
        ("full_q3", True, True, True),                        # 完整 Q3
    ]

    results = []
    for group_name, use_corr, use_elas, use_comp in group_defs:
        if group_name == "q2_baseline":
            # Q2 组：读取冻结方案，不重新优化
            plan = _load_q2_plan_dict(q2_plan_path, data) if q2_plan_path else None
        else:
            # 从共享边际派生情景（共同随机数）
            gamma_g = gamma_full if use_comp else 0.0
            scenarios = _derive_scenarios(
                marginals, data, use_corr, use_elas, dep_cfg, elas_cfg, seed)
            # 情景缩减
            reduced, _ = reduce_scenarios(
                data, scenarios, k=n_reduced, gamma=gamma_g, seed=seed)
            # 求解并选择唯一方案
            plan = _solve_and_select(
                data, reduced, beta, lambda_grid, eta, gamma_g,
                time_limit, mip_gap, selection_rule,
                ckpt_dir, config_hash)

        # 在共同样本外情景上复算
        if plan is None:
            results.append({
                "group": group_name, "mean_profit": np.nan,
                "std": np.nan, "p10": np.nan, "cvar": np.nan,
                "min": np.nan, "loss_prob": np.nan,
                "n_activations": 0,
            })
            continue

        frame = evaluate_fixed_plan(plan, eval_scenarios, data, gamma_full)
        n_act = _count_activations(plan)
        results.append({
            "group": group_name,
            "mean_profit": frame.mean,
            "std": frame.std,
            "p10": frame.p10,
            "cvar": frame.cvar,
            "min": frame.min_profit,
            "loss_prob": frame.loss_prob,
            "n_activations": n_act,
        })

    return pd.DataFrame(results)
