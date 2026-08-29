# -*- coding: utf-8 -*-
r"""因子 t-Copula 七年相关重排模块 (AGENT.md §3.4, Q3_尝试解答.md §4).

功能：
  1. 因子载荷矩阵 L（每作物类别一个 FactorMap）
  2. 潜在线性相关矩阵 R_lat（半正定，对角线为 1）
  3. t-Copula 秩分数（每情景共享一个卡方尺度，覆盖全部维度与七年）
  4. LHS 按秩重排（保留边际分层，注入相关结构）
  5. Kendall 审计（原始样本误差 ≤ 0.05，缩减样本 ≤ 0.15）

数学依据（Q3_尝试解答.md §4.2）：
  - 因子 AR(1) 路径：g[k,2024]~N(0,1)（稳态初始化，非零初始化），
    g[k,t]=ρ_k·g[k,t-1]+sqrt(1-ρ_k²)·ν[k,t]，ν~N(0,1)
  - t-Copula 分数：z[a,t]=(Σ_k L[a,k]·g[k,t]
    + sqrt(1-Σ_k L[a,k]²)·ε[a,t]) / sqrt(S_ω/ν)，
    S_ω~χ²(ν) 每情景内全部维度与七年共享
  - 目标 Kendall：τ*=(2/π)·arcsin(R_lat)
  - 潜在相关：R_lat[a_t,b_s]=Σ_k L[a,k]·L[b,k]·ρ_k^|t-s|
    + δ[a,b]·δ[t,s]·(1-Σ_k L[a,k]²)

运行环境：
  Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.10, pandas ≥ 1.5
  （base conda 环境: numpy 2.1.3, scipy 1.15.3, pandas 2.2.3 ✓）

使用示例（直接粘贴到命令行运行）：
  cd /d "d:\\时光归墟\\赛事\\数模\\practice_1\\q3_test"
  E:\anaconda\python.exe -c "import numpy as np, pandas as pd; from algorithms.dependency import *; cfg=DependencyConfig(5.0,1.0,0.5,BASE_LOADINGS); fms=build_factor_map({1:'grain',17:'vegetable',38:'fungi'},cfg); yrs=[2024,2025,2026,2027,2028,2029,2030]; fm=fms['grain']; R=build_latent_correlation(fm,yrs); sc=generate_t_rank_scores(500,fm,yrs,2024,5.0); lhs=np.random.default_rng(0).uniform(0,1,(500,sc.shape[1])); ro=reorder_lhs_by_ranks(lhs,sc); tau=(2/np.pi)*np.arcsin(np.clip(R,-1,1)); ps=build_audit_pairs(pd.DataFrame(tau),fm.dimensions,yrs); au=audit_dependency(ro,None,R,ps); print('min_eig',au.min_eigenvalue,'max_err',au.max_kendall_error,'n_pairs',len(ps))"

完整调用示例（Python 脚本）：
  import numpy as np, pandas as pd
  from algorithms.dependency import (
      BASE_LOADINGS, DependencyConfig, build_factor_map,
      build_latent_correlation, generate_t_rank_scores,
      reorder_lhs_by_ranks, build_audit_pairs, audit_dependency,
  )

  cfg = DependencyConfig(df=5.0, correlation_scale=1.0,
                         temporal_rho=0.5, loadings=BASE_LOADINGS)
  crop_groups = {1: "grain", 17: "vegetable", 38: "fungi"}
  fmaps = build_factor_map(crop_groups, cfg)
  years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]

  for grp, fm in fmaps.items():
      r_lat = build_latent_correlation(fm, years)
      scores = generate_t_rank_scores(2000, fm, years, seed=2024, df=cfg.df)
      lhs = np.random.default_rng(0).uniform(0, 1, (2000, scores.shape[1]))
      reordered = reorder_lhs_by_ranks(lhs, scores)
      tau_star = (2.0/np.pi) * np.arcsin(np.clip(r_lat, -1, 1))
      pairs = build_audit_pairs(pd.DataFrame(tau_star), fm.dimensions, years)
      audit = audit_dependency(reordered, None, r_lat, pairs)
      print(grp, audit.max_kendall_error, audit.min_eigenvalue)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats as sps


# ---- 基准载荷表（Q3_尝试解答.md §4.2）----
# 行 = 随机维度，列 = 公共因子
# 每行平方和均 < 1，保证特异方差存在
BASE_LOADINGS: dict[str, dict[str, float]] = {
    "cost_growth":        {"macro": +0.60, "global_climate":  0.00,
                            "category_demand":  0.00, "category_climate": 0.00},
    "base_demand_change": {"macro": -0.10, "global_climate":  0.00,
                            "category_demand": +0.50, "category_climate": 0.00},
    "price_change":       {"macro": +0.30, "global_climate": -0.30,
                            "category_demand": +0.35, "category_climate": 0.00},
    "yield_change":       {"macro":  0.00, "global_climate": +0.60,
                            "category_demand":  0.00, "category_climate": +0.25},
}


@dataclass
class DependencyConfig:
    """因子 t-Copula 配置参数。

    Attributes:
      df: t 分布自由度 ν（≥ 1；值越大越接近高斯；np.inf 表示高斯极限）
      correlation_scale: κ_R，载荷缩放系数（基准 1.0，扫描 0.5/1.0/1.25）
      temporal_rho: AR(1) 时间自相关系数（所有因子共用，基准 0.5）
      loadings: dim → {factor → loading}，基准载荷表 BASE_LOADINGS
    """
    df: float
    correlation_scale: float
    temporal_rho: float
    loadings: dict[str, dict[str, float]]


@dataclass
class FactorMap:
    """单作物类别的因子载荷映射。

    Attributes:
      dimensions: 随机维度名列表，如 ["cost_growth", "base_demand_change",
                                    "price_change", "yield_change"]
      factors: 因子名列表，如 ["macro", "global_climate",
                              "category_demand", "category_climate"]
      L: (n_dims, n_factors) 载荷矩阵，已乘 κ_R
      rho: (n_factors,) AR(1) 系数向量
    """
    dimensions: list[str]
    factors: list[str]
    L: np.ndarray
    rho: np.ndarray


@dataclass
class DependencyAudit:
    """Kendall 审计结果。

    Attributes:
      min_eigenvalue: R_lat 最小特征值（须 ≥ −1e-10）
      max_kendall_error: 审计对上 |τ* − τ_sample| 最大值
      target_kendall: (d, d) 目标 Kendall 矩阵，来自 R_lat
      sample_kendall: (d, d) 样本 Kendall 矩阵
      sample_spearman: (d, d) 样本 Spearman 矩阵（不与 R_lat 直接比较）
    """
    min_eigenvalue: float
    max_kendall_error: float
    target_kendall: pd.DataFrame
    sample_kendall: pd.DataFrame
    sample_spearman: pd.DataFrame


def build_factor_map(crop_groups: dict[int, str],
                     cfg: DependencyConfig) -> dict[str, FactorMap]:
    """为每个出现的作物类别构造 FactorMap。

    每个类别使用相同的基准载荷表（cfg.loadings），区别在于
    category_demand / category_climate 因子的 AR(1) 路径是类别独立的
    （在 generate_t_rank_scores 中由独立 RNG 抽取）。
    载荷统一乘 κ_R = cfg.correlation_scale，要求行平方和 < 1 以保证
    特异方差存在。

    Args:
      crop_groups: 作物编号 → 类别名（"grain"/"vegetable"/"fungi"）
      cfg: DependencyConfig

    Returns:
      dict 类别名 → FactorMap（按类别名排序）

    Raises:
      ValueError: 若某行载荷平方和 ≥ 1（特异方差不存在）
    """
    groups_present = sorted(set(crop_groups.values()))
    # 维度名和因子名从 cfg.loadings 推断
    dims = list(cfg.loadings.keys())
    factors = list(next(iter(cfg.loadings.values())).keys())
    n_dims = len(dims)
    n_factors = len(factors)

    # 构造基准 L 矩阵（未缩放）
    L_base = np.zeros((n_dims, n_factors), dtype=float)
    for a, dim in enumerate(dims):
        for k, fac in enumerate(factors):
            L_base[a, k] = float(cfg.loadings[dim][fac])

    # 缩放并验证行平方和 < 1
    L_scaled = L_base * cfg.correlation_scale
    row_sq = np.sum(L_scaled ** 2, axis=1)
    if np.any(row_sq >= 1.0 - 1e-12):
        worst = float(row_sq.max())
        raise ValueError(
            f"载荷行平方和 ≥ 1（最大 {worst:.6f}），特异方差不存在；"
            f"请降低 correlation_scale (当前 {cfg.correlation_scale})")

    # 所有因子共用 temporal_rho
    rho = np.full(n_factors, float(cfg.temporal_rho), dtype=float)

    result: dict[str, FactorMap] = {}
    for grp in groups_present:
        result[grp] = FactorMap(
            dimensions=list(dims),
            factors=list(factors),
            L=L_scaled.copy(),
            rho=rho.copy(),
        )
    return result


def build_latent_correlation(factor_map: FactorMap,
                              years: list[int]) -> np.ndarray:
    """构造潜在线性相关矩阵 R_lat。

    R_lat[a_t, b_s] = Σ_k L[a,k]·L[b,k]·ρ_k^|t−s|
                      + δ[a,b]·δ[t,s]·(1 − Σ_k L[a,k]²)

    其中 a, b 为维度索引，t, s 为年份索引；合并索引为
    idx = a·n_years + t_idx。

    Args:
      factor_map: 单类别的 FactorMap
      years: 年份列表（如 [2024, ..., 2030]）

    Returns:
      (n_dims·n_years, n_dims·n_years) 半正定矩阵，对角线为 1
    """
    L = factor_map.L                    # (n_dims, n_factors)
    rho = factor_map.rho                # (n_factors,)
    n_dims, n_factors = L.shape
    n_years = len(years)
    d = n_dims * n_years

    # 预计算 ρ_k^|t−s|：shape (n_factors, n_years, n_years)
    t_grid = np.arange(n_years)
    abs_diff = np.abs(t_grid[:, None] - t_grid[None, :])   # (n_years, n_years)
    rho_power = rho[:, None, None] ** abs_diff[None, :, :]  # (n_factors, n_years, n_years)

    # 因子协方差：C[a,b,t,s] = Σ_k L[a,k]·L[b,k]·ρ_k^|t−s|
    LL = L[:, None, :] * L[None, :, :]  # (n_dims, n_dims, n_factors)
    C = np.einsum('abk,kts->abts', LL, rho_power)  # (n_dims, n_dims, n_years, n_years)

    # 重排为 (d, d)：合并索引 idx = a·n_years + t
    R = C.transpose(0, 2, 1, 3).reshape(d, d)

    # 加特异方差到对角线
    var_idio = 1.0 - np.sum(L ** 2, axis=1)   # (n_dims,)，每维特异方差
    for a in range(n_dims):
        for ti in range(n_years):
            idx = a * n_years + ti
            R[idx, idx] += var_idio[a]

    # 对称化（消除浮点误差）+ 对角线归一
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)
    return R


def generate_t_rank_scores(n: int, factor_map: FactorMap,
                           years: list[int], seed: int,
                           df: float) -> np.ndarray:
    """生成 t-Copula 秩分数。

    每个情景 ω 抽取一个卡方 S_ω ~ χ²(df)，由该情景内全部维度和全部
    年份共享。因子路径采用稳态初始化 g[:,2024]~N(0,1)，递推仅用于
    2025—2030，不得改成零初始化。

    z[a,t,ω] = (Σ_k L[a,k]·g[k,t,ω]
               + sqrt(1−Σ_k L[a,k]²)·ε[a,t,ω]) / sqrt(S_ω/df)

    Args:
      n: 情景数
      factor_map: 单类别的 FactorMap
      years: 年份列表
      seed: 随机种子
      df: t 分布自由度（np.inf 表示高斯极限，不抽取卡方）

    Returns:
      (n, n_dims·n_years) t 分布分数矩阵，列序为
      [dim0_year0, dim0_year1, ..., dim0_year6, dim1_year0, ...]

    Note:
      每次调用独立抽取卡方尺度。若需跨类别共享卡方以诱导跨组尾部
      相关，调用方应在更高层构造合并 FactorMap 或预抽取卡方后传入。
      Kendall/Spearman 审计不受此影响（秩相关不受卡方缩放影响）。
    """
    rng = np.random.default_rng(seed)
    L = factor_map.L                     # (n_dims, n_factors)
    rho = factor_map.rho                 # (n_factors,)
    n_dims, n_factors = L.shape
    n_years = len(years)

    # ---- 因子 AR(1) 路径 g[k, t, ω] ----
    g = np.empty((n_factors, n_years, n), dtype=float)
    # 稳态初始化 g[k, 0, :] ~ N(0, 1)
    g[:, 0, :] = rng.standard_normal((n_factors, n))
    # 递推 g[k, t, :] = ρ_k·g[k, t-1, :] + sqrt(1−ρ_k²)·ν[k, t, :]
    for ti in range(1, n_years):
        nu = rng.standard_normal((n_factors, n))
        g[:, ti, :] = (rho[:, None] * g[:, ti - 1, :]
                       + np.sqrt(1.0 - rho ** 2)[:, None] * nu)

    # ---- 特异噪声 ε[a, t, ω] ~ N(0, 1) ----
    epsilon = rng.standard_normal((n_dims, n_years, n))

    # ---- 卡方尺度 S_ω ~ χ²(df)，每情景共享全部维度与年份 ----
    if np.isinf(df) or df > 1e6:
        # 高斯极限：不缩放
        scale = np.ones(n, dtype=float)
    else:
        S = rng.chisquare(df, size=n)            # (n,) 每情景一个
        scale = np.sqrt(S / df)                  # (n,)

    # ---- t 分数 z[a, t, ω] ----
    # 因子部分：Σ_k L[a,k]·g[k,t,ω]
    factor_part = np.einsum('ak,ktw->atw', L, g)   # (n_dims, n_years, n)
    # 特异部分：sqrt(1−Σ_k L[a,k]²)·ε[a,t,ω]
    idio_scale = np.sqrt(1.0 - np.sum(L ** 2, axis=1))  # (n_dims,)
    idio_part = idio_scale[:, None, None] * epsilon     # (n_dims, n_years, n)
    numerator = factor_part + idio_part                   # (n_dims, n_years, n)

    z = numerator / scale[None, None, :]                # (n_dims, n_years, n)

    # 重排为 (n, n_dims·n_years)：scores[ω, a·n_years+t] = z[a, t, ω]
    scores = z.transpose(2, 0, 1).reshape(n, n_dims * n_years)
    return scores


def reorder_lhs_by_ranks(lhs: np.ndarray,
                          scores: np.ndarray) -> np.ndarray:
    """按 t 分数秩重排 LHS 列，保留边际分层并注入相关结构。

    对每列 j：将 LHS 第 j 列的排序值按 scores 第 j 列的秩分配。
    即第 k 小分数位置获得第 k 小 LHS 值，使输出秩相关与分数一致。
    这等价于 Iman-Conover 方法中按秩匹配边际的分步实现。

    Args:
      lhs: (n, d) LHS 样本（每列一组边际分层值）
      scores: (n, d) t-Copula 分数

    Returns:
      (n, d) 重排后的样本，边际不变、相关结构与 scores 一致

    Raises:
      ValueError: 形状不匹配
    """
    if lhs.shape != scores.shape:
        raise ValueError(
            f"形状不匹配: lhs {lhs.shape} vs scores {scores.shape}")
    n, d = lhs.shape
    output = np.empty_like(lhs, dtype=float)
    for j in range(d):
        # order[k] = 第 k 小分数的原始索引
        order = np.argsort(scores[:, j], kind='mergesort')  # 稳定排序
        # 第 k 小分数位置 ← 第 k 小 LHS 值
        output[order, j] = np.sort(lhs[:, j])
    return output


def build_audit_pairs(target_kendall: pd.DataFrame,
                       dimensions: list[str], years: list[int],
                       threshold: float = 0.10) -> set[tuple[int, int]]:
    """构造 Kendall 审计对集合。

    包含两类无序对（均为合并索引 idx = a·n_years + t_idx）：
      1. 所有 |τ*| ≥ threshold 的非对角无序对
      2. 同一维度相邻年份的全部对（强制并入，不得手工删减）

    Args:
      target_kendall: (d, d) 目标 Kendall 矩阵 DataFrame
      dimensions: 维度名列表
      years: 年份列表
      threshold: τ* 绝对值门槛（默认 0.10）

    Returns:
      无序对集合 {(i, j), ...}，i < j
    """
    n_dims = len(dimensions)
    n_years = len(years)
    d = n_dims * n_years
    pairs: set[tuple[int, int]] = set()

    # 1. |τ*| ≥ threshold 的非对角无序对
    vals = target_kendall.values
    for i in range(d):
        for j in range(i + 1, d):
            if abs(vals[i, j]) >= threshold:
                pairs.add((i, j))

    # 2. 同维度相邻年份对（强制，不得删减）
    for a in range(n_dims):
        for ti in range(n_years - 1):
            idx1 = a * n_years + ti
            idx2 = a * n_years + ti + 1
            pairs.add((idx1, idx2))   # 已满足 idx1 < idx2

    return pairs


def _weighted_kendall_tau(x: np.ndarray, y: np.ndarray,
                           w: np.ndarray) -> float:
    """加权 Kendall τ（观测权重版）。

    τ = (Σ_{i<j} w_i·w_j·sgn(x_i−x_j)·sgn(y_i−y_j))
        / (Σ_{i<j} w_i·w_j)

    用于缩减样本的加权 Kendall 审计。O(n²) 但 n 通常 ≤ 50。
    x 平局或 y 平局的对不计入分子。
    """
    n = len(x)
    if n < 2:
        return 0.0
    # 按 x 稳定排序
    order = np.argsort(x, kind='mergesort')
    x_s = x[order]
    y_s = y[order]
    w_s = w[order]

    numerator = 0.0
    for i in range(n - 1):
        dx = x_s[i + 1:] - x_s[i]          # ≥ 0（已排序）
        dy = y_s[i + 1:] - y_s[i]
        contrib = np.sign(dx) * np.sign(dy)
        contrib[dx == 0] = 0.0             # x 平局不计
        numerator += w_s[i] * np.sum(w_s[i + 1:] * contrib)

    W = w_s.sum()
    denominator = (W ** 2 - np.sum(w_s ** 2)) / 2.0
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _weighted_spearman(x: np.ndarray, y: np.ndarray,
                         w: np.ndarray) -> float:
    """加权 Spearman ρ（观测权重版）。

    对秩做加权 Pearson 相关。用于缩减样本的 Spearman 审计。
    """
    n = len(x)
    if n < 2:
        return 0.0
    # 计算秩（稳定排序，0-indexed）
    rank_x = np.empty(n, dtype=float)
    rank_y = np.empty(n, dtype=float)
    rank_x[np.argsort(x, kind='mergesort')] = np.arange(n)
    rank_y[np.argsort(y, kind='mergesort')] = np.arange(n)

    W = w.sum()
    mx = np.sum(w * rank_x) / W            # 加权均值
    my = np.sum(w * rank_y) / W
    dx = rank_x - mx
    dy = rank_y - my
    cov = np.sum(w * dx * dy) / W
    vx = np.sum(w * dx ** 2) / W           # 加权方差
    vy = np.sum(w * dy ** 2) / W
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / np.sqrt(vx * vy))


def audit_dependency(samples: np.ndarray,
                     weights: np.ndarray | None,
                     r_lat: np.ndarray,
                     audit_pairs: set[tuple[int, int]]) -> DependencyAudit:
    """计算样本 Kendall/Spearman 并与 R_lat 目标对比。

    目标 Kendall：τ* = (2/π)·arcsin(R_lat)
    样本 Kendall：weights=None 用 scipy.stats.kendalltau（O(n log n)）；
                  weights 非空用观测权重加权 Kendall（O(n²)，n≤50）。
    最大误差仅在 audit_pairs 上计算。

    注意：R_lat 是潜在线性相关矩阵，不与样本 Spearman 直接比较。
    Spearman 矩阵仅作为诊断输出，不参与误差门槛判定。

    Args:
      samples: (n, d) 重排后样本（或 t 分数，秩不变）
      weights: (n,) 情景权重或 None（None 表示等权原始样本）
      r_lat: (d, d) 潜在线性相关矩阵
      audit_pairs: 审计对集合

    Returns:
      DependencyAudit

    Raises:
      ValueError: R_lat 形状与 samples 列数不匹配
    """
    n, d = samples.shape
    if r_lat.shape != (d, d):
        raise ValueError(
            f"R_lat 形状 {r_lat.shape} 与 samples 列数 {d} 不匹配")

    # ---- 目标 Kendall：τ* = (2/π)·arcsin(R_lat) ----
    r_clamped = np.clip(r_lat, -1.0, 1.0)
    tau_star = (2.0 / np.pi) * np.arcsin(r_clamped)
    labels = list(range(d))

    # ---- 样本 Kendall / Spearman 矩阵 ----
    sample_tau = np.eye(d, dtype=float)
    sample_rho = np.eye(d, dtype=float)

    if weights is None:
        # 无权重：用 scipy.stats（C 优化，O(n log n)）
        for i in range(d):
            for j in range(i + 1, d):
                tau, _ = sps.kendalltau(samples[:, i], samples[:, j])
                rho, _ = sps.spearmanr(samples[:, i], samples[:, j])
                sample_tau[i, j] = sample_tau[j, i] = float(tau)
                sample_rho[i, j] = sample_rho[j, i] = float(rho)
    else:
        # 加权：用自定义加权 Kendall / Spearman
        for i in range(d):
            for j in range(i + 1, d):
                tau = _weighted_kendall_tau(samples[:, i], samples[:, j], weights)
                rho = _weighted_spearman(samples[:, i], samples[:, j], weights)
                sample_tau[i, j] = sample_tau[j, i] = tau
                sample_rho[i, j] = sample_rho[j, i] = rho

    # ---- 最大 Kendall 误差（仅审计对）----
    max_err = 0.0
    for (i, j) in audit_pairs:
        err = abs(tau_star[i, j] - sample_tau[i, j])
        if err > max_err:
            max_err = err

    # ---- R_lat 最小特征值 ----
    eigvals = np.linalg.eigvalsh((r_lat + r_lat.T) / 2.0)
    min_eig = float(eigvals[0])

    return DependencyAudit(
        min_eigenvalue=min_eig,
        max_kendall_error=float(max_err),
        target_kendall=pd.DataFrame(tau_star, index=labels, columns=labels),
        sample_kendall=pd.DataFrame(sample_tau, index=labels, columns=labels),
        sample_spearman=pd.DataFrame(sample_rho, index=labels, columns=labels),
    )
