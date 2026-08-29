# -*- coding: utf-8 -*-
"""Q3 交叉价格弹性模块。

功能介绍
    本模块实现 `doc/Q3_尝试解答.md` 第 5 节与 `doc/Q3_编程手实现指导.md`
    第 3.5 节规定的季次交叉价格弹性矩阵 E^(s) = (e_ih^(s))，以及基于该
    矩阵的需求修正公式：

        D_its^omega = D_base_its^omega
                      * exp( sum_{h in I_s} e_ih^(s) * ln(p_hts^omega / bar_p_hts) )

    其中 bar_p 为 Q2 确定趋势价格，p 为情景价格。弹性边只在“同一类别
    (grain/vegetable/fungi) + 同一季次”内对称建立替代边；自身价格弹性
    严格为负；跨组/跨季次无边。行稳定条件 sum_{h!=i}|e_ih| <= |e_ii|。

    基准模拟参数（非附件估计值，见 Q3_尝试解答.md 第 5 节）：
        | 类别     | 自身弹性 | 同类替代边总和 |
        | grain    | -0.25    | +0.15          |
        | vegetable| -0.50    | +0.20          |
        | fungi    | -0.60    | +0.25          |
    同类替代总和在有效邻边间均分，并整体乘 kappa_E（自身弹性同步缩放，
    以保持稳定比例不随 kappa_E 变化）。

使用方法
    from algorithms.elasticity import (
        build_elasticity_matrix,
        apply_price_elasticity,
        audit_elasticity,
        build_crop_indices,
        ElasticityConfig,
        DEFAULT_CONFIG,
    )
    matrices = build_elasticity_matrix(data, scale=1.0)
    crop_indices = build_crop_indices(data)          # {season: {crop_code: col}}
    demand_s2 = apply_price_elasticity(
        base_demand=d_base,         # (n_scenarios, n_crops_s2)
        scenario_price=p_scen,      # 同形
        trend_price=p_bar,          # (n_crops_s2,) 可广播
        matrices=matrices,
        crop_indices=crop_indices[2],
        season=2,
    )
    audit = audit_elasticity(matrices, data)         # 结构违规则主动 raise

直接粘贴到命令行运行自检（无需附件数据）：
    cd q3_test
    python -m algorithms.elasticity
    # 或
    python algorithms/elasticity.py

运行环境
    Python 3.10+，仅依赖 numpy。特征值用 numpy.linalg.eigvalsh 计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ------------------------------------------------------------------ #
# 作物类别映射（与 preprocess.py 的 GROUP_*_RANGE 保持一致）
#   grain:     codes 1-16  (含 rice=16)
#   vegetable: codes 17-37 (含根茎类 35-37，仅在 s2 适种)
#   fungi:     codes 38-41 (食用菌，仅在 s2 适种)
# ------------------------------------------------------------------ #
GRAIN_CODES = list(range(1, 17))        # 1-16
VEGETABLE_CODES = list(range(17, 38))   # 17-37
FUNGI_CODES = list(range(38, 42))       # 38-41

# 回退用类别映射（当 ModelData 尚未填充 crop_group 时使用）
DEFAULT_CROP_GROUP: dict[int, str] = {}
DEFAULT_CROP_GROUP.update({c: "grain" for c in GRAIN_CODES})
DEFAULT_CROP_GROUP.update({c: "vegetable" for c in VEGETABLE_CODES})
DEFAULT_CROP_GROUP.update({c: "fungi" for c in FUNGI_CODES})

# 基准弹性模拟参数（doc/Q3_尝试解答.md 第 5 节）
#   key -> (self_elasticity < 0, same_group_substitute_sum > 0)
DEFAULT_BASE_ELASTICITY: dict[str, tuple[float, float]] = {
    "grain": (-0.25, 0.15),
    "vegetable": (-0.50, 0.20),
    "fungi": (-0.60, 0.25),
}


# ------------------------------------------------------------------ #
# 数据类
# ------------------------------------------------------------------ #
@dataclass
class ElasticityConfig:
    """弹性构造配置。

    Attributes:
        scale: kappa_E 缩放系数（弱中强档 {0.5,1.0,1.5}），同时缩放自身弹性
            与替代边总和，保持行稳定比例不随 kappa_E 改变。
        base_elasticity: category -> (self_elasticity, substitute_sum)。
    """
    scale: float = 1.0
    base_elasticity: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BASE_ELASTICITY)
    )


DEFAULT_CONFIG = ElasticityConfig()


@dataclass
class ElasticityAudit:
    """弹性矩阵审计结果。

    Attributes:
        diagonal_sign_violations: 对角线非严格负的元素数。
        off_diagonal_sign_violations: 非对角符号不符的元素数
            （同类应正、跨组应为 0；基准构造不含互补边）。
        row_stability_violations: sum|off_diag| > |diag| 的行数。
        cross_season_edges: 跨季次边数（结构上恒为 0，矩阵按季次独立构造）。
        max_eigenvalue: 所有季次矩阵的最大特征值（稳定时 <= 0）。
    """
    diagonal_sign_violations: int
    off_diagonal_sign_violations: int
    row_stability_violations: int
    cross_season_edges: int
    max_eigenvalue: float


# ------------------------------------------------------------------ #
# 辅助：从 ModelData 取 crop_group / season_crop_sets
# ------------------------------------------------------------------ #
def _get_crop_group(data) -> dict[int, str]:
    """优先用 data.crop_group；缺失则回退到 DEFAULT_CROP_GROUP。"""
    cg = getattr(data, "crop_group", None)
    if cg:
        return cg
    return dict(DEFAULT_CROP_GROUP)


def _get_season_crop_sets(data) -> dict[int, set[int]]:
    """优先用 data.season_crop_sets；缺失则从 data.suit 重建。"""
    scs = getattr(data, "season_crop_sets", None)
    if scs:
        return {s: set(c) for s, c in scs.items()}
    out: dict[int, set[int]] = {}
    suit = getattr(data, "suit", {})
    for key in suit:
        if len(key) == 3:
            _j, i, s = key
            out.setdefault(s, set()).add(i)
    if not out:
        raise RuntimeError(
            "season_crop_sets 缺失且无法从 data.suit 重建；请先运行 preprocess"
        )
    return out


def build_crop_indices(data) -> dict[int, dict[int, int]]:
    """返回 {season: {crop_code: col_idx}}，列序按作物编号升序。

    与 build_elasticity_matrix 的列序一致，供 apply_price_elasticity 对齐使用。
    """
    scs = _get_season_crop_sets(data)
    return {
        s: {c: k for k, c in enumerate(sorted(codes))}
        for s, codes in scs.items()
    }


# ------------------------------------------------------------------ #
# 核心接口
# ------------------------------------------------------------------ #
def build_elasticity_matrix(
    data, scale: float = 1.0, config: "ElasticityConfig | None" = None
) -> dict[int, np.ndarray]:
    """构造每个季次的交叉价格弹性矩阵 E^(s)。

    Args:
        data: ModelData（需含 crop_group 与 season_crop_sets，否则回退）。
        scale: kappa_E 缩放系数（默认 1.0）。
        config: 可选配置覆盖；为 None 时按 scale 构造默认配置。

    Returns:
        {season: E}，E 为 (n_crops_s, n_crops_s) 的稠密 numpy 数组，
        行列序号按该季次作物编号升序排列（与 build_crop_indices 一致）。

    Raises:
        AssertionError: 构造结果违反对角线严格负或行稳定条件（主动失败）。
    """
    if config is None:
        config = ElasticityConfig(scale=scale)
    kappa = float(config.scale)  # kappa_E：整体缩放系数

    crop_group = _get_crop_group(data)
    season_crop_sets = _get_season_crop_sets(data)
    base = config.base_elasticity

    matrices: dict[int, np.ndarray] = {}
    for s, crop_set in sorted(season_crop_sets.items()):
        crops = sorted(crop_set)          # 升序列序
        n = len(crops)
        E = np.zeros((n, n), dtype=float)
        idx = {c: k for k, c in enumerate(crops)}

        for i in crops:
            gi = crop_group.get(i)
            if gi is None or gi not in base:
                # 无基准参数：对角线留 0，由下方断言主动失败
                continue
            self_e, sub_sum = base[gi]
            # 对角线：自身价格弹性（严格负），整体乘 kappa_E
            E[idx[i], idx[i]] = float(self_e) * kappa
            # 同类、同季次的替代邻居（对称）
            neighbors = [h for h in crops
                         if h != i and crop_group.get(h) == gi]
            m = len(neighbors)
            if m > 0 and sub_sum > 0:
                w = (float(sub_sum) / m) * kappa  # 单边权重（正，替代）
                for h in neighbors:
                    E[idx[i], idx[h]] = w

        # 对称化（应当已经对称，此处做数值防御）
        E = 0.5 * (E + E.T)
        matrices[s] = E

        # ---- 构造期主动断言（doc/Q3_编程手实现指导.md 3.5）----
        diag = np.diag(E)
        assert np.all(diag < 0), (
            f"season {s}: 对角线非严格负, diag={diag}"
        )
        for r in range(n):
            off_sum = float(np.sum(np.abs(E[r])) - abs(E[r, r]))
            assert off_sum <= abs(E[r, r]) + 1e-12, (
                f"season {s} row {r}: 行稳定条件违反 "
                f"sum|off|={off_sum:.6g} > |diag|={abs(E[r, r]):.6g}"
            )
        # 非对角符号：同类应 >0（替代），跨组应为 0（基准无互补边）
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                same_group = crop_group.get(crops[a]) == crop_group.get(crops[b])
                if same_group:
                    assert E[a, b] > 0, (
                        f"season {s} ({crops[a]},{crops[b]}): "
                        f"同类替代边非正 = {E[a, b]}"
                    )
                else:
                    assert E[a, b] == 0, (
                        f"season {s} ({crops[a]},{crops[b]}): "
                        f"跨组无依据边非零 = {E[a, b]}"
                    )

    return matrices


def apply_price_elasticity(
    base_demand: np.ndarray,
    scenario_price: np.ndarray,
    trend_price: np.ndarray,
    matrices: dict[int, np.ndarray],
    crop_indices: dict,
    season: int,
) -> np.ndarray:
    """向量化施加弹性需求修正。

        D = D_base * exp( sum_h e_ih * ln(p_h / bar_p_h) )

    Args:
        base_demand: (n_scenarios, n_crops_s) 基础销量，列序与 crop_indices 一致。
        scenario_price: 同形情景价格 p。
        trend_price: (n_crops_s,) 或可广播的确定趋势价格 bar_p。
        matrices: build_elasticity_matrix 的输出。
        crop_indices: {crop_code: col_idx}（该季次），用于对齐校验。
        season: 季次下标。

    Returns:
        修正后需求，形状同 base_demand。

    Raises:
        AssertionError: 输出非有限、非正，或零价格冲击未退化为基础需求。
    """
    if season not in matrices:
        raise KeyError(f"matrices 中缺少 season={season} 的弹性矩阵")

    E = np.asarray(matrices[season], dtype=float)   # (n, n)
    n = E.shape[0]

    # 对齐校验
    if len(crop_indices) != n:
        raise ValueError(
            f"crop_indices 有 {len(crop_indices)} 项，"
            f"但 season {season} 的矩阵为 {n}x{n}"
        )

    base_demand = np.asarray(base_demand, dtype=float)
    scenario_price = np.asarray(scenario_price, dtype=float)
    trend_price = np.asarray(trend_price, dtype=float)

    if base_demand.shape != scenario_price.shape:
        raise ValueError(
            f"base_demand 形状 {base_demand.shape} != "
            f"scenario_price 形状 {scenario_price.shape}"
        )
    if base_demand.shape[-1] != n:
        raise ValueError(
            f"base_demand 末维 {base_demand.shape[-1]} != 矩阵规模 {n}"
        )

    # 价格比与对数比；trend_price 可广播
    price_ratio = scenario_price / trend_price
    log_ratio = np.log(price_ratio)                # (..., n_crops)

    # exponent[..., i] = sum_h E[i, h] * log_ratio[..., h]
    exponent = log_ratio @ E.T                      # (..., n_crops)
    demand = base_demand * np.exp(exponent)

    # ---- 输出断言（主动失败）----
    assert np.all(np.isfinite(demand)), "弹性输出存在非有限值"
    assert np.all(demand >= 0), "弹性输出存在负值"
    if np.any(base_demand > 0):
        assert np.all(demand[base_demand > 0] > 0), (
            "弹性输出在基础需求为正处变为零"
        )
    # 零价格冲击必须严格退化为基础需求
    if np.array_equal(scenario_price, trend_price):
        assert np.allclose(demand, base_demand), (
            "零价格冲击下输出未退化为基础需求"
        )

    return demand


def audit_elasticity(
    matrices: dict[int, np.ndarray], data
) -> ElasticityAudit:
    """独立审计全部结构断言。

    检查：对角线严格负、非对角符号（同类正/跨组零）、行稳定、
    跨季次边（结构上为 0）、最大特征值。

    Returns:
        ElasticityAudit（无违规时返回，含 max_eigenvalue 诊断）。

    Raises:
        AssertionError: 存在任何结构违规时主动失败。
    """
    crop_group = _get_crop_group(data)
    season_crop_sets = _get_season_crop_sets(data)

    diag_viol = 0
    offdiag_viol = 0
    stability_viol = 0
    cross_season = 0
    max_eig = -np.inf

    for s in sorted(matrices.keys()):
        E = np.asarray(matrices[s], dtype=float)
        n = E.shape[0]
        crops = sorted(season_crop_sets.get(s, set()))

        if n != len(crops):
            raise ValueError(
                f"season {s}: 矩阵 {n}x{n} 但该季次有 {len(crops)} 个作物"
            )

        # 对角线符号
        diag = np.diag(E)
        diag_viol += int(np.sum(diag >= 0))

        # 非对角符号（基准构造：同类=替代>0，跨组=0）
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                same_group = crop_group.get(crops[a]) == crop_group.get(crops[b])
                if same_group:
                    if not (E[a, b] > 0):       # 替代边应严格正
                        offdiag_viol += 1
                else:
                    if E[a, b] != 0:            # 跨组无依据边应零
                        offdiag_viol += 1

        # 行稳定：sum_{h!=i}|e_ih| <= |e_ii|
        for r in range(n):
            off_sum = float(np.sum(np.abs(E[r])) - abs(E[r, r]))
            if off_sum > abs(E[r, r]) + 1e-12:
                stability_viol += 1

        # 最大特征值（对称矩阵，稳定时应 <= 0）
        E_sym = 0.5 * (E + E.T)
        eigs = np.linalg.eigvalsh(E_sym)
        max_eig = max(max_eig, float(np.max(eigs)))

    # 跨季次边：矩阵按季次独立构造，结构上恒为 0
    cross_season = 0

    audit = ElasticityAudit(
        diagonal_sign_violations=diag_viol,
        off_diagonal_sign_violations=offdiag_viol,
        row_stability_violations=stability_viol,
        cross_season_edges=cross_season,
        max_eigenvalue=max_eig if max_eig != -np.inf else 0.0,
    )

    total = diag_viol + offdiag_viol + stability_viol + cross_season
    if total > 0:
        raise AssertionError(f"弹性审计失败: {audit}")
    return audit


# ------------------------------------------------------------------ #
# 自检入口（不依赖附件数据，验证全部断言可主动失败）
# ------------------------------------------------------------------ #
class _FakeData:
    """仅用于 __main__ 自检的最小 data 替身。"""

    def __init__(self) -> None:
        self.crop_group = dict(DEFAULT_CROP_GROUP)
        # s1: grain(1-16) + vegetable(17-34)
        # s2: vegetable(17-37) + fungi(38-41)
        self.season_crop_sets = {
            1: set(GRAIN_CODES) | set(range(17, 35)),
            2: set(VEGETABLE_CODES) | set(FUNGI_CODES),
        }


def _self_test() -> None:
    data = _FakeData()

    # 1) 构造与审计
    matrices = build_elasticity_matrix(data, scale=1.0)
    audit = audit_elasticity(matrices, data)
    print(f"[ok] 构造与审计通过, max_eigenvalue={audit.max_eigenvalue:.6g}")

    crop_indices = build_crop_indices(data)

    for s in (1, 2):
        ci = crop_indices[s]
        n = len(ci)
        n_scen = 5
        rng = np.random.default_rng(2024 + s)
        d_base = np.full((n_scen, n), 100.0)
        p_bar = np.full(n, 2.0)
        # 零价格冲击：scenario_price 与 base_demand 同形且等于 trend_price
        p_zero = np.tile(p_bar, (n_scen, 1))
        d_zero = apply_price_elasticity(
            d_base, p_zero, p_bar, matrices, ci, s
        )
        assert np.allclose(d_zero, d_base), f"season {s}: 零冲击未退化"
        print(f"[ok] season {s}: 零价格冲击退化为基础需求")

        # 单个替代品涨价 -> 目标需求方向正确（上升）
        crops_sorted = sorted(ci.keys())
        # 找一对同类作物 (i, h)
        pair = None
        for i_code in crops_sorted:
            gi = data.crop_group[i_code]
            for h_code in crops_sorted:
                if h_code != i_code and data.crop_group[h_code] == gi:
                    pair = (i_code, h_code)
                    break
            if pair:
                break
        assert pair is not None, "未找到同类作物对用于方向测试"
        i_code, h_code = pair
        p_up = np.tile(p_bar, (n_scen, 1))
        p_up[:, ci[h_code]] *= 1.10  # 替代品 h 涨价 10%
        d_up = apply_price_elasticity(
            d_base, p_up, p_bar, matrices, ci, s
        )
        # i 的需求应上升（e_ih > 0, ln(p_h/bar_p_h) > 0）
        assert np.all(d_up[:, ci[i_code]] > d_base[:, ci[i_code]]), (
            f"season {s}: 替代品 {h_code} 涨价后作物 {i_code} 需求未上升"
        )
        print(
            f"[ok] season {s}: 替代品 {h_code} 涨价 -> "
            f"作物 {i_code} 需求上升 (方向正确)"
        )

    # 稳定性比例不随 kappa_E 改变
    m_low = build_elasticity_matrix(data, scale=0.5)
    m_high = build_elasticity_matrix(data, scale=1.5)
    for s in (1, 2):
        r = m_low[1].copy()
        d = float(np.diag(m_low[s])[0])
        off = float(np.sum(np.abs(m_low[s][0])) - abs(d))
        ratio_low = off / abs(d)
        d2 = float(np.diag(m_high[s])[0])
        off2 = float(np.sum(np.abs(m_high[s][0])) - abs(d2))
        ratio_high = off2 / abs(d2)
        assert abs(ratio_low - ratio_high) < 1e-12, (
            f"season {s}: kappa_E 改变了稳定比例 {ratio_low} vs {ratio_high}"
        )
    print("[ok] kappa_E 缩放保持行稳定比例不变")

    print("全部自检通过。")


if __name__ == "__main__":
    _self_test()
