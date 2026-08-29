# -*- coding: utf-8 -*-
"""Q3 可视化 — 9 张逻辑图（doc/Q3_编程手实现指导.md section 3.14）。

功能介绍
    生成问题 3 的 9 张正式逻辑图，分三类各 3 张：
      原始/模拟数据图:
        1. raw_q3_target_sample_kendall   目标 vs 样本 Kendall 相关热力图
        2. raw_q3_marginal_ranges         边际参数变化率箱线图（需求/产量/成本/价格）
        3. raw_q3_elasticity_network      交叉价格弹性网络图（节点=作物，边=替代）
      过程图:
        4. process_q3_scenario_reduction  缩减前后利润分布 + 利润分层代表
        5. process_q3_risk_frontier       风险前沿 E[Pi] vs CVaR，标注膝点
        6. process_q3_ablation            四组消融对比柱状图
      结果图:
        7. result_q3_area_heatmap         种植面积热力图（54 地块 × 7 年）
        8. result_q3_paired_profit_difference  Q2-Q3 配对利润差直方图
        9. result_q3_q2_comparison        Q2 vs Q3 利润分布对比（小提琴+箱线）

    每张图均输出：SVG 矢量、300 DPI PNG、`_qa/` 下灰度预览。
    缺少所需数据时跳过对应图并在返回字典中不包含该项。

使用方法
    from algorithms.plots import generate_figures
    paths = generate_figures(data, scenarios=scen, reduced=red,
                             frontier_points=frontier, selected_plan=plan,
                             q2_plan=q2plan, q3_profits=q3p, q2_profits=q2p,
                             ablation_df=abl, dependency_audit=daudit,
                             elasticity_matrices=emat, out_dir=FIG_DIR)
    # 只生成指定图:
    paths = generate_figures(data, figures=["result_q3_area_heatmap"], out_dir=FIG_DIR)

直接粘贴到命令行运行自检（不依赖附件，仅验证导出链路）:
    cd q3_test
    python -m algorithms.plots
    # 或
    python algorithms/plots.py

运行环境
    Python 3.10+，依赖 numpy / matplotlib / pandas。
    使用 Agg 后端，无需图形界面；中文字体回退 Microsoft YaHei / SimHei / DejaVu Sans。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .preprocess import ModelData
from . import paths

# ---- 学术配色（色盲友好）----
C_GRAIN = "#0072B2"      # 粮食蓝
C_VEG = "#009E73"        # 蔬菜绿
C_FUNGI = "#D55E00"      # 食用菌朱
C_Q3 = "#0072B2"
C_Q2 = "#E69F00"
GROUP_COLOR = {"grain": C_GRAIN, "vegetable": C_VEG, "fungi": C_FUNGI}

# 9 张图名称（按文档顺序）
FIGURE_NAMES = [
    "raw_q3_target_sample_kendall",
    "raw_q3_marginal_ranges",
    "raw_q3_elasticity_network",
    "process_q3_scenario_reduction",
    "process_q3_risk_frontier",
    "process_q3_ablation",
    "result_q3_area_heatmap",
    "result_q3_paired_profit_difference",
    "result_q3_q2_comparison",
]


def _setup_style() -> None:
    """配置 matplotlib 学术风格与中文字体（Agg 后端，无图形界面）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",          # SVG 文本可编辑
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def _save(fig, name: str, out_dir: Path) -> dict:
    """保存图：SVG + 300 DPI PNG + `_qa` 灰度预览。

    Returns:
        {"svg": svg_path, "png": png_path, "qa": qa_path}
    """
    paths_out = {}
    svg_path = out_dir / f"{name}.svg"
    png_path = out_dir / f"{name}.png"
    qa_dir = out_dir / "_qa"
    qa_dir.mkdir(exist_ok=True)
    qa_path = qa_dir / f"{name}_gray.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    # 灰度预览：读回出版 PNG 并按 ITU-R BT.601 加权
    import matplotlib.pyplot as plt2
    fig2 = plt2.figure()
    ax2 = fig2.add_axes([0, 0, 1, 1])      # 全幅面轴，imshow 是 Axes 方法
    img = plt2.imread(str(png_path))
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    ax2.imshow(gray, cmap="gray")
    ax2.axis("off")
    fig2.savefig(qa_path, dpi=150, bbox_inches="tight")
    plt2.close(fig2)
    plt2.close(fig)
    return {"svg": svg_path, "png": png_path, "qa": qa_path}


# ------------------------------------------------------------------ #
# 原始/模拟数据图
# ------------------------------------------------------------------ #
def _fig_target_sample_kendall(dependency_audit) -> "Figure":
    """图1: 目标 Kendall vs 样本 Kendall 相关矩阵热力图对比。"""
    import matplotlib.pyplot as plt
    tgt = dependency_audit.target_kendall        # pd.DataFrame 目标相关
    sam = dependency_audit.sample_kendall        # pd.DataFrame 样本相关
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    vmax = max(1e-6, float(max(np.abs(tgt.values).max(),
                               np.abs(sam.values).max())))
    for ax, df, title in ((axes[0], tgt, "目标 Kendall τ"),
                          (axes[1], sam, "样本 Kendall τ")):
        im = ax.imshow(df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("维度")
        ax.set_ylabel("维度")
        fig.colorbar(im, ax=ax, fraction=0.046, label="τ")
    err = float(getattr(dependency_audit, "max_kendall_error", np.nan))
    fig.suptitle(f"目标 vs 样本 Kendall 相关（最大误差={err:.4f}）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _fig_marginal_ranges(scenarios, data) -> "Figure":
    """图2: 边际参数变化率箱线图（需求/产量/成本/价格相对 2023 基线）。"""
    import matplotlib.pyplot as plt
    n = scenarios.n
    sub_n = min(n, 300)            # 每键抽样上限，控制点量
    sub_keys = 50                  # 每类采样键上限

    def _rates(dct, base_get, key_len):
        """收集 (值-基线)/基线 变化率。base_get(key) 返回基线标量。"""
        vals = []
        for k in list(dct.keys())[:sub_keys]:
            b = base_get(k)
            if not b or b <= 0:
                continue
            arr = np.asarray(dct[k]).ravel()[:sub_n]
            vals.extend(((arr - b) / b).tolist())
        return vals

    demand_r = _rates(scenarios.demand, lambda k: data.D.get((k[0], k[2]), 0.0)
                      if len(k) == 3 else 0.0, 3)
    yield_r = _rates(scenarios.yield_, lambda k: data.q.get((k[0], k[1], k[3]), 0.0)
                     if len(k) == 4 else 0.0, 4)
    cost_r = _rates(scenarios.cost, lambda k: data.c.get((k[0], k[1], k[3]), 0.0)
                    if len(k) == 4 else 0.0, 4)
    price_r = _rates(scenarios.price, lambda k: data.p.get((k[0], k[2]), 0.0)
                     if len(k) == 3 else 0.0, 3)

    fig, ax = plt.subplots(figsize=(8, 5))
    data_boxes = [demand_r, yield_r, cost_r, price_r]
    labels = ["需求", "亩产", "成本", "价格"]
    colors = [C_VEG, C_GRAIN, C_FUNGI, "#CC79A7"]
    try:                                   # matplotlib>=3.9 重命名为 tick_labels
        bp = ax.boxplot(data_boxes, tick_labels=labels, patch_artist=True,
                        showfliers=False, widths=0.55)
    except TypeError:
        bp = ax.boxplot(data_boxes, labels=labels, patch_artist=True,
                        showfliers=False, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    ax.set_ylabel("相对 2023 基线变化率")
    ax.set_title("Q3 边际参数变化率分布（LHS 样本）")
    fig.tight_layout()
    return fig


def _fig_elasticity_network(elasticity_matrices, data) -> "Figure":
    """图3: 交叉价格弹性网络图。节点=作物，边=同类替代（正），节点色=类别。"""
    import matplotlib.pyplot as plt
    if elasticity_matrices is None:           # 缺失则按默认构造
        from .elasticity import build_elasticity_matrix
        elasticity_matrices = build_elasticity_matrix(data, scale=1.0)
    crop_group = getattr(data, "crop_group", {})
    season_crop_sets = getattr(data, "season_crop_sets", {})

    fig, axes = plt.subplots(1, len(elasticity_matrices),
                             figsize=(7 * len(elasticity_matrices), 6.5))
    if len(elasticity_matrices) == 1:
        axes = [axes]
    for ax, (s, E) in zip(axes, sorted(elasticity_matrices.items())):
        E = np.asarray(E)
        codes = sorted(season_crop_sets.get(s, []))
        n = len(codes)
        # 圆形布局，按类别排序使同类相邻
        ordered = sorted(codes, key=lambda c: (crop_group.get(c, ""), c))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(ordered, angles)}
        idx = {c: k for k, c in enumerate(codes)}
        # 画替代边（非对角正值）
        for a in range(n):
            for b in range(a + 1, n):
                w = float(E[a, b])
                if w > 0:
                    xa, ya = pos[codes[a]]
                    xb, yb = pos[codes[b]]
                    ax.plot([xa, xb], [ya, yb], color=C_FUNGI,
                            alpha=0.12, linewidth=0.5, zorder=1)
        # 画节点
        for c in codes:
            x, y = pos[c]
            g = crop_group.get(c, "vegetable")
            ax.scatter(x, y, s=90, c=GROUP_COLOR.get(g, "gray"),
                       edgecolors="black", linewidths=0.6, zorder=3)
            ax.text(x, y, str(c), ha="center", va="center", fontsize=5.5,
                    zorder=4)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"第 {s} 季（{n} 作物）")
    # 图例
    handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=9,
                          markerfacecolor=GROUP_COLOR[g], markeredgecolor="k",
                          label={"grain": "粮食", "vegetable": "蔬菜",
                                 "fungi": "食用菌"}[g])
               for g in ("grain", "vegetable", "fungi")]
    handles.append(plt.Line2D([0], [0], color=C_FUNGI, alpha=0.5, linewidth=1.5,
                              label="替代边"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("交叉价格弹性网络（节点=作物，边=同类替代）", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


# ------------------------------------------------------------------ #
# 过程图
# ------------------------------------------------------------------ #
def _vectorized_proxy_profit(scenarios, data, plan_x, max_n=2000):
    """向量化代理利润（供缩减前后分布图使用）。plan_x: (j,i,t,s)->area。"""
    n = scenarios.n
    idx = np.arange(min(n, max_n))
    profits = np.zeros(len(idx))
    for (j, i, t, s), area in plan_x.items():
        if area <= 0:
            continue
        q = scenarios.yield_.get((j, i, t, s))
        if q is None:
            continue
        c = scenarios.cost.get((j, i, t, s), np.zeros(n))
        p = scenarios.price.get((i, t, s), np.zeros(n))
        d = scenarios.demand.get((i, t, s), np.full(n, np.inf))
        Q = q[idx] * area
        u = np.minimum(Q, d[idx])
        profits += p[idx] * u - c[idx] * area
    return profits


def _fig_scenario_reduction(scenarios, reduced, data) -> "Figure":
    """图4: 情景缩减前后利润分布 + 利润十等频分层代表数。"""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # 左：缩减前后利润分布
    plan_x = {k: v for k, v in data.bar_x.items() if v > 0}   # 2023 基线方案
    before = _vectorized_proxy_profit(scenarios, data, plan_x)
    after = np.asarray(reduced.proxy_profits)
    ax1.hist(before, bins=40, density=True, alpha=0.45, color="#0072B2",
             label=f"缩减前 N={len(before)}")
    if len(after):
        w = np.asarray(reduced.weights)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        ax1.hist(after, bins=20, density=True, weights=w, alpha=0.6,
                 color="#D55E00", label=f"缩减后 K={len(after)}")
    ax1.axvline(before.mean(), color="#0072B2", linestyle="--", linewidth=1,
                label=f"前均值={before.mean():.0f}")
    ax1.set_xlabel("代理利润 (元)")
    ax1.set_ylabel("密度")
    ax1.set_title("缩减前后利润分布")
    ax1.legend(frameon=False, fontsize=8)

    # 右：十等频分层代表数（最低 10% 层尾部保护高亮）
    order = np.argsort(before)
    n = len(before)
    layer_size = max(1, n // 10)
    layers = [order[i * layer_size:(i + 1) * layer_size if i < 9 else n]
              for i in range(10)]
    rep_idx = set(int(i) for i in np.asarray(reduced.indices))
    counts = [sum(1 for i in lay if i in rep_idx) for lay in layers]
    xlab = [f"L{i+1}" for i in range(10)]
    bars = ax2.bar(xlab, counts, color="#009E73", alpha=0.7)
    bars[0].set_color(C_FUNGI)              # 最低 10% 尾部层
    ax2.set_xlabel("利润十等频层（L1=最低 10%）")
    ax2.set_ylabel("代表情景数")
    ax2.set_title(f"分层代表数（K={len(rep_idx)}，尾部保护层高亮）")
    fig.tight_layout()
    return fig


def _fig_risk_frontier(frontier_points, selected_plan) -> "Figure":
    """图5: 风险前沿 E[Pi] vs CVaR，膝点高亮。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pts = frontier_points or []

    def _get(p, keys):
        for k in keys:
            if isinstance(p, dict) and k in p and p[k] is not None:
                return p[k]
        return None

    lam = [_get(p, ("lambda", "risk_lambda", "lam")) for p in pts]
    epi = [_get(p, ("expected_profit", "e_pi", "E_pi", "mean_profit")) for p in pts]
    cvar = [_get(p, ("lower_tail_cvar", "cvar", "LCVaR", "tail_cvar")) for p in pts]
    elig = [_get(p, ("eligible", "feasible")) for p in pts]
    valid = [(l, e, c, el) for l, e, c, el in zip(lam, epi, cvar, elig)
             if e is not None and c is not None]
    if valid:
        ls, es, cs, _ = zip(*valid)
        sc = ax.scatter(cs, es, c=ls, cmap="viridis", s=55,
                        edgecolors="black", linewidths=0.5, zorder=3)
        fig.colorbar(sc, ax=ax, label="风险权重 λ")
        for l, e, c in zip(ls, es, cs):
            ax.annotate(f"λ={l:.1f}", (c, e), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    # 膝点：选中方案的 λ 或最大距离到端点连线
    knee = None
    sel_lam = _get(selected_plan or {}, ("lambda", "risk_lambda"))
    if sel_lam is not None and valid:
        knee = min(valid, key=lambda t: abs(t[0] - sel_lam))
    elif len(valid) >= 3:
        xs = np.array([t[2] for t in valid])
        ys = np.array([t[1] for t in valid])
        # 归一化后计算到首末点连线的最大距离
        xr = xs.max() - xs.min() or 1.0
        yr = ys.max() - ys.min() or 1.0
        xn, yn = (xs - xs.min()) / xr, (ys - ys.min()) / yr
        d = np.abs((yn[0] - yn[-1]) * xn - (xn[0] - xn[-1]) * yn
                   + xn[0] * yn[-1] - xn[-1] * yn[0])
        knee = valid[int(np.argmax(d))]
    if knee is not None:
        ax.scatter([knee[2]], [knee[1]], s=220, facecolors="none",
                   edgecolors=C_FUNGI, linewidths=2.5, zorder=4, label="膝点")
        ax.legend(frameon=False)
    ax.set_xlabel("下尾利润 CVaR (元)")
    ax.set_ylabel("期望利润 E[Π] (元)")
    ax.set_title("Q3 风险前沿（膝点高亮）")
    fig.tight_layout()
    return fig


def _fig_ablation(ablation_df) -> "Figure":
    """图6: 四组消融对比柱状图。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    df = ablation_df
    # 识别组名列与指标列
    group_col = next((c for c in ("group", "config", "name", "方案")
                      if c in df.columns), df.columns[0])
    metric_cols = [c for c in df.columns
                   if c != group_col and pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        ax.text(0.5, 0.5, "无可用数值指标", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    groups = [str(g) for g in df[group_col].tolist()]
    x = np.arange(len(groups))
    width = 0.8 / len(metric_cols)
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"]
    for k, m in enumerate(metric_cols):
        ax.bar(x + (k - (len(metric_cols) - 1) / 2) * width,
               df[m].values, width, label=str(m),
               color=colors[k % len(colors)], alpha=0.85,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("指标值")
    ax.set_title("四组消融对比")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 结果图
# ------------------------------------------------------------------ #
def _extract_plan_areas(plan):
    """从 plan 字典提取 {(j,i,t,s): area}。兼容 {x: {...}} 与直接映射。"""
    if plan is None:
        return {}
    if isinstance(plan, dict) and "x" in plan and isinstance(plan["x"], dict):
        return plan["x"]
    if isinstance(plan, dict) and "area" in plan and isinstance(plan["area"], dict):
        return plan["area"]
    return plan if isinstance(plan, dict) else {}


def _fig_area_heatmap(selected_plan, data) -> "Figure":
    """图7: 种植面积热力图（54 地块 × 7 年，跨作物与季次求和）。"""
    import matplotlib.pyplot as plt
    x = _extract_plan_areas(selected_plan)
    years = data.years
    n_plots = len(data.plot_names)
    mat = np.zeros((len(years), n_plots))      # 行=年, 列=地块
    for (j, i, t, s), area in x.items():
        if t in years and 0 <= j < n_plots:
            mat[years.index(t), j] += float(area)
    fig, ax = plt.subplots(figsize=(15, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu",
                   interpolation="nearest")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    # x 轴：每 5 地块标一次，避免拥挤
    step = 5
    ax.set_xticks(range(0, n_plots, step))
    ax.set_xticklabels([f"P{k+1}" for k in range(0, n_plots, step)])
    ax.set_xlabel("地块编号")
    ax.set_ylabel("年份")
    ax.set_title("Q3 种植面积热力图（54 地块 × 7 年）")
    cbar = fig.colorbar(im, ax=ax, label="面积 (亩)")
    if getattr(cbar, "solids", None) is not None:
        cbar.solids.set_rasterized(False)
    fig.tight_layout()
    return fig


def _fig_paired_profit_difference(q3_profits, q2_profits) -> "Figure":
    """图8: Q2-Q3 配对利润差直方图（共同随机数，逐情景相减）。"""
    import matplotlib.pyplot as plt
    q3 = np.asarray(q3_profits, dtype=float)
    q2 = np.asarray(q2_profits, dtype=float)
    m = min(len(q3), len(q2))
    diff = q3[:m] - q2[:m]                     # 配对差，>0 表示 Q3 更优
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(diff, bins=50, color="#009E73", alpha=0.7, edgecolor="black",
            linewidth=0.3)
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1,
               label="零差")
    ax.axvline(diff.mean(), color=C_FUNGI, linestyle="--", linewidth=1.5,
               label=f"均值={diff.mean():.0f}")
    ax.axvline(np.median(diff), color=C_Q3, linestyle=":", linewidth=1.5,
               label=f"中位数={np.median(diff):.0f}")
    pos_pct = 100.0 * np.mean(diff > 0) if m else 0.0
    ax.set_xlabel("配对利润差 Π_Q3 − Π_Q2 (元)")
    ax.set_ylabel("频数")
    ax.set_title(f"Q2-Q3 配对利润差（N={m}，Q3 更优占比 {pos_pct:.1f}%）")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _fig_q2_comparison(q3_profits, q2_profits) -> "Figure":
    """图9: Q2 vs Q3 利润分布对比（小提琴 + 叠加箱线）。"""
    import matplotlib.pyplot as plt
    q3 = np.asarray(q3_profits, dtype=float)
    q2 = np.asarray(q2_profits, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    data_list = [q2, q3]
    pos = [1, 2]
    parts = ax.violinplot(data_list, positions=pos, showmeans=False,
                          showmedians=False, showextrema=False)
    for body, c in zip(parts["bodies"], (C_Q2, C_Q3)):
        body.set_facecolor(c)
        body.set_alpha(0.45)
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
    bp = ax.boxplot(data_list, positions=pos, widths=0.18, patch_artist=True,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], (C_Q2, C_Q3)):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # 均值标记
    for p, arr, c in zip(pos, data_list, (C_Q2, C_Q3)):
        ax.scatter(p, arr.mean(), marker="D", color="white",
                   edgecolors=c, s=45, zorder=5, label="均值" if p == 1 else "")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"Q2 基线\n(N={len(q2)})",
                        f"Q3 风险方案\n(N={len(q3)})"])
    ax.set_ylabel("利润 (元)")
    ax.set_title("Q2 vs Q3 样本外利润分布对比")
    # 统计摘要文本
    txt = (f"Q2 均值={q2.mean():.0f}  最差10%={np.sort(q2)[:max(1,len(q2)//10)].mean():.0f}\n"
           f"Q3 均值={q3.mean():.0f}  最差10%={np.sort(q3)[:max(1,len(q3)//10)].mean():.0f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.8))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 主入口
# ------------------------------------------------------------------ #
def generate_figures(data: ModelData, scenarios=None, reduced=None,
                     frontier_points: list = None, selected_plan: dict = None,
                     q2_plan: dict = None, q3_profits: np.ndarray = None,
                     q2_profits: np.ndarray = None,
                     ablation_df=None, dependency_audit=None,
                     elasticity_matrices=None,
                     out_dir: Path = None,
                     figures: list = None) -> dict:
    """生成 Q3 图表。返回 {fig_name: {svg, png, qa}} 路径字典。

    Args:
      data: ModelData（必填）。
      scenarios: Q3ScenarioSet（原始情景）。
      reduced: ReducedScenarioSet（缩减后情景）。
      frontier_points: 风险前沿点列表（每点为 dict）。
      selected_plan: Q3 选中方案（含面积映射）。
      q2_plan: Q2 基线方案（可选）。
      q3_profits / q2_profits: 样本外配对利润数组。
      ablation_df: 四组消融结果 DataFrame。
      dependency_audit: 依赖审计（含 target/sample Kendall）。
      elasticity_matrices: 季次弹性矩阵字典。
      out_dir: 输出目录（默认 paths.FIG_DIR）。
      figures: 指定生成的图名列表（None=全部，缺数据自动跳过）。
    """
    _setup_style()
    out_dir = Path(out_dir) if out_dir else paths.FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = figures if figures is not None else list(FIGURE_NAMES)
    results: dict = {}

    # 图名 -> (构建函数, 数据是否就绪)
    builders = {
        "raw_q3_target_sample_kendall": (
            lambda: _fig_target_sample_kendall(dependency_audit),
            dependency_audit is not None and
            getattr(dependency_audit, "target_kendall", None) is not None),
        "raw_q3_marginal_ranges": (
            lambda: _fig_marginal_ranges(scenarios, data),
            scenarios is not None),
        "raw_q3_elasticity_network": (
            lambda: _fig_elasticity_network(elasticity_matrices, data),
            elasticity_matrices is not None or
            getattr(data, "season_crop_sets", None)),
        "process_q3_scenario_reduction": (
            lambda: _fig_scenario_reduction(scenarios, reduced, data),
            scenarios is not None and reduced is not None),
        "process_q3_risk_frontier": (
            lambda: _fig_risk_frontier(frontier_points, selected_plan),
            frontier_points is not None),
        "process_q3_ablation": (
            lambda: _fig_ablation(ablation_df),
            ablation_df is not None),
        "result_q3_area_heatmap": (
            lambda: _fig_area_heatmap(selected_plan, data),
            selected_plan is not None),
        "result_q3_paired_profit_difference": (
            lambda: _fig_paired_profit_difference(q3_profits, q2_profits),
            q3_profits is not None and q2_profits is not None),
        "result_q3_q2_comparison": (
            lambda: _fig_q2_comparison(q3_profits, q2_profits),
            q3_profits is not None and q2_profits is not None),
    }

    for name in targets:
        if name not in builders:
            continue
        builder, ready = builders[name]
        if not ready:
            continue                      # 缺数据，静默跳过
        try:
            fig = builder()
            results[name] = _save(fig, name, out_dir)
        except Exception as exc:          # 单图失败不影响其他图
            print(f"[plots] 跳过 {name}: {exc}")
    return results


# ------------------------------------------------------------------ #
# 自检入口：用合成数据验证 9 张图导出链路（不依赖附件）
# ------------------------------------------------------------------ #
def _self_test() -> None:
    """合成最小数据，验证 9 张图均能生成 SVG/PNG/灰度预览。"""
    from dataclasses import dataclass, field
    from .scenarios import Q3ScenarioSet
    from .scenario_reduction import ReducedScenarioSet

    @dataclass
    class _DAudit:
        target_kendall: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
            np.eye(4) * 0.5 + 0.5 * np.eye(4)))
        sample_kendall: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
            np.eye(4) * 0.48 + 0.52 * np.eye(4)))
        max_kendall_error: float = 0.02

    class _Data:
        years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
        plot_names = [f"P{k}" for k in range(54)]
        plot_type = ["平旱地"] * 54
        plot_area = [20.0] * 54
        crop_group = {i: ("grain" if i <= 16 else "vegetable" if i <= 37
                          else "fungi") for i in range(1, 42)}
        season_crop_sets = {1: set(range(1, 17)), 2: set(range(17, 42))}
        bar_x = {(0, 1, 2024, 1): 10.0, (1, 2, 2024, 1): 8.0}
        D = {(1, 1): 1000.0, (2, 1): 800.0}
        q = {(0, 1, 1): 100.0, (1, 2, 1): 90.0}
        c = {(0, 1, 1): 50.0, (1, 2, 1): 40.0}
        p = {(1, 1): 2.0, (2, 1): 3.0}

    data = _Data()
    n = 200
    scen = Q3ScenarioSet(
        weights=np.full(n, 1 / n),
        demand={(1, 2024, 1): np.full(n, 1000.0) * (1 + np.random.default_rng(1).uniform(-0.05, 0.05, n))},
        demand_base={(1, 2024, 1): np.full(n, 1000.0)},
        yield_={(0, 1, 2024, 1): np.full(n, 100.0) * (1 + np.random.default_rng(2).uniform(-0.1, 0.1, n))},
        cost={(0, 1, 2024, 1): np.full(n, 50.0) * (1 + np.random.default_rng(3).uniform(0.04, 0.06, n))},
        price={(1, 2024, 1): np.full(n, 2.0) * (1 + np.random.default_rng(4).uniform(-0.01, 0.01, n))},
        trend_price={(1, 2024, 1): np.full(n, 2.0)}, n=n,
    )
    red = ReducedScenarioSet(
        indices=np.arange(20), weights=np.full(20, 1 / 20),
        demand={k: v[:20] for k, v in scen.demand.items()},
        yield_={k: v[:20] for k, v in scen.yield_.items()},
        cost={k: v[:20] for k, v in scen.cost.items()},
        price={k: v[:20] for k, v in scen.price.items()},
        k=20, n_original=n,
        proxy_profits=np.random.default_rng(5).normal(1e5, 1e4, 20),
    )
    from .elasticity import build_elasticity_matrix
    emat = build_elasticity_matrix(data, scale=1.0)
    rng = np.random.default_rng(7)
    out = generate_figures(
        data, scenarios=scen, reduced=red,
        frontier_points=[{"lambda": l, "expected_profit": 1e6 - 2e5 * l,
                          "lower_tail_cvar": 5e5 + 3e5 * l, "eligible": True}
                         for l in np.arange(0, 1.01, 0.1)],
        selected_plan={"x": {(j, (j % 5) + 1, 2024 + (j % 7), 1): 5.0
                             for j in range(54)}},
        q3_profits=rng.normal(1.2e6, 1e5, 500),
        q2_profits=rng.normal(1.15e6, 1.2e5, 500),
        ablation_df=pd.DataFrame({
            "group": ["Q2基线", "无相关", "无弹性", "完整Q3"],
            "期望利润": [1.15e6, 1.18e6, 1.16e6, 1.21e6],
            "CVaR": [5.2e5, 5.0e5, 5.1e5, 5.6e5],
        }),
        dependency_audit=_DAudit(),
        elasticity_matrices=emat,
        out_dir=paths.Q3_ROOT / "outputs" / "q3" / "figures" / "_selftest",
    )
    print(f"[self-test] 生成 {len(out)} 张图:")
    for name, p in out.items():
        print(f"  {name}: svg={p['svg'].name} png={p['png'].name} qa={p['qa'].name}")


if __name__ == "__main__":
    _self_test()
