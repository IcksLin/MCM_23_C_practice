# -*- coding: utf-8 -*-
"""Q2 visualization — 9 required figures (AGENT.md section 17).

Raw data (3):
  1. raw_q2_uncertainty_ranges — annual change ranges per parameter
  2. raw_q2_lhs_marginal — LHS sample marginal distributions
  3. raw_q2_scenario_comparison — raw vs reduced scenario stats

Process (3):
  4. process_q2_flowchart — pipeline flowchart
  5. process_q2_risk_frontier — risk frontier with knee point
  6. process_q2_convergence — scenario count or bound convergence

Result (3):
  7. result_q2_area_heatmap — final annual area heatmap
  8. result_q2_profit_distribution — OOS profit distribution (3 baselines)
  9. result_q2_profit_comparison — expected/tail/stress profit comparison
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from . import paths


def generate_figures(data, scenarios=None, reduced=None,
                     frontier=None, evaluation=None,
                     plan=None, oos_profits=None,
                     output_dir=None, dpi=300, seed=2024) -> list:
    """Generate all Q2 figures. Returns list of saved file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    output_dir = Path(output_dir or paths.FIG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # ---- Raw 1: uncertainty ranges ----
    if scenarios:
        saved.append(_fig_uncertainty_ranges(scenarios, data, output_dir, dpi))

    # ---- Raw 2: LHS marginal ----
    if scenarios:
        saved.append(_fig_lhs_marginal(scenarios, data, output_dir, dpi))

    # ---- Raw 3: scenario comparison ----
    if scenarios and reduced:
        saved.append(_fig_scenario_comparison(scenarios, reduced, output_dir, dpi))

    # ---- Process 1: flowchart ----
    saved.append(_fig_flowchart(output_dir, dpi))

    # ---- Process 2: risk frontier ----
    if frontier is not None:
        saved.append(_fig_risk_frontier(frontier, output_dir, dpi))

    # ---- Process 3: convergence ----
    saved.append(_fig_convergence(frontier, output_dir, dpi))

    # ---- Result 1: area heatmap ----
    if plan and data:
        saved.append(_fig_area_heatmap(plan, data, output_dir, dpi))

    # ---- Result 2: profit distribution ----
    if oos_profits is not None:
        saved.append(_fig_profit_distribution(oos_profits, output_dir, dpi))

    # ---- Result 3: profit comparison ----
    if evaluation is not None:
        saved.append(_fig_profit_comparison(evaluation, output_dir, dpi))

    return saved


def _save(fig, name, output_dir, dpi):
    skill_scripts = (Path.home() / ".codex" / "skills" / "math-modeling"
                     / "tools" / "figure" / "scripts")
    if not (skill_scripts / "export_figure.py").is_file():
        raise RuntimeError(f"科研可视化导出器不存在: {skill_scripts}")
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from export_figure import export_figure
    export_figure(fig, basename=str(output_dir / name),
                  formats=["svg", "png"], dpi=dpi,
                  size_inches=tuple(fig.get_size_inches()),
                  grayscale_preview=True, tight=False)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return output_dir / f"{name}.png"


def _fig_uncertainty_ranges(scenarios, data, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    years = data.years
    # demand range per year
    dem_lo, dem_hi, dem_mean = [], [], []
    for t in years:
        # collect all demand values for year t
        all_d = []
        for key, arr in scenarios.demand.items():
            # key is (i, t, s)
            if len(key) == 3 and key[1] == t:
                all_d.extend(arr.tolist())
        if all_d:
            dem_lo.append(np.percentile(all_d, 10))
            dem_hi.append(np.percentile(all_d, 90))
            dem_mean.append(np.mean(all_d))
        else:
            dem_lo.append(0); dem_hi.append(0); dem_mean.append(0)
    ax.fill_between(range(len(years)), dem_lo, dem_hi, alpha=0.3, label="10%-90%")
    ax.plot(range(len(years)), dem_mean, "o-", label="均值")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_xlabel("年份")
    ax.set_ylabel("预期销量 (斤)")
    ax.set_title("各年预期销量变化区间")
    ax.legend()
    return _save(fig, "raw_q2_uncertainty_ranges", output_dir, dpi)


def _fig_lhs_marginal(scenarios, data, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # demand shock
    shocks = []
    for key, arr in scenarios.demand.items():
        i, t, s = key
        base = data.D.get((i, s), 0)
        if base > 0:
            shocks.extend(((arr - base) / base).tolist())
    axes[0].hist(shocks, bins=50, density=True, alpha=0.7, color="steelblue")
    axes[0].set_title("销量冲击分布")
    axes[0].set_xlabel("(D-D₀)/D₀")
    # yield shock
    shocks = []
    for key, arr in scenarios.yield_.items():
        j, i, t, s = key
        base = data.q.get((j, i, s), 0)
        if base > 0:
            shocks.extend(((arr - base) / base).tolist())
    axes[1].hist(shocks, bins=50, density=True, alpha=0.7, color="seagreen")
    axes[1].set_title("亩产冲击分布")
    axes[1].set_xlabel("(q-q₀)/q₀")
    # price shock
    shocks = []
    for key, arr in scenarios.price.items():
        i, t, s = key
        base = data.p.get((i, s), 0)
        if base > 0:
            shocks.extend(((arr - base) / base).tolist())
    axes[2].hist(shocks, bins=50, density=True, alpha=0.7, color="coral")
    axes[2].set_title("价格变化分布")
    axes[2].set_xlabel("(p-p₀)/p₀")
    fig.tight_layout()
    return _save(fig, "raw_q2_lhs_marginal", output_dir, dpi)


def _fig_scenario_comparison(scenarios, reduced, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    # Compare an interpretable aggregate demand index before/after reduction.
    raw_matrix = np.vstack(list(scenarios.demand.values()))
    red_matrix = np.vstack(list(reduced.demand.values()))
    raw_index = raw_matrix.mean(axis=0)
    red_index = red_matrix.mean(axis=0)
    ax.hist(raw_index, bins=min(30, max(8, scenarios.n // 10)), density=True,
            alpha=0.45, color="#0072B2", label=f"原始 N={scenarios.n}")
    ax.scatter(red_index, np.zeros_like(red_index), s=18 + 180 * reduced.weights,
               marker="|", color="#D55E00", label=f"代表 K={reduced.k}")
    ax.set_xlabel("跨作物平均预期销量（斤）")
    ax.set_ylabel("密度")
    ax.set_title("情景缩减保留销量分布")
    ax.legend(frameon=False)
    return _save(fig, "raw_q2_scenario_comparison", output_dir, dpi)


def _fig_flowchart(output_dir, dpi):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    steps = ["LHS生成", "PAM缩减", "随机MILP", "三级字典序", "样本外验证"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, (s, c) in enumerate(zip(steps, colors)):
        rect = mpatches.FancyBboxPatch((i*2.4+0.1, 1.5), 2.0, 1.0,
                                       boxstyle="round,pad=0.1",
                                       facecolor=c, edgecolor="black", alpha=0.8)
        ax.add_patch(rect)
        ax.text(i*2.4+1.1, 2.0, s, ha="center", va="center", fontsize=11, color="white")
        if i < len(steps)-1:
            ax.annotate("", xy=(i*2.4+2.3, 2.0), xytext=(i*2.4+2.1, 2.0),
                        arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Q2 求解流程", fontsize=13)
    return _save(fig, "process_q2_flowchart", output_dir, dpi)


def _fig_risk_frontier(frontier, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    if "expected_profit" in frontier.columns:
        valid = frontier.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["lower_tail_cvar", "expected_profit"])
        if "eligible" in valid:
            valid = valid[valid["eligible"]]
        ax.scatter(valid["lower_tail_cvar"], valid["expected_profit"],
                   c=valid["lambda"], cmap="viridis", s=38)
        for _, row in valid.iterrows():
            ax.annotate(f"λ={row['lambda']:.1f}",
                        (row["lower_tail_cvar"], row["expected_profit"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("下尾利润 CVaR (元)")
        ax.set_ylabel("期望利润 (元)")
        ax.set_title("风险前沿")
    return _save(fig, "process_q2_risk_frontier", output_dir, dpi)


def _fig_convergence(frontier, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("各风险权重的求解最优性间隙")
    ax.set_xlabel("风险权重 λ")
    ax.set_ylabel("MIP gap")
    if frontier is not None and "mip_gap" in frontier:
        valid = frontier.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["lambda", "mip_gap"])
        ax.scatter(valid["lambda"], valid["mip_gap"], color="#0072B2", s=30)
        ax.axhline(0.001, color="#D55E00", linestyle="--", linewidth=1,
                   label="正式阈值 0.001")
        ax.legend(frameon=False)
    return _save(fig, "process_q2_convergence", output_dir, dpi)


def _fig_area_heatmap(plan, data, output_dir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np
    x = plan["x"] if isinstance(plan, dict) and "x" in plan else plan
    # build matrix: years x plots
    years = data.years
    plots = data.plot_names
    mat = np.zeros((len(years), len(plots)))
    for (j, i, t, s), area in x.items():
        if t in years:
            ti = years.index(t)
            mat[ti, j] += area
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.pcolormesh(np.arange(mat.shape[1] + 1),
                       np.arange(mat.shape[0] + 1), mat,
                       cmap="viridis", shading="flat")
    ax.set_ylim(mat.shape[0], 0)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("地块")
    ax.set_ylabel("年份")
    ax.set_title("最终年度种植面积热力图")
    fig.colorbar(im, ax=ax, label="面积 (亩)")
    return _save(fig, "result_q2_area_heatmap", output_dir, dpi)


def _fig_profit_distribution(oos_profits, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    if isinstance(oos_profits, dict):
        for label, vals in oos_profits.items():
            ax.hist(vals, bins=50, alpha=0.5, label=label, density=True)
        ax.legend()
    else:
        ax.hist(oos_profits, bins=50, density=True, color="steelblue")
    ax.set_xlabel("利润 (元)")
    ax.set_ylabel("密度")
    ax.set_title("样本外利润分布")
    return _save(fig, "result_q2_profit_distribution", output_dir, dpi)


def _fig_profit_comparison(evaluation, output_dir, dpi):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    if isinstance(evaluation, dict):
        labels = list(evaluation.keys())
        metric_names = ["均值", "最差10%均值", "最小值"]
        values = []
        for label in labels:
            samples = np.asarray(evaluation[label], dtype=float)
            tail_n = max(1, int(np.ceil(0.1 * len(samples))))
            values.append([samples.mean(), np.sort(samples)[:tail_n].mean(),
                           samples.min()])
        values = np.asarray(values)
        xpos = np.arange(len(labels))
        width = 0.24
        colors = ["#0072B2", "#E69F00", "#009E73"]
        for idx, metric in enumerate(metric_names):
            ax.bar(xpos + (idx - 1) * width, values[:, idx], width,
                   label=metric, color=colors[idx])
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("利润 (元)")
        ax.set_title("基线与风险方案的样本外利润")
        ax.legend(frameon=False)
    return _save(fig, "result_q2_profit_comparison", output_dir, dpi)
