# -*- coding: utf-8 -*-
"""Figure generation for Q1 (AGENT.md section 11).

Produces >=9 logical figures (raw / process / result), each as SVG + PNG
(>=300 DPI). Uses a non-interactive Agg backend so the module is safe to
import from a CLI runner. Also exposes ``yearly_economics`` which the runner
reuses for the yearly stats table.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from .preprocess import ModelData

# Chinese font fallback (SimHei / Microsoft YaHei), then DejaVu Sans.
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

PLOT_TYPES = ["平旱地", "梯田", "山坡地", "水浇地", "普通大棚", "智慧大棚"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _first_plot_of_type(data: ModelData) -> dict:
    out = {}
    for j, t in enumerate(data.plot_type):
        if t not in out:
            out[t] = j
    return out


def save_fig(fig, name: str, output_dir, dpi: int = 300) -> list:
    """Save figure as .png (dpi) and .svg, close, return [png, svg] paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = output_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def yearly_economics(sol: dict, data: ModelData) -> pd.DataFrame:
    """Per-year revenue / cost / profit table from a solution dict.

    Columns: year, scenario, normal_revenue, half_revenue, cost, profit,
    total_Q, total_u, surplus.  Independent of the solver objective so it
    works for both primary and lexicographic solutions.
    """
    years = data.years
    scenario = sol.get("scenario", 1)
    u_map, Q_map = {}, {}
    if sol.get("u") is not None and len(sol["u"]):
        for _, r in sol["u"].iterrows():
            u_map[(int(r["crop_code"]), int(r["year"]), int(r["season"]))] = float(r["u"])
    if sol.get("Q") is not None and len(sol["Q"]):
        for _, r in sol["Q"].iterrows():
            Q_map[(int(r["crop_code"]), int(r["year"]), int(r["season"]))] = float(r["Q"])
    x_rows = []
    if sol.get("x") is not None and len(sol["x"]):
        for _, r in sol["x"].iterrows():
            x_rows.append(r)

    out = []
    for t in years:
        normal_rev = 0.0
        total_u = 0.0
        for (i, ts, s), uv in u_map.items():
            if ts != t:
                continue
            p = data.p.get((i, s), 0.0)
            normal_rev += p * uv
            total_u += uv
        # half-price surplus must be summed over Q (not u) so that crops with
        # Q>0 but u=0 (all surplus sold at half price) are still counted.
        half_rev = 0.0
        total_Q = 0.0
        for (i, ts, s), qv in Q_map.items():
            if ts != t:
                continue
            total_Q += qv
            if scenario == 2:
                p = data.p.get((i, s), 0.0)
                uv = u_map.get((i, ts, s), 0.0)
                half_rev += 0.5 * p * (qv - uv)
        cost = 0.0
        for r in x_rows:
            if int(r["year"]) != t:
                continue
            cost += data.c[(int(r["plot_idx"]), int(r["crop_code"]),
                            int(r["season"]))] * float(r["area"])
        out.append({
            "year": t, "scenario": scenario,
            "normal_revenue": normal_rev, "half_revenue": half_rev,
            "cost": cost, "profit": normal_rev + half_rev - cost,
            "total_Q": total_Q, "total_u": total_u,
            "surplus": total_Q - total_u,
        })
    return pd.DataFrame(out, columns=[
        "year", "scenario", "normal_revenue", "half_revenue",
        "cost", "profit", "total_Q", "total_u", "surplus"])


# ---------------------------------------------------------------------------
# raw-data figures (>=3)
# ---------------------------------------------------------------------------

def _fig_plot_area_by_type(data, output_dir, dpi):
    counts = {t: 0 for t in PLOT_TYPES}
    areas = {t: 0.0 for t in PLOT_TYPES}
    for j, t in enumerate(data.plot_type):
        counts[t] = counts.get(t, 0) + 1
        areas[t] = areas.get(t, 0.0) + data.A[j]
    types = [t for t in PLOT_TYPES if counts[t] > 0]
    cvals = [counts[t] for t in types]
    avals = [areas[t] for t in types]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(types))
    ax1.bar(x, cvals, color="#4C78A8", alpha=0.85, label="地块数量")
    ax1.set_ylabel("地块数量", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(x, avals, "-o", color="#E45756", label="总面积(亩)")
    ax2.set_ylabel("总面积(亩)", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    for xi, av in zip(x, avals):
        ax2.annotate(f"{av:.0f}", (xi, av), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#E45756")
    ax1.set_title("各土地类型的地块数量与总面积")
    fig.tight_layout()
    return save_fig(fig, "raw_q1_plot_area_by_type", output_dir, dpi)


def _fig_crop_yield_proxy(data, output_dir, dpi):
    recs = []
    for (i, s), d in data.D.items():
        if d > 0:
            recs.append({"crop": data.crop_names.get(i, str(i)),
                         "season": s, "D": d})
    df = pd.DataFrame(recs).sort_values("D", ascending=False) if recs else \
        pd.DataFrame(columns=["crop", "season", "D"])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    if len(df):
        x = np.arange(len(df))
        colors = ["#54A24B" if r["season"] == 1 else "#E45756"
                  for _, r in df.iterrows()]
        ax.bar(x, df["D"].values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(df["crop"].values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("2023 预期销售量代理 D[i,s] (斤)")
    ax.set_title("2023 年各作物预期销售量代理值（亩产量×面积汇总）")
    ax.legend(handles=[Patch(color="#54A24B", label="第一季"),
                       Patch(color="#E45756", label="第二季")],
              loc="upper right")
    fig.tight_layout()
    return save_fig(fig, "raw_q1_crop_yield_proxy_2023", output_dir, dpi)


def _fig_profit_heatmap(data, output_dir, dpi):
    first = _first_plot_of_type(data)
    types = [t for t in PLOT_TYPES if t in first]
    codes = data.crop_codes
    M = np.full((len(types), len(codes)), np.nan)
    for ti, t in enumerate(types):
        j = first[t]
        for ci, i in enumerate(codes):
            best = np.nan
            for s in (1, 2):
                if data.suit.get((j, i, s)) == 1:
                    p = data.p.get((i, s), 0.0)
                    q = data.q.get((j, i, s), 0.0)
                    c = data.c.get((j, i, s), 0.0)
                    prof = p * q - c
                    if np.isnan(best) or prof > best:
                        best = prof
            if not np.isnan(best):
                M[ti, ci] = best
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels(types)
    ax.set_xticks(np.arange(len(codes)))
    ax.set_xticklabels([data.crop_names.get(i, str(i)) for i in codes],
                       rotation=45, ha="right", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("亩均利润 (元/亩)")
    ax.set_title("各土地类型—作物的亩均利润热力图（p_mid·q − c）")
    fig.tight_layout()
    return save_fig(fig, "raw_q1_profit_heatmap", output_dir, dpi)


# ---------------------------------------------------------------------------
# process figures (>=3)
# ---------------------------------------------------------------------------

def _fig_flowchart(data, output_dir, dpi):
    steps = ["读取 Excel", "数据清洗", "建立 MILP 模型", "求解 (HiGHS)",
             "约束校验", "导出结果"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    bw = 0.9
    for k, s in enumerate(steps):
        x0 = k + (1 - bw) / 2
        box = FancyBboxPatch((x0, 0.3), bw, 0.4,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#333",
                             facecolor="#9EC5FE")
        ax.add_patch(box)
        ax.text(k + 0.5, 0.5, s, ha="center", va="center", fontsize=10)
        if k < len(steps) - 1:
            ax.annotate("", xy=(k + 1 + (1 - bw) / 2 - 0.03, 0.5),
                         xytext=(k + 0.5 + bw / 2, 0.5),
                         arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
    ax.set_title("问题1 数据—建模—求解—校验流程图")
    fig.tight_layout()
    return save_fig(fig, "process_q1_flowchart", output_dir, dpi)


def _fig_bound_convergence(data, output_dir, dpi, bound_history):
    fig, ax = plt.subplots(figsize=(9, 5))
    if not bound_history:
        ax.text(0.5, 0.5,
                "未提供求解器上下界历史\n(scipy.optimize.milp 不直接暴露节点级边界；\n"
                "运行 --figures 时由主流程注入定时子求解结果)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        ax.set_title("求解器原始解与对偶界收敛（情形1定时子求解）")
        fig.tight_layout()
        return save_fig(fig, "process_q1_bound_convergence", output_dir, dpi)
    times = [h["time"] for h in bound_history]
    primal = [h["primal"] for h in bound_history]
    dual = [h.get("dual", float("nan")) for h in bound_history]
    gap = [h.get("gap", float("nan")) for h in bound_history]
    ax.plot(times, primal, "-o", color="#4C78A8", label="原始解 Z (incumbent)")
    ax.plot(times, dual, "-s", color="#E45756", label="对偶上界 (-dual_bound)")
    ax.set_xlabel("求解时间 (s)")
    ax.set_ylabel("利润 Z (元)")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(times, gap, "--^", color="#54A24B", label="MIP gap")
    ax2.set_ylabel("MIP gap", color="#54A24B")
    ax2.tick_params(axis="y", labelcolor="#54A24B")
    ax2.legend(loc="upper right")
    ax.set_title("求解器原始解与对偶界收敛（情形1定时子求解）")
    fig.tight_layout()
    return save_fig(fig, "process_q1_bound_convergence", output_dir, dpi)


def _fig_constraint_slack(data, output_dir, dpi, audit):
    if audit is None:
        audit = {}
    if hasattr(audit, "to_dict"):
        audit = audit.to_dict()
    metrics = {
        "面积守恒违反": abs(float(audit.get("max_area_conservation_violation", 0.0))),
        "非适种面积": abs(float(audit.get("max_unsuitable_area", 0.0))),
        "最小面积违反": abs(float(audit.get("max_min_area_violation", 0.0))),
        "连作违反数": float(audit.get("monoculture_violation_count", 0)),
        "豆类最小松弛": float(audit.get("legume_min_slack", 0.0)),
        "水浇地模式冲突": float(audit.get("irrigated_mode_conflict_count", 0)),
        "u超过D": abs(float(audit.get("max_u_exceeds_D", 0.0))),
        "u超过Q": abs(float(audit.get("max_u_exceeds_Q", 0.0))),
        "利润重算差": abs(float(audit.get("profit_recompute_diff", 0.0))),
        "Excel回读差": abs(float(audit.get("excel_roundtrip_diff", 0.0))),
    }
    names = list(metrics.keys())
    vals = np.array([metrics[n] for n in names], dtype=float)
    vals = np.where(vals < 1e-9, 1e-9, vals)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(names)), vals, color="#4C78A8")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("量级 (对数刻度)")
    ax.set_title("情形1约束审计指标量级（对数刻度，1e-9 为地板）")
    ax.axhline(1e-6, color="#E45756", linestyle="--", linewidth=1,
               label="容差 1e-6")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, "process_q1_constraint_slack", output_dir, dpi)


# ---------------------------------------------------------------------------
# result figures (>=3)
# ---------------------------------------------------------------------------

def _fig_area_heatmap(data, output_dir, dpi, sol):
    years = data.years
    codes = data.crop_codes
    M = np.zeros((len(years), len(codes)))
    if sol is not None and sol.get("x") is not None and len(sol["x"]):
        for _, r in sol["x"].iterrows():
            yi = years.index(int(r["year"]))
            ci = codes.index(int(r["crop_code"]))
            M[yi, ci] += float(r["area"])
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years)
    ax.set_xticks(np.arange(len(codes)))
    ax.set_xticklabels([data.crop_names.get(i, str(i)) for i in codes],
                       rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="种植面积 (亩)")
    ax.set_title("情形1：2024—2030 年各作物总种植面积热力图")
    fig.tight_layout()
    return save_fig(fig, "result_q1_area_heatmap", output_dir, dpi)


def _fig_profit_breakdown(data, output_dir, dpi, sol1, sol2):
    e1 = yearly_economics(sol1, data) if sol1 else None
    e2 = yearly_economics(sol2, data) if sol2 else None
    metrics = [("normal_revenue", "正常价收入"),
               ("half_revenue", "半价收入"),
               ("cost", "成本"),
               ("profit", "利润")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    years = data.years
    x = np.arange(len(years))
    w = 0.4
    for ax, (key, title) in zip(axes.flat, metrics):
        if e1 is not None:
            ax.bar(x - w / 2, e1[key].values, width=w, color="#4C78A8",
                   label="情形1")
        if e2 is not None:
            ax.bar(x + w / 2, e2[key].values, width=w, color="#F58518",
                   label="情形2")
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("两种销售情形的年度收入与成本对比", fontsize=13)
    fig.tight_layout()
    return save_fig(fig, "result_q1_profit_breakdown", output_dir, dpi)


def _fig_production_vs_sales(data, output_dir, dpi, sol1, sol2):
    e1 = yearly_economics(sol1, data) if sol1 else None
    e2 = yearly_economics(sol2, data) if sol2 else None
    years = data.years
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.38
    if e1 is not None:
        ax.bar(x - w / 2, e1["total_u"].values, width=w, color="#4C78A8",
               label="情形1 正常销量 u")
        ax.bar(x - w / 2, e1["surplus"].values, width=w,
               bottom=e1["total_u"].values, color="#9EC5FE",
               label="情形1 浪费 (Q−u)")
    if e2 is not None:
        ax.bar(x + w / 2, e2["total_u"].values, width=w, color="#F58518",
               label="情形2 正常销量 u")
        ax.bar(x + w / 2, e2["surplus"].values, width=w,
               bottom=e2["total_u"].values, color="#FFB4A2",
               label="情形2 半价销量 (Q−u)")
    if e1 is not None:
        ax.plot(x, e1["total_Q"].values, "-o", color="#333",
                label="情形1 总产量 Q")
    if e2 is not None:
        ax.plot(x, e2["total_Q"].values, "--s", color="#666",
                label="情形2 总产量 Q")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("产量 / 销量 (斤)")
    ax.set_title("年度总产量 Q 的构成：正常销量与超额产量")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "result_q1_production_vs_sales", output_dir, dpi)


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

def generate_figures(data, sol1=None, sol2=None, output_dir=None,
                     audit1=None, audit2=None, bound_history=None,
                     dpi=300, seed=2024) -> list:
    """Generate all >=9 figures (raw/process/result) as SVG+PNG.

    Returns list of saved Path objects.  Each figure is wrapped so a single
    failure does not abort the whole batch.
    """
    if output_dir is None:
        from .paths import FIG_DIR
        output_dir = FIG_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_out = []

    def _run(builder):
        try:
            paths_out.extend(builder())
        except Exception as e:  # noqa: BLE001
            print(f"[plots] figure failed: {e}")

    _run(lambda: _fig_plot_area_by_type(data, output_dir, dpi))
    _run(lambda: _fig_crop_yield_proxy(data, output_dir, dpi))
    _run(lambda: _fig_profit_heatmap(data, output_dir, dpi))
    _run(lambda: _fig_flowchart(data, output_dir, dpi))
    _run(lambda: _fig_bound_convergence(data, output_dir, dpi, bound_history))
    _run(lambda: _fig_constraint_slack(data, output_dir, dpi, audit1))
    _run(lambda: _fig_area_heatmap(data, output_dir, dpi, sol1))
    _run(lambda: _fig_profit_breakdown(data, output_dir, dpi, sol1, sol2))
    _run(lambda: _fig_production_vs_sales(data, output_dir, dpi, sol1, sol2))
    return paths_out
