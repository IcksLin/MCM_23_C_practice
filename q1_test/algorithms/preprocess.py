# -*- coding: utf-8 -*-
"""Data cleaning and ModelData assembly.

Responsibilities (AGENT.md section 4):
  * strip whitespace, drop ``_x000d_`` artefacts
  * explicit numeric conversion with failure log
  * forward-fill merged-cell group fields (plot name done in io_data)
  * 智慧大棚第一季 inherits 普通大棚第一季 (yield / cost / price)
  * build suitability e[j][i][s], expected-sales proxy D_is,
    2023 historical state (bar_x, bar_y, r_2023), 重茬 adjacency pairs E,
    rolling 3-year legume windows
  * data assertions (54 plots, 41 crops, 87 plantings, 107 stats, 1213 mu)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math
import pandas as pd

from .io_data import RawData


# ---- crop groupings (codes 1..41) ----
LEGUME_CODES = list(range(1, 6)) + list(range(17, 20))     # 1-5, 17-19
GRAIN_CODES = list(range(1, 16))                            # 1-15 (excl. rice 16)
RICE_CODE = 16
VEG_CODES = list(range(17, 35))                              # 17-34
ROOT_CODES = list(range(35, 38))                            # 35-37 (水浇地 s2)
MUSHROOM_CODES = list(range(38, 42))                        # 38-41 (普通大棚 s2)

PLOT_TYPES_SINGLE = ("平旱地", "梯田", "山坡地")           # 单季粮食


def _clean_str(v):
    if v is None:
        return ""
    s = str(v).replace("_x000d_", "").strip()
    return s


def _to_num(v, log_rows, sheet, r, c):
    """Explicit numeric conversion; record failures."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except ValueError:
        log_rows.append({"sheet": sheet, "row": r, "col": c, "raw": str(v)})
        return float("nan")


def _parse_price(raw):
    """'2.50-4.00' -> (lo, hi, mid). Single value -> (v, v, v)."""
    s = raw.replace(" ", "")
    if "-" in s:
        lo, hi = s.split("-", 1)
        lo, hi = float(lo), float(hi)
    else:
        lo = hi = float(s)
    return lo, hi, (lo + hi) / 2.0


def _season_norm(s):
    """单季/第一季 -> 1, 第二季 -> 2."""
    s = _clean_str(s)
    if "二" in s:
        return 2
    return 1


@dataclass
class ModelData:
    # plots
    plot_names: list
    plot_type: list
    plot_area: list
    plot_idx: dict
    plot_seasons: list               # list of seasons per plot (e.g. [1] or [1,2])
    # crops
    crop_codes: list
    crop_names: dict
    legume_set: set
    # years
    years: list = field(default_factory=lambda: [2024, 2025, 2026, 2027, 2028, 2029, 2030])
    # parameters
    q: dict = field(default_factory=dict)     # (j, i, s) -> yield 斤/亩
    c: dict = field(default_factory=dict)     # (j, i, s) -> cost 元/亩
    p: dict = field(default_factory=dict)     # (i, s) -> price mid 元/斤
    p_lo: dict = field(default_factory=dict)
    p_hi: dict = field(default_factory=dict)
    A: dict = field(default_factory=dict)    # j -> area
    D: dict = field(default_factory=dict)    # (i, s) -> expected sales proxy 斤
    suit: dict = field(default_factory=dict) # (j, i, s) -> 0/1
    # history
    bar_x: dict = field(default_factory=dict)  # (j, i, s) -> 2023 area
    bar_y: dict = field(default_factory=dict)  # (j, i, s) -> 2023 activation
    r_2023: dict = field(default_factory=dict) # j -> 0/1 (水浇地 only)
    # constraints
    adj_pairs: list = field(default_factory=list)     # (j, i, (t_a,s_a), (t_b,s_b))
    legume_windows: list = field(default_factory=list)  # (j, [years...], hist_legume_area)
    # template
    tpl_crop_col: dict = field(default_factory=dict)  # crop_code -> col idx
    tpl_row_s1: dict = field(default_factory=dict)    # plot name -> row idx
    tpl_row_s2: dict = field(default_factory=dict)
    # audit
    clean_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    assertions: dict = field(default_factory=dict)
    # raw provenance
    f1_sha: str = ""
    f2_sha: str = ""
    template1_sha: str = ""


def _build_suitability(md: ModelData) -> None:
    """e[j][i][s] by plot type per AGENT.md section 3.6."""
    for j, ptype in enumerate(md.plot_type):
        if ptype in PLOT_TYPES_SINGLE:
            for i in GRAIN_CODES:
                md.suit[(j, i, 1)] = 1
        elif ptype == "水浇地":
            md.suit[(j, RICE_CODE, 1)] = 1
            for i in VEG_CODES:
                md.suit[(j, i, 1)] = 1
            for i in ROOT_CODES:
                md.suit[(j, i, 2)] = 1
        elif ptype == "普通大棚":
            for i in VEG_CODES:
                md.suit[(j, i, 1)] = 1
            for i in MUSHROOM_CODES:
                md.suit[(j, i, 2)] = 1
        elif ptype == "智慧大棚":
            for i in VEG_CODES:
                md.suit[(j, i, 1)] = 1
                md.suit[(j, i, 2)] = 1


def _build_params(raw: RawData, md: ModelData, log_rows: list) -> None:
    """Fill q, c, p from stats; inherit 智慧大棚 s1 from 普通大棚 s1."""
    # indexed stats: (land_type, crop_code, season) -> (yield, cost, price_raw)
    stat_map = {}
    for _, row in raw.stats_2023.iterrows():
        code = row["crop_code"]
        land = _clean_str(row["land_type"])
        s = _season_norm(row["season"])
        yld = _to_num(row["yield"], log_rows, "stats", row.get("seq", ""), 6)
        cost = _to_num(row["cost"], log_rows, "stats", row.get("seq", ""), 7)
        price_raw = _clean_str(row["price_raw"])
        stat_map[(land, code, s)] = (yld, cost, price_raw)

    # inherit 智慧大棚第一季 from 普通大棚第一季
    inherited = 0
    for key, val in list(stat_map.items()):
        land, code, s = key
        if land == "普通大棚" and s == 1:
            new_key = ("智慧大棚", code, 1)
            if new_key not in stat_map:
                stat_map[new_key] = val
                inherited += 1
    md.assertions["inherited_smart_s1"] = inherited

    # fill q, c per (j, i, s); p per (i, s)
    for j, ptype in enumerate(md.plot_type):
        for (jj, i, s), v in md.suit.items():
            if jj != j:
                continue
            key = (ptype, i, s)
            if key not in stat_map:
                raise RuntimeError(f"missing stat for {key}")
            yld, cost, price_raw = stat_map[key]
            md.q[(j, i, s)] = yld
            md.c[(j, i, s)] = cost
            lo, hi, mid = _parse_price(price_raw)
            # price per (i, s); assert consistency
            if (i, s) in md.p:
                if abs(md.p[(i, s)] - mid) > 1e-9:
                    raise RuntimeError(
                        f"price mismatch for (i={i},s={s}): {md.p[(i,s)]} vs {mid}")
            else:
                md.p[(i, s)] = mid
                md.p_lo[(i, s)] = lo
                md.p_hi[(i, s)] = hi
    # check no missing params after inheritance
    missing = [(j, i, s) for (j, i, s) in md.suit if (j, i, s) not in md.q]
    md.assertions["missing_param_combos"] = missing
    if missing:
        raise RuntimeError(f"missing q for combos: {missing[:5]} ...")


def _build_history(raw: RawData, md: ModelData, log_rows: list) -> None:
    """bar_x / bar_y for 2023; r_2023 for 水浇地."""
    for _, row in raw.planting_2023.iterrows():
        plot = _clean_str(row["plot"])
        if plot not in md.plot_idx:
            continue
        j = md.plot_idx[plot]
        code = row["crop_code"]
        if code is None:
            continue
        try:
            i = int(code)
        except (ValueError, TypeError):
            log_rows.append({"sheet": "planting", "row": _, "col": "code", "raw": str(code)})
            continue
        s = _season_norm(row["season"])
        area = _to_num(row["area"], log_rows, "planting", plot, 5)
        # 智慧大棚 s1 had no stat row originally; q comes via inheritance -> ok now
        md.bar_x[(j, i, s)] = md.bar_x.get((j, i, s), 0.0) + area
        md.bar_y[(j, i, s)] = 1
    # r_2023 for 水浇地: 1 if rice was planted (s1, code 16), else 0
    for j, ptype in enumerate(md.plot_type):
        if ptype == "水浇地":
            md.r_2023[j] = 1 if md.bar_y.get((j, RICE_CODE, 1), 0) == 1 else 0


def _build_demand(md: ModelData) -> None:
    """D[i,s] = sum_j q[j,i,s] * bar_x[j,i,s] (2023 yield proxy)."""
    for (j, i, s), area in md.bar_x.items():
        q = md.q.get((j, i, s), 0.0)
        md.D[(i, s)] = md.D.get((i, s), 0.0) + q * area


def _plot_slot_list(j: int, md: ModelData) -> list:
    """Ordered (t, s) slots 2023..2030 for plot j (2023 from actual history)."""
    ptype = md.plot_type[j]
    slots = []
    # 2023
    has_s1_2023 = any((j, i, 1) in md.bar_y for i in range(1, 42))
    has_s2_2023 = any((j, i, 2) in md.bar_y for i in range(1, 42))
    if ptype in PLOT_TYPES_SINGLE:
        if has_s1_2023:
            slots.append((2023, 1))
    else:
        if has_s1_2023:
            slots.append((2023, 1))
        if has_s2_2023:
            slots.append((2023, 2))
    # 2024..2030: all potential seasons for this plot type
    for t in md.years:
        for s in md.plot_seasons[j]:
            slots.append((t, s))
    return slots


def _build_adjacency(md: ModelData) -> None:
    """重茬 pairs E (AGENT.md 3.7): temporally consecutive slots, same crop
    forbidden. Boundary 2023<->2024 included via bar_y."""
    for j in range(len(md.plot_names)):
        slots = _plot_slot_list(j, md)
        for a in range(len(slots) - 1):
            ta, sa = slots[a]
            tb, sb = slots[a + 1]
            # crops suitable in BOTH slots
            for i in range(1, 42):
                if md.suit.get((j, i, sa)) and md.suit.get((j, i, sb)):
                    md.adj_pairs.append((j, i, (ta, sa), (tb, sb)))


def _build_legume_windows(md: ModelData) -> None:
    """Rolling 3-year legume coverage (AGENT.md 3.8)."""
    full = [2023] + md.years  # 2023..2030
    for j in range(len(md.plot_names)):
        for k in range(len(full) - 2):
            window = full[k:k + 3]
            hist = 0.0
            if 2023 in window:
                for i in LEGUME_CODES:
                    for s in (1, 2):
                        hist += md.bar_x.get((j, i, s), 0.0)
            md.legume_windows.append((j, window, hist))


def _build_template_mapping(raw: RawData, md: ModelData) -> None:
    # crop code -> column index in template (col 3..43 for codes 1..41)
    # template order matches crop_codes order 1..41
    for code in md.crop_codes:
        idx = code - 1                       # 0-based among crops
        md.tpl_crop_col[code] = 3 + idx       # excel column
    # plot name -> row (s1: rows 2..55, s2: rows 56..83)
    for r, name in enumerate(raw.template_plot_s1):
        if name:
            md.tpl_row_s1[name] = 2 + r
    for r, name in enumerate(raw.template_plot_s2):
        if name:
            md.tpl_row_s2[name] = 56 + r


def _assertions(raw: RawData, md: ModelData) -> None:
    a = md.assertions
    a["n_plots"] = len(md.plot_names)
    a["n_crops"] = len(md.crop_codes)
    a["n_planting_2023"] = len(raw.planting_2023)
    a["n_stats_2023"] = len(raw.stats_2023)
    a["total_area"] = sum(md.plot_area)
    # regression check on smart-greenhouse inheritance (AGENT.md section 4)
    inh_yield = 0.0
    inh_cost = 0.0
    for j, ptype in enumerate(md.plot_type):
        if ptype == "智慧大棚":
            for (jj, i, s), area in md.bar_x.items():
                if jj == j and s == 1:
                    inh_yield += md.q[(j, i, s)] * area
                    inh_cost += md.c[(j, i, s)] * area
    a["smart_s1_inherited_yield"] = inh_yield   # expect ~12270
    a["smart_s1_inherited_cost"] = inh_cost     # expect ~7080
    # hard assertions
    assert a["n_plots"] == 54, f"plots {a['n_plots']} != 54"
    assert a["n_crops"] == 41, f"crops {a['n_crops']} != 41"
    assert a["n_planting_2023"] == 87, f"planting {a['n_planting_2023']} != 87"
    assert a["n_stats_2023"] == 107, f"stats {a['n_stats_2023']} != 107"
    assert abs(a["total_area"] - 1213.0) < 1e-6, f"area {a['total_area']} != 1213"
    assert abs(inh_yield - 12270.0) < 1e-6, f"inh_yield {inh_yield} != 12270"
    assert abs(inh_cost - 7080.0) < 1e-6, f"inh_cost {inh_cost} != 7080"


def preprocess(raw: RawData) -> ModelData:
    plot_names = [_clean_str(n) for n in raw.plots["name"].tolist()]
    plot_type = [_clean_str(t) for t in raw.plots["type"].tolist()]
    plot_area = [float(a) for a in raw.plots["area"].tolist()]
    plot_seasons = [
        [1] if t in PLOT_TYPES_SINGLE else [1, 2] for t in plot_type
    ]
    md = ModelData(
        plot_names=plot_names,
        plot_type=plot_type,
        plot_area=plot_area,
        plot_idx={n: k for k, n in enumerate(plot_names)},
        plot_seasons=plot_seasons,
        crop_codes=list(range(1, 42)),
        crop_names={int(r["code"]): _clean_str(r["name"])
                    for _, r in raw.crops.iterrows() if r["code"] is not None},
        legume_set=set(LEGUME_CODES),
    )
    md.A = {j: a for j, a in enumerate(md.plot_area)}

    log_rows: list = []
    _build_suitability(md)
    _build_params(raw, md, log_rows)
    _build_history(raw, md, log_rows)
    _build_demand(md)
    _build_adjacency(md)
    _build_legume_windows(md)
    _build_template_mapping(raw, md)
    _assertions(raw, md)

    md.clean_log = pd.DataFrame(log_rows, columns=["sheet", "row", "col", "raw"])
    md.f1_sha = raw.f1_sha
    md.f2_sha = raw.f2_sha
    md.template1_sha = raw.template1_sha
    return md
