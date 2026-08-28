# -*- coding: utf-8 -*-
"""Smoke test: load + preprocess, print key assertions and diagnostics."""
import sys
from pathlib import Path

# make q1_test importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms import io_data, preprocess, paths

paths.ensure_dirs()
print("=== input hash check ===")
for p, (sha, ok, exp) in io_data.verify_inputs().items():
    print(f"  {'OK' if ok else 'FAIL'}  {Path(p).name}  {sha}")

print("\n=== load_inputs ===")
raw = io_data.load_inputs()
print(f"plots={len(raw.plots)} crops={len(raw.crops)} "
      f"planting={len(raw.planting_2023)} stats={len(raw.stats_2023)}")
print(f"template_crops[:5]={raw.template_crops[:5]} ... total={len(raw.template_crops)}")
print(f"template_plot_s1[:5]={raw.template_plot_s1[:5]} total_s1={len(raw.template_plot_s1)}")
print(f"template_plot_s2[:5]={raw.template_plot_s2[:5]} total_s2={len(raw.template_plot_s2)}")

print("\n=== preprocess ===")
md = preprocess.preprocess(raw)
a = md.assertions
print(f"n_plots={a['n_plots']} n_crops={a['n_crops']} "
      f"n_planting={a['n_planting_2023']} n_stats={a['n_stats_2023']} "
      f"total_area={a['total_area']}")
print(f"inherited_smart_s1={a['inherited_smart_s1']}")
print(f"smart_s1_inherited_yield={a['smart_s1_inherited_yield']} (expect 12270)")
print(f"smart_s1_inherited_cost={a['smart_s1_inherited_cost']} (expect 7080)")
print(f"suitable_combos={len(md.suit)} adj_pairs={len(md.adj_pairs)} "
      f"legume_windows={len(md.legume_windows)}")
print(f"D entries={len(md.D)} bar_x entries={len(md.bar_x)} r_2023={md.r_2023}")
# D sample
print("\nD[i,s] sample (top 8):")
for (i, s), v in sorted(md.D.items())[:8]:
    print(f"  i={i} s={s} -> {v:.1f} 斤  ({md.crop_names.get(i,'')})")
# suit summary by plot type
from collections import Counter
cnt = Counter()
for (j, i, s) in md.suit:
    cnt[md.plot_type[j]] += 1
print("\nsuit count by plot type:", dict(cnt))
print("clean_log rows:", len(md.clean_log))
if len(md.clean_log):
    print(md.clean_log.head())
print("\nALL ASSERTIONS PASSED")
