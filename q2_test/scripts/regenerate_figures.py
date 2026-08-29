# -*- coding: utf-8 -*-
"""Regenerate Q2 figures from the latest stored smoke/formal results."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pandas as pd

Q2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Q2_ROOT))

from algorithms import io_data, paths, preprocess
from algorithms.plots import generate_figures
from algorithms.scenario_reduction import reduce_scenarios
from algorithms.scenarios import generate_raw_scenarios


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = preprocess.preprocess(io_data.load_inputs())
    repro = json.loads(paths.REPRO_PATH.read_text(encoding="utf-8"))
    raw = generate_raw_scenarios(
        data, repro["raw_scenarios"], repro["seed"], repro["distribution"])
    baseline = {(j, i, year, season): area
                for (j, i, season), area in data.bar_x.items()
                for year in data.years}
    reduced = reduce_scenarios(
        data, raw, repro["reduced_scenarios"], baseline, repro["seed"])
    stored = pd.read_csv(paths.SELECTED_PLAN_CSV)
    plan = {"x": {(data.plot_idx[row.plot], int(row.crop_code),
                   int(row.year), int(row.season)): float(row.area)
                  for row in stored.itertuples()}}
    frontier = pd.read_csv(paths.RISK_FRONTIER_CSV)
    oos = pd.read_csv(paths.OUT_OF_SAMPLE_PROFITS)
    comparison = {column: oos[column].to_numpy() for column in oos.columns}
    generate_figures(data, raw, reduced, frontier, comparison, plan, comparison)
    report_file = paths.DOC_DIR / "Q2_建模实现报告.md"
    outputs = [p for p in paths.Q2_OUT_DIR.rglob("*")
               if p.is_file() and p != paths.REPRO_PATH]
    if report_file.exists():
        outputs.append(report_file)
    repro["output_sha256"] = {
        (str(p.relative_to(paths.Q2_ROOT)) if p.is_relative_to(paths.Q2_ROOT)
         else str(p)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in outputs
    }
    paths.REPRO_PATH.write_text(
        json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Q2 figures regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
