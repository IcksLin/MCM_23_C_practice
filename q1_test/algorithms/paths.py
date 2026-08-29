# -*- coding: utf-8 -*-
"""Path configuration resolved relative to this file's location.

All paths are derived from the package location so the project is portable
across machines without editing hardcoded global addresses.
"""
from __future__ import annotations
from pathlib import Path

# algorithms/ -> parent = q1_test root
Q1_ROOT = Path(__file__).resolve().parent.parent
# q1_test/ -> parent = practice_1 project root
PROJECT_ROOT = Q1_ROOT.parent

# ---- Read-only inputs (do not modify) ----
# 题目附件集中存放在 doc/C题/ 下
C_DIR = PROJECT_ROOT / "doc" / "C题"
F1_PATH = C_DIR / "附件1.xlsx"          # 54 plots & 41 crops
F2_PATH = C_DIR / "附件2.xlsx"          # 2023 planting & statistics
TEMPLATE1_PATH = C_DIR / "附件3" / "result1_1.xlsx"  # scenario-1 template
TEMPLATE2_PATH = C_DIR / "附件3" / "result1_2.xlsx"  # scenario-2 template

# Expected SHA-256 of frozen inputs (AGENT.md section 2)
EXPECTED_SHA = {
    F1_PATH: "5E98BF5E1B247624397E57E74759DA293BD72005DAA0E2DF3710A0DAC0E9EF6A",
    F2_PATH: "869081A3AB47D3BF8D0955106B622AAF0FD2C068FADA7948DA69B20EBF1D00CE",
    TEMPLATE1_PATH: "4F2484C0D70A5C4D047163F2EE6EF486949E813330466F46DEF4BD7D98AF06AF",
    TEMPLATE2_PATH: "6166D43F5A64BF9D1657E80D4AEE7F10F54BB1A5695B81A28A0AC5E657297649",
}

# ---- Output directories ----
OUT_DIR = Q1_ROOT / "outputs"
DATA_CLEAN_DIR = OUT_DIR / "data_cleaning"               # data-cleaning results
Q1_OUT_DIR = OUT_DIR / "q1"                              # q1 outputs
FIG_DIR = Q1_OUT_DIR / "figures"                         # figures (svg+png)
LOG_DIR = Q1_OUT_DIR / "logs"                            # solver logs
RESULT1_PATH = Q1_OUT_DIR / "result1_1.xlsx"             # scenario-1 workbook
RESULT2_PATH = Q1_OUT_DIR / "result1_2.xlsx"             # scenario-2 workbook
AUDIT_PATH = Q1_OUT_DIR / "audit.csv"                   # constraint audit
STATS_PATH = Q1_OUT_DIR / "yearly_stats.csv"            # yearly profit & prod
SENS_ETA_PATH = Q1_OUT_DIR / "sensitivity_eta.csv"      # eta sensitivity
SENS_DELTA_PATH = Q1_OUT_DIR / "sensitivity_delta.csv"  # delta sensitivity
SENS_DEMAND_PATH = Q1_OUT_DIR / "sensitivity_demand.csv"  # demand sensitivity
REPRO_PATH = Q1_OUT_DIR / "repro.json"                  # reproduction manifest
P1_PATH = Q1_OUT_DIR / "p1_test.xlsx"                   # P1 minimal chain test

# Text reports live under q1_test/doc/
DOC_DIR = Q1_ROOT / "doc"


def configure_output_dir(output_dir) -> None:
    """把本次运行全部产物重定向到Config指定目录。"""
    global OUT_DIR, DATA_CLEAN_DIR, Q1_OUT_DIR, FIG_DIR, LOG_DIR
    global RESULT1_PATH, RESULT2_PATH, AUDIT_PATH, STATS_PATH
    global SENS_ETA_PATH, SENS_DELTA_PATH, SENS_DEMAND_PATH, REPRO_PATH
    global P1_PATH, DOC_DIR
    root = Path(output_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    protected = (C_DIR / "附件3").resolve()
    if root == protected or protected in root.parents:
        raise ValueError("输出目录不得位于题目原始附件目录")
    OUT_DIR = Q1_OUT_DIR = root
    DATA_CLEAN_DIR = root / "data_cleaning"
    FIG_DIR = root / "figures"
    LOG_DIR = root / "logs"
    RESULT1_PATH = root / "result1_1.xlsx"
    RESULT2_PATH = root / "result1_2.xlsx"
    AUDIT_PATH = root / "audit.csv"
    STATS_PATH = root / "yearly_stats.csv"
    SENS_ETA_PATH = root / "sensitivity_eta.csv"
    SENS_DELTA_PATH = root / "sensitivity_delta.csv"
    SENS_DEMAND_PATH = root / "sensitivity_demand.csv"
    REPRO_PATH = root / "repro.json"
    P1_PATH = root / "p1.xlsx"
    DOC_DIR = root


def ensure_dirs() -> None:
    """Create all output directories if missing."""
    for d in (DATA_CLEAN_DIR, Q1_OUT_DIR, FIG_DIR, LOG_DIR, DOC_DIR):
        d.mkdir(parents=True, exist_ok=True)
