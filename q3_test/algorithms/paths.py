# -*- coding: utf-8 -*-
"""Q3 路径配置 — 全部使用相对路径，保证工程可移植性。

路径推导逻辑：
  本文件位于 q3_test/algorithms/paths.py
  parent = q3_test/algorithms/
  parent.parent = q3_test/         (Q3_ROOT)
  parent.parent.parent = practice_1/ (PROJECT_ROOT)
"""
from __future__ import annotations
from pathlib import Path

# algorithms/ -> q3_test/
Q3_ROOT = Path(__file__).resolve().parent.parent
# q3_test/ -> practice_1/
PROJECT_ROOT = Q3_ROOT.parent

# ---- 只读输入（不修改）----
C_DIR = PROJECT_ROOT / "doc" / "C题"
F1_PATH = C_DIR / "附件1.xlsx"          # 54 地块 + 41 作物
F2_PATH = C_DIR / "附件2.xlsx"          # 2023 种植 + 亩产/成本/价格
TEMPLATE2_PATH = C_DIR / "附件3" / "result2.xlsx"   # Q2/Q3输出模板

# Q2 基线输出（只读，用于配对比较）
Q2_OUT_DIR = PROJECT_ROOT / "q2_test" / "outputs" / "q2"
Q2_SELECTED_PLAN = Q2_OUT_DIR / "selected_plan.csv"
Q2_AUDIT = Q2_OUT_DIR / "audit_q2.csv"
Q2_REPRO = Q2_OUT_DIR / "repro_q2.json"
Q2_RISK_FRONTIER = Q2_OUT_DIR / "risk_frontier.csv"

# 建模依据
ANALYSIS_REPORT = PROJECT_ROOT / "题目分析报告.md"
TERMS_TABLE = PROJECT_ROOT / "术语表格.md"

# SHA-256 校验（AGENT.md section 4）
EXPECTED_SHA = {
    F1_PATH: "5E98BF5E1B247624397E57E74759DA293BD72005DAA0E2DF3710A0DAC0E9EF6A",
    F2_PATH: "869081A3AB47D3BF8D0955106B622AAF0FD2C068FADA7948DA69B20EBF1D00CE",
    TEMPLATE2_PATH: "6A1BA9FC28D14D0A4A795E5F0B7261FB6E32165517AFEE62BCD1931ABA5BEE8A",
    ANALYSIS_REPORT: "C5DAE0CFEB81344F49E2D2D6ED704A7368C56AFD2D1A1B343609EC99499AFB17",
    TERMS_TABLE: "ED61B2FC173DA28A0FF0010749454D5027C5F18099EF235DA412BFE6B5ABE61B",
    Q2_SELECTED_PLAN: "ACAE51363E165A007A3EBAE76DE1470C226673003552AEEFEEFE4D37A1FFC491",
    Q2_AUDIT: "699571C4DB10987E68E2D1220C65820CA775CEBCF9F7999291B2BD03B70DC251",
    Q2_REPRO: "E5854369C453922C1F4306012A563C04B3F1962EBEBF0083B16E12038260FF7A",
}

# ---- 输出目录 ----
OUT_DIR = Q3_ROOT / "outputs"
Q3_OUT_DIR = OUT_DIR / "q3"
FIG_DIR = Q3_OUT_DIR / "figures"
LOG_DIR = Q3_OUT_DIR / "logs"
CKPT_DIR = Q3_OUT_DIR / "checkpoints"

# 输出文件路径
RESULT3_CANDIDATE = Q3_OUT_DIR / "result3_candidate.xlsx"
SELECTED_PLAN_Q3 = Q3_OUT_DIR / "selected_plan_q3.csv"
DEPENDENCY_TARGET_CSV = Q3_OUT_DIR / "dependency_target.csv"
DEPENDENCY_SAMPLE_CSV = Q3_OUT_DIR / "dependency_sample.csv"
ELASTICITY_MATRIX_CSV = Q3_OUT_DIR / "elasticity_matrix.csv"
SCENARIO_SUMMARY_Q3 = Q3_OUT_DIR / "scenario_summary_q3.csv"
RISK_FRONTIER_Q3 = Q3_OUT_DIR / "risk_frontier_q3.csv"
ABLATION_Q3 = Q3_OUT_DIR / "ablation_q3.csv"
PAIRED_PROFITS_Q2_Q3 = Q3_OUT_DIR / "paired_profits_q2_q3.csv"
OUT_OF_SAMPLE_METRICS_Q3 = Q3_OUT_DIR / "out_of_sample_metrics_q3.csv"
AUDIT_Q3 = Q3_OUT_DIR / "audit_q3.csv"
REPRO_Q3 = Q3_OUT_DIR / "repro_q3.json"

# 文字报告
DOC_DIR = Q3_ROOT / "doc"


def ensure_dirs() -> None:
    """创建所有输出目录。"""
    for d in (OUT_DIR, Q3_OUT_DIR, FIG_DIR, LOG_DIR, CKPT_DIR, DOC_DIR):
        d.mkdir(parents=True, exist_ok=True)
