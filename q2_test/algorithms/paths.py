# -*- coding: utf-8 -*-
"""Q2 路径配置 — 全部使用相对路径，保证工程可移植性。

路径推导逻辑：
  本文件位于 q2_test/algorithms/paths.py
  parent = q2_test/algorithms/
  parent.parent = q2_test/         (Q2_ROOT)
  parent.parent.parent = practice_1/ (PROJECT_ROOT)
"""
from __future__ import annotations
from pathlib import Path

# algorithms/ -> q2_test/
Q2_ROOT = Path(__file__).resolve().parent.parent
# q2_test/ -> practice_1/
PROJECT_ROOT = Q2_ROOT.parent

# ---- 只读输入（不修改）----
# 题目附件集中存放在 doc/C题/ 下
C_DIR = PROJECT_ROOT / "doc" / "C题"
F1_PATH = C_DIR / "附件1.xlsx"          # 54 地块 + 41 作物
F2_PATH = C_DIR / "附件2.xlsx"          # 2023 种植 + 亩产/成本/价格
TEMPLATE2_PATH = C_DIR / "附件3" / "result2.xlsx"   # Q2 输出模板

# Q1 结果（情景缩减的代理利润和基线比较用）
Q1_OUT_DIR = PROJECT_ROOT / "q1_test" / "outputs" / "q1"
Q1_RESULT1_1 = Q1_OUT_DIR / "result1_1.xlsx"   # 情形1 基线方案
Q1_RESULT1_2 = Q1_OUT_DIR / "result1_2.xlsx"   # 情形2 基线方案

# 建模依据
ANALYSIS_REPORT = PROJECT_ROOT / "题目分析报告.md"
TERMS_TABLE = PROJECT_ROOT / "术语表格.md"

# SHA-256 校验（AGENT.md section 2）
EXPECTED_SHA = {
    F1_PATH: "5E98BF5E1B247624397E57E74759DA293BD72005DAA0E2DF3710A0DAC0E9EF6A",
    F2_PATH: "869081A3AB47D3BF8D0955106B622AAF0FD2C068FADA7948DA69B20EBF1D00CE",
    TEMPLATE2_PATH: "6A1BA9FC28D14D0A4A795E5F0B7261FB6E32165517AFEE62BCD1931ABA5BEE8A",
}

# ---- 输出目录 ----
OUT_DIR = Q2_ROOT / "outputs"
Q2_OUT_DIR = OUT_DIR / "q2"                        # Q2 输出根
FIG_DIR = Q2_OUT_DIR / "figures"                    # 图表 (svg+png)
LOG_DIR = Q2_OUT_DIR / "logs"                       # 求解日志
RESULT2_PATH = Q2_OUT_DIR / "result2.xlsx"           # 唯一输出工作簿
SCENARIO_SUMMARY = Q2_OUT_DIR / "scenario_summary.csv"        # 情景摘要
RISK_FRONTIER_CSV = Q2_OUT_DIR / "risk_frontier.csv"          # 风险前沿
OUT_OF_SAMPLE_PROFITS = Q2_OUT_DIR / "out_of_sample_profits.csv"  # 样本外利润
OUT_OF_SAMPLE_METRICS = Q2_OUT_DIR / "out_of_sample_metrics.csv"  # 样本外指标
AUDIT_PATH = Q2_OUT_DIR / "audit_q2.csv"            # 约束审计
REPRO_PATH = Q2_OUT_DIR / "repro_q2.json"           # 复现清单
SELECTED_PLAN_CSV = Q2_OUT_DIR / "selected_plan.csv" # 可读的最终种植方案
SCENARIO_CACHE = Q2_OUT_DIR / "scenarios_raw.parquet"  # 原始情景缓存

# 文字报告
DOC_DIR = Q2_ROOT / "doc"


def ensure_dirs() -> None:
    """创建所有输出目录（如果不存在）。"""
    for d in (OUT_DIR, Q2_OUT_DIR, FIG_DIR, LOG_DIR, DOC_DIR):
        d.mkdir(parents=True, exist_ok=True)
