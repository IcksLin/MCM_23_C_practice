# 2024 年高教社杯全国大学生数学建模竞赛 C 题 — 农作物的种植策略

> 本工程是学习与研究用草稿，不直接作为参赛作品提交。正式使用前需由队伍人工核对、改写。

## 工程总览

本工程实现 2024 年国赛 C 题的建模与求解，按问题分为独立子工程：

| 子工程 | 路径 | 状态 | 说明 |
|---|---|---|---|
| 问题 1 | `q1_test/` | 代码已完成，求解验证中 | 两种滞销情形的确定性 MILP |
| 问题 2 | `q2_test/` | 代码已完成，求解验证中 | 情景随机 MILP + CVaR 风险权衡 |
| 问题 3 | — | 待规划 | 不确定参数相关性与鲁棒优化 |

## 目录结构

```text
practice_1/
├── doc/                          # 顶层文档
│   ├── C题/                       # 只读赛题附件
│   │   ├── C题.pdf
│   │   ├── 附件1.xlsx             # 54 地块 + 41 作物基本情况
│   │   ├── 附件2.xlsx             # 2023 种植 + 亩产/成本/价格
│   │   └── 附件3/                 # 输出模板
│   │       ├── result1_1.xlsx     # 问题1 情形1 模板
│   │       ├── result1_2.xlsx     # 问题1 情形2 模板
│   │       └── result2.xlsx        # 问题2 模板
│   └── 2024_C题_农作物的种植策略.md
├── q1_test/                      # 问题1 子工程
│   ├── AGENT.md                   # Q1 实现指导
│   ├── algorithms/                # 算法模块
│   ├── scripts/                   # 工具脚本
│   ├── outputs/
│   │   ├── data_cleaning/         # 数据清洗产物
│   │   └── q1/                    # Q1 输出产物
│   ├── doc/                       # Q1 文字报告
│   ├── README.md
│   ├── requirements.txt
│   └── 一键运行问题1.bat
├── q2_test/                      # 问题2 子工程
│   ├── AGENT.md                   # Q2 实现指导（权威合同）
│   ├── algorithms/                # 算法模块（已实现）
│   │   ├── paths.py               # 路径配置（相对路径）
│   │   ├── io_data.py             # 只读 Excel 读取
│   │   ├── preprocess.py          # 数据清洗 & ModelData
│   │   ├── scenarios.py           # LHS 情景生成
│   │   ├── scenario_reduction.py  # PAM k-medoids 缩减
│   │   ├── model.py               # 均值-CVaR 随机 MILP
│   │   ├── solve.py               # 三级字典序求解
│   │   ├── risk.py                # CVaR 复算 & 膝点选择
│   │   ├── validate.py            # 约束审计
│   │   ├── export_ooxml.py        # 安全 Excel 回填
│   │   └── plots.py               # 9 张图表
│   ├── scripts/                   # 工具脚本
│   │   └── run_q2.py              # Q2 主入口
│   ├── outputs/q2/                # Q2 输出产物（运行时生成）
│   ├── doc/                       # Q2 文字报告（运行时生成）
│   ├── README.md
│   ├── requirements.txt
│   └── 一键运行问题2.bat
├── 题目分析报告.md                 # 全题建模分析（Q1-Q3）
├── 术语表格.md                     # 术语、符号、单位统一
└── 使用指南.md                     # Skill 使用说明
```

## 环境配置

依赖 Anaconda/Miniconda 管理的 Python 环境。激活环境后安装依赖：

```powershell
# 问题1
cd q1_test
python -m pip install -r requirements.txt

# 问题2
cd q2_test
python -m pip install -r requirements.txt
```

核心依赖：`numpy`, `pandas`, `scipy`(含 HiGHS MILP 求解器), `openpyxl`, `matplotlib`。

## 运行方式

### 问题 1

```powershell
cd q1_test
# 一键运行（Windows）
一键运行问题1.bat

# 或手动运行
python scripts/run_q1.py --scenario both --eta 0.5 --delta 0.001 --mip-gap 0.001 --time-limit 600 --sensitivity --figures --reports
```

### 问题 2

```powershell
cd q2_test
# 一键运行（Windows）
一键运行问题2.bat

# 或手动运行（全量）
python scripts/run_q2.py --seed 2024 --raw-scenarios 1000 --reduced-scenarios 30 --beta 0.90 --lambda-grid 0:1:0.1 --out-sample 5000 --mip-gap 0.001 --time-limit 600

# 冒烟测试（快速验证）
python scripts/run_q2.py --raw-scenarios 20 --reduced-scenarios 5 --out-sample 50 --time-limit 30 --mip-gap 0.05 --figures
```

## 关键文档

| 文档 | 用途 |
|---|---|
| `题目分析报告.md` | 全题建模分析，第 6 章 Q1、第 12 章 Q2 为权威模型合同 |
| `术语表格.md` | 术语、符号、单位统一 |
| `q1_test/AGENT.md` | Q1 编程实现指导 |
| `q2_test/AGENT.md` | Q2 编程实现指导（权威合同） |

## 交付边界

- 原始 PDF、附件和模板只读，不修改。
- 各子工程独立管理算法、脚本和输出，使用相对路径保证可移植性。
- 未完成门禁检查时，不得声称编程交付完成。
- 所有数值、公式、图表、引用须独立验证后方可用于正式提交。
