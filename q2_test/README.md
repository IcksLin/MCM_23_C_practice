# 问题 2 编程实现与运行说明

本目录实现 2024 年数学建模国赛 C 题问题 2 的情景随机 MILP 求解，包含 LHS 情景生成、PAM k-medoids 情景缩减、均值-CVaR 风险权衡、三级字典序优化、样本外评估和安全 Excel 回填。

## 一键运行

Windows 下双击 `一键运行问题2.bat`，或在本目录终端执行：

```bat
一键运行问题2.bat
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 手动运行

```powershell
# 正式求解（11个lambda各含三级MILP，实际耗时取决于CPU和gap）
python scripts/run_q2.py --seed 2024 --raw-scenarios 1000 --reduced-scenarios 30 --beta 0.90 --lambda-grid 0:1:0.1 --out-sample 5000 --mip-gap 0.001 --time-limit 600 --eta 0.5 --figures --reports

# 完整流水线冒烟测试
python scripts/pipeline_test.py

# P1 门禁（最小纵向验证）
python scripts/p1_test.py
```

主要参数：

- `--raw-scenarios N`：LHS 原始情景数（默认 1000）。
- `--reduced-scenarios K`：PAM 缩减后情景数（默认 30）。
- `--beta`：CVaR 置信水平（默认 0.90，即最差 10% 尾部）。
- `--lambda-grid start:stop:step`：风险偏好网格（默认 0:1:0.1，11 个点）。
- `--out-sample N`：样本外评估情景数（默认 5000）。
- `--eta`：最小种植面积比例。
- `--mip-gap`、`--time-limit`：求解终止条件。
- `--allow-uncertified`：仅用于P1/冒烟测试。正式运行不使用；未达到认证门槛时程序以退出码2结束并明确标记候选解。

## 模型口径

- 决策变量与 Q1 一致（地块—作物—年份—季次面积 + 激活二元变量），但所有参数（需求、产量、成本、价格）均为随机情景。
- LHS 抽样生成 N 个原始情景，PAM k-medoids 缩减为 K 个代表性情景（带尾部保护，最差利润层至少占 10%）。
- 目标函数：均值-CVaR 加权 Z_lambda = (1-lambda)*E[Pi] + lambda*LCVaR_beta。
- 三级字典序优化：① 最大化 Z_lambda → ② 最大化 E[Pi]（约束 Z_lambda >= Z* - eps）→ ③ 最小化种植激活数（约束 E[Pi] >= E* - eps）。
- 风险前沿膝点选择：在 lambda 网格上寻找利润-CVaR 帕累托前沿的膝点。

## 代码职责

- `algorithms/paths.py`：相对路径配置。
- `algorithms/io_data.py`：只读 Excel 读取（附件 1/2 + result2 模板）。
- `algorithms/preprocess.py`：数据清洗 & ModelData 组装。
- `algorithms/scenarios.py`：LHS 情景生成（需求/产量/成本/价格）。
- `algorithms/scenario_reduction.py`：PAM k-medoids 情景缩减（带尾部保护）。
- `algorithms/model.py`：均值-CVaR 随机 MILP 构建。
- `algorithms/solve.py`：三级字典序求解 + LP 约束违反审计。
- `algorithms/risk.py`：CVaR 复算 & 风险前沿膝点选择。
- `algorithms/validate.py`：约束审计。
- `algorithms/export_ooxml.py`：安全 Excel 模板回填（ZIP/XML 操作）。
- `algorithms/plots.py`：9 张图表生成。
- `scripts/run_q2.py`：正式流水线入口 + 进度条。
- `scripts/p1_test.py`：P1 门禁（最小纵向验证）。
- `scripts/pipeline_test.py`：极小参数完整流水线测试。

## 产物文件说明

所有产物位于 `outputs/q2/`：

| 文件 | 含义 |
|---|---|
| `result2.xlsx` | 最终种植方案（2024—2030，7 个年份工作表），回填到 result2 模板 |
| `risk_frontier.csv` | 风险前沿表：每个 lambda 的 Z_lambda、期望利润、CVaR、激活数、gap、求解时间、字典序阶段、认证状态 |
| `scenario_summary.csv` | 情景摘要：原始情景数、种子、分布类型、各参数维度 |
| `selected_plan.csv` | 选中方案的非零种植计划（地块/作物/年份/季次/面积） |
| `out_of_sample_profits.csv` | 选中方案、Q1基线和风险中性方案在同一随机流上的样本外利润 |
| `out_of_sample_metrics.csv` | 样本外评估汇总：均值、标准差、分位数、最低利润、亏损概率、CVaR |
| `audit_q2.csv` | 约束审计报告：面积/轮作/销售/利润/CVaR/整数等约束的违反量和可行判定 |
| `repro_q2.json` | 可复现性清单：输入哈希、Python 版本、参数、求解统计、输出哈希 |
| `figures/` | 9 张图表（PNG + SVG），灰度预览位于 `figures/_qa/` |

## 当前产物状态

当前 `outputs/q2/` 是修复后的流水线冒烟产物（10个原始情景、1个代表情景、20个样本外情景、仅 `lambda=0`），用于证明读取、求解、审计、Excel回填和绘图能够贯通。它约束可行但未认证，**不是问题2正式最终答案**。正式结果必须由上述1000/30/5000唯一命令重新生成，并依据 `audit_q2.csv`、风险前沿完整性和求解认证状态决定能否用于问题3。
