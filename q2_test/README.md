# 问题2：情景随机 MILP 与均值-CVaR

## 状态

已实现 LHS、PAM、均值-CVaR、三级字典序、OOS、审计和 Excel。当前 `lambda=0/0.1` 有可行点，其余点超时无 incumbent；风险前沿不完整，未认证。

## 配置与运行

当前结果镜像位于 `configs/current_result_mirror/`；其中冻结Q1基线，避免隐式读取Q1普通输出目录。

```powershell
.\run_problem.ps1 -ConfigPath q2_test\configs\current_result_mirror\config.yaml
```

## 当前基准

- `lambda=0` 训练期望利润：24,376,595.14元。
- 5000情景 OOS 平均利润：24,377,039.02元。
- OOS 最差10%平均利润：24,172,746.88元。
- Stage 1 修正 gap：102.36%，理论界很松，只能作为可行基线。

`risk_frontier.csv` 旧 `dual_bound=544` 来自 Stage 3 激活数，不是利润上界。求解层已新增 `objective_value/objective_bound/corrected_gap`，现有 CSV 需重跑才会更新。

## 产物

冻结参考结果位于 `doc/results/q2/`；镜像运行输出到 `doc/results/q2/reproduced/`。历史求解产物仍保留于 `outputs/q2/`。

## 工程结构

```text
q2_test/
├─ algorithms/
│  ├─ io_data.py, preprocess.py       # 数据读取与ModelData
│  ├─ scenarios.py                    # LHS需求/产量/成本/价格情景
│  ├─ scenario_reduction.py           # PAM代表情景与权重
│  ├─ model.py, solve.py              # 均值-CVaR MILP与三级字典序
│  ├─ risk.py, validate.py            # CVaR/前沿选择与约束复算
│  ├─ export_ooxml.py                 # result2模板安全回填
│  └─ plots.py                        # 9类正式图表
├─ scripts/run_q2.py                  # 正式主入口
├─ scripts/p1_test.py, pipeline_test.py
├─ configs/current_result_mirror/     # 独立Config、冻结Q1基线、manifest
├─ outputs/q2/
├─ doc/
└─ run.ps1
```

## 产物说明

| 文件 | 含义 | 交付判定 |
|---|---|---|
| `result2.xlsx` | 选中的7年种植方案 | 当前为未认证可行基线 |
| `selected_plan.csv` | 非零面积长表 | Q3共同随机数对照输入 |
| `risk_frontier.csv` | 各lambda的Z、期望利润、CVaR、gap和字典序状态 | 当前前沿不完整 |
| `scenario_summary.csv` | 情景规模、分布和维度 | 生成审计 |
| `out_of_sample_profits.csv` | 每个OOS情景利润 | 统计检验原始值 |
| `out_of_sample_metrics.csv` | 均值、标准差、分位数和尾部均值 | 论文性能表来源 |
| `audit_q2.csv` | 硬约束、复算差与导出审计 | 可行性主证据 |
| `repro_q2.json` | 输入/输出哈希、参数和环境 | 复现清单 |
