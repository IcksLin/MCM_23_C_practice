# 问题3：相关、弹性与豆科互补优化

## 状态

已实现逐年边际情景、因子 t-Copula、价格弹性、豆科前茬互补、情景缩减、随机 MILP、OOS、审计和 Excel。

## 论文主候选流程

1. `q3_paper_candidate.yaml`：`N=2000,K=3,lambda=0.5`，释放整数变量搜索新结构。
2. `q3_paper_refine.yaml`：固定新 `y/r/b` 结构，用 `K=30`和 `lambda={0,0.5,1}` 精炼连续面积（仅作条件优化）。
3. 在5000个独立共同随机数情景上比较 Q2/Q3，并用多种子 OOS 复评。

## 当前结果

- 全整数发现 gap：31.4722%，未证明全局最优。
- 固定新结构后三个 lambda 均 `optimal`，条件 gap 0。
- `K=30` 期望利润：32,485,065.28元；90% CVaR：32,132,437.75元。
- 5000情景 OOS 平均利润：32,369,673.15元。
- 相对 Q2 可行基线增加：8,010,485.24元（32.88%）。
- 硬约束最大残差：8.73e-11。
- 完整实际边际代表审计：84列、333个固定Kendall对；seed=2024、N=2000的原始最大误差0.0462，门槛0.05。
- 多种子合格 OOS（3024/N=2000、3025/N=2000、3026/N=5000）平均：Q3 32,368,489.03元，Q2 24,358,774.52元，平均提升8,009,714.50元；三组配对正收益率均100%。
- seed=2025全整数冷启动900秒候选被否决：OOS 30,266,799.63元，gap 40.49%，均差于主候选。

条件 gap 0 不代表 Q3 全整数全局最优。原始情景相关审计已闭环，但K=30 PAM缩减集最大Kendall误差0.578且方向不一致，因此K=30只能用于条件方案搜索，不能作为正式风险分布认证；正式风险结论使用未缩减 OOS。

## 镜像运行

```powershell
.\run_problem.ps1 -ConfigPath q3_test\configs\current_result_mirror\config.yaml
```

镜像已冻结Q2基线和Q3发现结构，不依赖普通输出目录。输出写入 `doc/results/q3/reproduced/`。

## 产物

当前冻结参考结果位于 `doc/results/q3/`，其中 `result3.xlsx` 是方案展示工作簿，不冒充题目官方模板。历史实验和否决候选保留于 `outputs/q3/experiments/`。

gap 口径见 `../doc/三问MIP_gap口径说明.md`，实验基准见 `../doc/三问实验基准与对照.csv`。

## 工程结构

```text
q3_test/
├─ algorithms/
│  ├─ dependency.py                   # 因子t-Copula与相关审计
│  ├─ elasticity.py                   # 同季次交叉价格弹性
│  ├─ scenarios.py                    # 逐年边际、相关重排和弹性修正
│  ├─ scenario_reduction.py           # 分层PAM、尾部保护和缩减审计
│  ├─ model.py, solve.py              # 豆科互补MILP、发现/精炼求解
│  ├─ risk.py, evaluate.py            # 风险前沿、配对OOS与消融接口
│  ├─ validate.py                     # 农业、互补、相关、弹性和利润审计
│  └─ export_ooxml.py, plots.py       # Excel回填与图表
├─ scripts/run_q3.py                  # 主入口
├─ scripts/evaluate_multiseed.py      # 冻结方案多随机种子OOS复评
├─ scripts/p1_test.py                 # 最小纵向测试
├─ configs/current_result_mirror/     # 独立Config、冻结Q2/Q3结构、manifest
├─ outputs/q3/                        # 方案、审计、OOS、gap和复现产物
├─ doc/                               # 尝试解答、实现指导和代码审计
├─ tests/
└─ run.ps1
```

## 产物说明

| 文件 | 含义 | 交付判定 |
|---|---|---|
| `doc/results/q3/result3.xlsx` | K=30条件精炼的7年种植方案 | 优质近似解，非全局最优、非官方模板 |
| `selected_plan_q3.csv` | 627条非零种植记录 | 发现与精炼之间的整数结构输入 |
| `risk_frontier_q3.csv` | 三个lambda的条件目标、理论界和gap | 三点方案相同，不称完整风险前沿 |
| `paired_profits_q2_q3.csv` | 5000个共同随机数下Q2/Q3利润与差值 | OOS对照的原始证据 |
| `multiseed_oos_q3.csv` | 3个独立随机流的OOS统计、Kendall门禁和bootstrap区间 | 多随机流稳健性证据；3026的N=2000门禁失败，正式采用其N=5000复验 |
| `reduction_audit_q3.csv` | 84列、333对的原始/缩减Kendall门禁 | 原始通过、K=30缩减失败的机器证据 |
| `audit_q3.csv` | 硬约束、CVaR、互补、相关和弹性审计 | 当前硬约束通过，整体未认证 |
| `gap_summary_q3.csv` | 全整数发现gap与固定结构条件gap | 论文gap的唯一区分表 |
| `repro_q3.json` | 随机种子、N/K/OOS、lambda、求解模式和环境 | 当前精炼运行复现入口 |
| `delivery_manifest_q3.json` | 全链命令、环境、状态与输入/输出SHA-256 | 发现—精炼—多种子证据索引 |
| `experiments/seed2024_main_before_seed2025/` | seed=2024主候选冻结快照 | 当前晋级方案备份 |
| `experiments/seed2025_discovery_rejected/` | 900秒独立冷启动候选 | 被训练目标、gap和OOS共同否决 |
| `experiments/seed3026_n5000_oos.csv` | 第三随机流扩大到N=5000的复验 | Kendall误差0.0249，通过门槛 |
