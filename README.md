# 2024 年高教社杯数学建模 C 题：农作物种植策略

> 工程已实现问题1—3的建模、求解、审计和结果导出。未通过最优性认证的结果只能称为“可行解”或“优质近似解”。

## 当前状态

| 子工程 | 实现状态 | 当前结果 | 认证边界 |
|---|---|---|---|
| `q1_test/` | 两种滞销情形确定性 MILP、审计、Excel、敏感性已实现 | 有可行解 | gap 较大，未证明最优 |
| `q2_test/` | LHS + PAM + 均值-CVaR + 三级字典序已实现 | `lambda=0/0.1` 有可行点 | 前沿不完整，未认证 |
| `q3_test/` | 相关情景、弹性、豆科互补、全整数发现与固定结构精炼已实现 | 新种植结构 + `K=30` 精炼方案 | 全整数 gap 31.47%；条件 gap 0 |

## 统一实验基准

- `doc/三问实验基准与对照.csv`：当前 incumbent、正向理论界、修正 gap、OOS 和审计状态。
- `doc/三问实验评价标准.md`：后续调参的判定规则。
- `doc/三问MIP_gap口径说明.md`：最大化目标的正向界与 gap 换算。

## 运行

```powershell
.\run_problem.ps1 -ConfigPath q1_test\configs\current_result_mirror\config.yaml
.\run_problem.ps1 -ConfigPath q2_test\configs\current_result_mirror\config.yaml
.\run_problem.ps1 -ConfigPath q3_test\configs\current_result_mirror\config.yaml
```

统一启动器只接收Config路径，先校验镜像输入SHA-256，再按 `problem` 调用对应主程序。三问Config互不继承。使用 `-ValidateOnly` 可只做解析与哈希预检。输出统一写入 `doc/results/qN/reproduced/`；题目模板只读。

## 交付原则

1. `doc/C题/` 原始附件和模板只读。
2. 硬约束不通过时，利润再高也不得入选。
3. Stage 3 最少激活数 gap 不得冒充利润 gap。
4. Q3 条件 gap 0 不等于全整数全局 gap 0。
5. 论文数值必须可追溯到 CSV/JSON 产物。

## 全局工程结构

```text
practice_1/
├─ doc/
│  ├─ C题/                         # 题面PDF、附件1/2、附件3结果模板（只读）
│  ├─ results/q1|q2|q3/            # 当前参考结果及镜像复现输出
│  ├─ 三问实验基准与对照.csv       # 统一基准数值
│  ├─ 三问实验评价标准.md           # 新实验入选规则
│  └─ 三问MIP_gap口径说明.md        # gap与正向理论界
├─ q1_test/                      # 问题1独立工程
├─ q2_test/                      # 问题2独立工程
├─ q3_test/                      # 问题3独立工程
├─ scripts/run_with_config.py    # Config校验与问题路由
├─ run_problem.ps1               # 统一启动脚本
├─ 题目分析报告.md                 # Q1—Q3数学模型
├─ 术语表格.md                     # 符号、单位和口径
└─ 使用指南.md
```

每个子工程统一采用 `algorithms/ + scripts/ + configs/ + outputs/ + doc/`。具体模块和产物含义见对应子问题 README。

## 全局核心产物

| 产物 | 路径 | 用途 |
|---|---|---|
| Q1两种方案 | `doc/results/q1/result1_1.xlsx`、`result1_2.xlsx` | 当前冻结参考方案 |
| Q2方案 | `doc/results/q2/result2.xlsx` | 当前冻结参考方案 |
| Q3优质近似解 | `doc/results/q3/result3.xlsx` | 非官方模板的方案展示文件 |
| 三问数值基准 | `doc/三问实验基准与对照.csv` | 后续优化对照的唯一统一表 |
| gap口径 | `doc/三问MIP_gap口径说明.md` | 防止把最小化负目标或Stage 3 gap当成利润gap |
