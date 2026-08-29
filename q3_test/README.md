# 问题 3 工程规划与实现入口

本目录用于实现 2024 年高教社杯全国大学生数学建模竞赛 C 题问题 3。当前只完成工程骨架、数学方案和编程手实施合同，尚未编写求解代码，也没有问题 3 数值结果。

## 阅读顺序

1. `../题目分析报告.md` 第 13 章：权威数学模型合同。
2. `../术语表格.md`：符号、单位和禁止混用说明。
3. `doc/Q3_尝试解答.md`：问题 3 的完整解答路线。
4. `doc/Q3_编程手实现指导.md`：模块接口、测试和运行流程。
5. `AGENT.md`：交给下游 Agent 的强制实施约束。

## 当前工程状态

- 根目录第 13 章此前已完成建模质检；本工程交接文档的 M1 状态以本次最终交付回执为准。
- 问题 2 正式流水线已经运行，但风险前沿不完整，推荐方案未通过最优性认证。
- 问题 2 当前方案可以作为可行对照基线，禁止称为“问题 2 已证明最优解”。
- `q3_test` 不得修改 `q1_test`、`q2_test` 或 `doc/C题` 中的任何文件。

## 规划目录

```text
q3_test/
├─ algorithms/
│  ├─ __init__.py
│  ├─ paths.py
│  ├─ io_data.py
│  ├─ preprocess.py
│  ├─ dependency.py
│  ├─ elasticity.py
│  ├─ scenarios.py
│  ├─ scenario_reduction.py
│  ├─ model.py
│  ├─ solve.py
│  ├─ risk.py
│  ├─ evaluate.py
│  ├─ validate.py
│  ├─ export_ooxml.py
│  └─ plots.py
├─ configs/
│  └─ q3_default.yaml
├─ scripts/
│  ├─ smoke_test.py
│  ├─ p1_test.py
│  ├─ pipeline_test.py
│  ├─ run_q3.py
│  └─ regenerate_figures.py
├─ tests/
│  ├─ test_dependency.py
│  ├─ test_elasticity.py
│  ├─ test_complementarity.py
│  ├─ test_scenarios.py
│  └─ test_risk_selection.py
├─ doc/
│  ├─ Q3_尝试解答.md
│  └─ Q3_编程手实现指导.md
├─ outputs/q3/
│  ├─ figures/
│  └─ logs/
├─ AGENT.md
├─ README.md
├─ requirements.txt
├─ requirements-solver-optional.txt
└─ 一键运行问题3.bat
```

上表是目标结构。当前仅创建文档、目录占位和依赖规划；`.py`、配置文件与批处理必须由下游编程手按 P1 顺序实现，不能创建空壳后一键运行却输出伪结果。

## 主数据流

```text
只读附件 + Q2可行基线
  → 数据清洗与输入哈希
  → Q2边际LHS样本
  → 因子t-Copula相关重排
  → 交叉价格弹性修正需求
  → 尾部保护与相关保护的情景缩减
  → 含豆类前茬互补增益的随机MILP
  → 三级字典序求解与风险前沿选择
  → 四组消融和共同随机数样本外评估
  → 约束/相关/利润/Excel审计
  → 表格、图表、报告与复现清单
```

## 预期命令合同

下游完成实现后，应支持：

```powershell
# 最小纵向测试
python scripts/p1_test.py

# 小规模完整流水线
python scripts/pipeline_test.py

# 正式求解；具体规模可在P1/P2后根据求解器能力调整
python scripts/run_q3.py --config configs/q3_default.yaml --figures --reports
```

正式入口必须打印进度条，并在未达到交付门槛时返回非零退出码。不得提供绕过认证的默认参数。

## 明确非目标

- 当前阶段不运行问题 3 正式求解。
- 不把单年附件数据伪装成估计得到的真实相关系数或真实弹性。
- 不实现未触发回退条件的分布鲁棒优化或多智能体模型。
- 不覆盖问题 2 的结果工作簿。

## 验收门槛

- P1：相关生成、弹性、互补线性化、最小 MILP、复算和输出闭环通过。
- 正式计算前：`R_lat` 半正定；主要 Kendall 误差不超过 0.05；全部边际范围正确。
- 结果：农业硬约束最大违约不超过 `1e-6`；利润/CVaR复算差不超过 `1e-4` 元。
- 比较：Q2 与 Q3 使用共同随机数和同一批 Q3 样本外情景。
- P2：代码、结果、每类至少 3 张图和复现清单冻结后独立通过。

请下游 Agent 先阅读并确认 `AGENT.md`，再开始实现。
