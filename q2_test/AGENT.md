# 问题 2 下游 Agent 实现指导

> 本文件是问题 2 的权威实施合同，可直接交给下游 Agent。目标是实现并验证 2024 年高教社杯全国大学生数学建模竞赛 C 题问题 2；不得重新发明模型、虚构输入分布或修改赛题附件。

## 1. 角色、目标与禁止事项

下游 Agent 应按“编程手”职责工作：实现 Python 求解、真实运行、结果表、图表、日志和复现清单。建模结论已经冻结在工程根目录 `题目分析报告.md` 第 12 章。

必须做到：

1. 生成一套适用于全部不确定情景的 2024—2030 年共同种植方案。
2. 使用情景随机 MILP，在期望利润与低收益尾部风险之间进行可追溯权衡。
3. 保留问题 1 的全部农业硬约束、2023 历史边界和管理便利性约束。
4. 将唯一推荐方案写入 `outputs/q2/result2.xlsx`。
5. 独立复算全部收益、风险和约束，不只读取求解器目标值。

禁止：

- 修改 `../doc/C题/` 中任何文件。
- 覆盖 `../q1_test/` 的代码或结果。
- 把题面区间误写成真实概率分布。
- 为每个情景分别设置种植面积决策，再挑选最好情景；最终所有情景必须共享同一组种植决策。
- 未经建模手确认，删除适种、重茬、豆类、水浇地模式或最小面积约束。
- 在结果未认证时使用“全局最优”表述。

## 2. 路径与输入快照

工程根目录：

```text
D:\时光归墟\赛事\数模\practice_1
```

问题 2 工程目录：

```text
D:\时光归墟\赛事\数模\practice_1\q2_test
```

原始输入开始运行前必须校验：

| 文件 | SHA-256 |
|---|---|
| `../doc/C题/C题.pdf` | `C7B5E58BFF4189B8AFBA5505F7BFF7D4F08280FC291C51EF3F46134EBBF74F9A` |
| `../doc/C题/附件1.xlsx` | `5E98BF5E1B247624397E57E74759DA293BD72005DAA0E2DF3710A0DAC0E9EF6A` |
| `../doc/C题/附件2.xlsx` | `869081A3AB47D3BF8D0955106B622AAF0FD2C068FADA7948DA69B20EBF1D00CE` |
| `../doc/C题/附件3/result2.xlsx` | `6A1BA9FC28D14D0A4A795E5F0B7261FB6E32165517AFEE62BCD1931ABA5BEE8A` |
| `../题目分析报告.md` | `10590BD483936EA5E1736E6DA79CBDB77FF1D791148FBD942D7F1DBD9E3652BD` |
| `../术语表格.md` | `DB42CC1A165525945DEF217E6DBFC7BFB0016C1B80B92242FE53E31B66E0F583` |

问题 1 的 `result1_1.xlsx` 仅用于情景缩减的代理利润和基线比较。它属于生成产物，开始实现时应记录当前哈希和 `audit.csv` 状态；若随后改变，应使对应 P1/P2 结论失效并重新验证。

## 3. 最终采用的模型

只实施一个模型体系：

```text
Latin Hypercube Sampling
→ 利润分层、尾部保护的 PAM k-medoids 情景缩减
→ 均值—下尾 CVaR 情景随机 MILP
→ 三级字典序优化
→ 样本外评估与压力测试
```

预算不确定集鲁棒 MILP 只是模型失败后的重新建模方向，本任务不实现、不比较，不得把其通用公式冒充第二套完成模型。

## 4. 不确定参数生成规则

基期为 2023 年，令 `k=t-2023`。

### 4.1 预期销量

小麦和玉米逐年抽取：

$$
g^D_{its\omega}\sim U(0.05,0.10),
\qquad
D_{its\omega}=D_{i,t-1,s,\omega}(1+g^D_{its\omega}).
$$

其他作物每个未来年相对 2023 年变化：

$$
D_{its\omega}=D_{i,2023,s}(1+\epsilon^D_{its\omega}),
\qquad
\epsilon^D_{its\omega}\sim U(-0.05,0.05).
$$

其他作物不进行逐年复合漂移。

### 4.2 亩产量

$$
q_{jit s\omega}=q_{ji,2023,s}(1+\epsilon^q_{jit s\omega}),
\qquad
\epsilon^q_{jit s\omega}\sim U(-0.10,0.10).
$$

亩产量必须保留 `plot` 或 `land_type` 索引，不能只按作物存储；附件中同一作物在不同地类的亩产量不同。

### 4.3 种植成本

主模型按题面中心趋势：

$$
c_{jits}=c_{ji,2023,s}(1.05)^k.
$$

4%—6% 年增长只用于敏感性检验，不得伪装为题面给出的随机区间。

### 4.4 销售价格

- 粮食：`p_t=p_2023`。
- 蔬菜：`p_t=p_2023(1.05)^k`。
- 羊肚菌：`p_t=p_2023(0.95)^k`。
- 其他食用菌：逐年抽取下降率 `g^p∈[0.01,0.05]` 并复合递推。

### 4.5 分布声明

题目只给变化范围，没有给概率分布。主模型采用区间内均匀分布作为基准假设；必须额外使用同区间三角分布和参数端点进行复验。问题 2 暂不人为加入参数相关性，因为相关性属于问题 3 的明确要求。

### 4.6 超产口径

问题 2 没有重新规定超产处理。主结果采用保守口径：

$$
u_{its\omega}=\min(Q_{its\omega},D_{its\omega}),
$$

超出部分不产生收入。50% 折价销售只作为口径敏感性，并与主结果分开报告。

## 5. 决策变量与情景变量

共同决策不带情景下标：

- `x_jits>=0`：地块—作物—年份—季次种植面积，亩。
- `y_jits∈{0,1}`：对应种植组合是否启用。
- `r_jt∈{0,1}`：水浇地模式变量。

情景变量带 `omega`：

- `Q_its_omega`：情景总产量，斤。
- `u_its_omega`：情景正常价销量，斤。
- `Pi_omega`：情景七年利润，元。
- `xi_omega>=0`：CVaR 下行超额变量，元。
- `zeta`：下尾利润分位辅助变量，元。

不能建立 `x_jits_omega`。若每个情景有独立面积，模型就提前知道未来，且无法输出唯一的 `result2.xlsx`。

## 6. 情景利润与 CVaR

情景产量：

$$
Q_{its}^{\omega}=\sum_j q_{jits}^{\omega}x_{jits}.
$$

销量线性上界：

$$
0\le u_{its}^{\omega}\le Q_{its}^{\omega},
\qquad
u_{its}^{\omega}\le D_{its}^{\omega}.
$$

情景利润：

$$
\Pi_{\omega}
=\sum_{t,i,s}p_{its}^{\omega}u_{its}^{\omega}
-\sum_{j,i,t,s}c_{jits}^{\omega}x_{jits}.
$$

主置信水平 `beta=0.90`。下尾利润 CVaR：

$$
\operatorname{LCVaR}_{\beta}(\Pi)
=\zeta-
\frac{1}{1-\beta}
\sum_{\omega}\pi_{\omega}\xi_{\omega},
$$

$$
\xi_{\omega}\ge\zeta-\Pi_{\omega},
\qquad \xi_{\omega}\ge0.
$$

风险目标：

$$
\max Z_{\lambda}
=(1-\lambda)\sum_{\omega}\pi_{\omega}\Pi_{\omega}
+\lambda\operatorname{LCVaR}_{\beta}(\Pi),
\qquad \lambda\in[0,1].
$$

## 7. 必须继承的问题 1 硬约束

复用 `q1_test` 的已验证逻辑，但复制或抽离后必须重新测试，不能直接假设正确。

1. 适种矩阵及变量上下界。
2. 每个有效季次的地块面积守恒。
3. 平旱地、梯田、山坡地、水浇地、普通大棚、智慧大棚的作物与季次规则。
4. 水浇地单季水稻与两季蔬菜模式互斥。
5. 水浇地选择蔬菜模式时，第二季必须在大白菜、白萝卜、红萝卜中**恰好启用一种**；水稻模式时三者启用数为0。
6. 同年相邻季及跨年相邻季禁止连续重茬。
7. 2023 历史种植到 2024 决策的边界约束。
8. 任意滚动三年窗口豆类累计种植面积不少于地块面积。
9. 最小非零种植面积：主值 `x>=0.5*A*y`；其他比例只做敏感性分析。

任何情景只改变利润参数，不改变土地面积、适种、轮作等农业可行域，因此硬约束只建立一次。

## 8. 三级字典序求解

金额容差定义为：

$$
\varepsilon(V)=\max\{1\text{ 元},10^{-6}|V|\}.
$$

每个 `lambda` 均执行三阶段：

1. 最大化 `Z_lambda`，得到 `Z_lambda*`。
2. 加入 `Z_lambda >= Z_lambda* - epsilon(Z_lambda*)`，最大化期望利润，得到 `E*`。
3. 加入 `E[Pi] >= E* - epsilon(E*)`，最小化 `sum(y)`。

第二级不能省略。特别是 `lambda=1` 时，纯 CVaR 目标可能不推动非尾部情景的 `u` 达到 `min(Q,D)`。最终导出的风险指标必须固定 `x` 后独立复算真实 `u=min(Q,D)`。

## 9. 情景生成与缩减

### 9.1 原始情景

- 使用 `scipy.stats.qmc.LatinHypercube`。
- 默认种子 2024。
- 生成 `N0=1000` 个原始情景。
- 将 `[0,1]` 分层样本映射到第 4 节的参数范围。

长表至少包含：

```text
scenario, year, plot, land_type, crop, season,
demand, yield, cost, price
```

销量、售价允许在地块行中重复，但必须通过唯一的作物—年份—季次表生成；亩产量、成本不得丢失地块或地类维度。

### 9.2 尾部保护缩减

1. 固定问题 1 的 `result1_1`，复算1000个原始情景的代理利润。
2. 按代理利润分成10个等频层。
3. 最低利润层至少分配 `ceil(0.1K)` 个代表情景。
4. 对标准化变化率向量，在每层内使用固定种子的 PAM k-medoids 和 L1 距离。
5. 代表情景必须来自原始样本，不能使用不在题面区间内的聚类中心。
6. 情景权重为所属簇样本数除以1000；断言权重非负且总和为1。

本工程不强制引入额外的 k-medoids 第三方包。建议在 `scenario_reduction.py` 中实现确定性 PAM，并为以下行为写单元测试：L1 距离矩阵、固定种子/固定初始 medoid、交换更新、空簇重新指派、目标值不增以及距离并列时按原始 `scenario` 编号破同分。若改用第三方包，必须把精确包名和版本写入 `requirements.txt` 与复现清单。

分别求解 `K=20,30,50`。如果 `K=30` 与 `K=50` 的期望利润、下尾利润或主要作物面积变化超过2%，正式结果改用50个代表情景。

## 10. 风险前沿与唯一方案选择

求解：

```text
lambda = 0.0, 0.1, ..., 1.0
```

每个方案的期望利润和下尾利润必须由固定种植面积独立复算。将两项指标分别归一化到 `[0,1]`，计算每个非端点到两端连线的垂直距离。

- 最大距离至少为0.02：选择距离最大点。
- 垂距在 `1e-9` 内并列：依次选择较大 `lambda`、较高期望利润、较少启用次数。
- 最大距离小于0.02：选择下尾利润达到前沿最大值99%的最小 `lambda`，再按期望利润和启用次数破同分。

选择逻辑必须写成确定性函数并单元测试，不允许人工观察图片后挑选。

## 11. 建议代码结构与接口

```text
q2_test/
├─ algorithms/
│  ├─ paths.py
│  ├─ io_data.py
│  ├─ preprocess.py
│  ├─ scenarios.py
│  ├─ scenario_reduction.py
│  ├─ model.py
│  ├─ solve.py
│  ├─ risk.py
│  ├─ validate.py
│  ├─ export_ooxml.py
│  └─ plots.py
├─ scripts/
│  ├─ smoke_test.py
│  ├─ p1_test.py
│  ├─ pipeline_test.py
│  └─ run_q2.py
├─ outputs/q2/
├─ doc/
├─ requirements.txt
└─ run.ps1
```

建议接口：

```python
load_inputs() -> RawData
preprocess(raw: RawData) -> ModelData
generate_raw_scenarios(data, n: int, seed: int, distribution="uniform") -> ScenarioSet
reduce_scenarios(raw, k: int, baseline_plan) -> ReducedScenarioSet
build_q2_model(data, scenarios, beta: float, risk_lambda: float) -> Model
solve_risk_stage(model) -> SolveResult
solve_expected_stage(model, risk_result) -> SolveResult
solve_fragmentation_stage(model, expected_result) -> Solution
recompute_scenario_profits(plan, scenarios) -> DataFrame
select_unique_plan(frontier: DataFrame) -> str
evaluate_fixed_plan(plan, scenarios) -> EvaluationReport
validate_solution(plan, data, scenarios) -> AuditReport
export_result2_workbook(plan, template_path, output_path) -> None
generate_figures(data, scenarios, frontier, evaluation, output_dir) -> None
```

## 12. 测试顺序

### 12.1 确定性单元测试

至少覆盖：

1. 所有情景参数严格位于题面范围。
2. 小麦、玉米销量按年复合；其他作物不复合漂移。
3. 粮食、蔬菜、普通食用菌和羊肚菌价格递推分别正确。
4. 同作物不同地类的亩产量、成本没有被错误合并。
5. `Q<D`、`Q=D`、`Q>D` 时 `u=min(Q,D)`。
6. 手算3情景例子的期望利润和CVaR与代码一致。
7. `lambda=1` 经过第二级求解后，独立利润复算无差异。
8. 风险前沿的膝点、并列和无膝点选择规则均有固定预期结果。
9. PAM输出是原始样本，最低利润层medoid数量达标，权重和为1。

### 12.2 作者最小纵向测试（独立 P1 的前置条件）

使用真实附件、少量真实地块、2—3年、5个原始情景和3个代表情景：

```text
输入哈希
→ 清洗
→ 情景生成
→ 情景缩减
→ 随机MILP三级求解
→ 独立利润/CVaR复算
→ OOXML候选回填
→ 模板结构审计
```

作者测试要求：退出码0；无NaN/Inf；所有硬约束最大违约不超过 `1e-6`；利润与CVaR复算差不超过 `1e-4` 元；Excel面积回读差不超过 `1e-4` 亩。

作者测试通过后冻结输入、命令和日志，立即派发一个**未参与代码实现和修正**的质检 Subagent 执行独立 P1。该 Subagent 应从问题2工程目录 `D:\时光归墟\赛事\数模\practice_1\q2_test` 在隔离环境或只读副本中运行最小命令，核对真实输入追溯、单位、范围、农业约束、情景边界、`lambda=1` 利润复算和 OOXML 结构，并按 `math-modeling/references/Subagent调度.md` 返回固定回执。独立 P1 未返回 `PASS`，不得运行完整风险前沿、正式图表和样本外实验；作者脚本通过不得表述为独立 P1 通过。

## 13. 求解器和运行策略

优先级：

1. Gurobi或COPT：支持多线程和MIP start，优先复用问题1方案及相邻 `lambda` 方案热启动。
2. CPLEX或SCIP。
3. HiGHS：可运行但大规模情景MILP可能很慢，且SciPy接口不便热启动。

先后顺序：`K=5` 冒烟、`K=20`、`K=30`、必要时 `K=50`。同一机器不要并行运行多个抢占全部CPU的MILP。GPU不是当前MILP的有效直接加速方式。

每次求解必须记录：状态、incumbent、best bound、MIP gap、节点数、时间和求解器版本。若达到时限但有可行解，保留可行方案并标记未认证；没有有限 incumbent 时禁止输出正式工作簿。

## 14. 样本外与压力验证

最终方案用独立随机流生成至少5000个新情景，只复算、不重新优化。输出：

- 平均利润、标准差；
- 10%利润分位数；
- 最低10%情景平均利润；
- 最小利润、亏损概率；
- 各年利润和主要作物面积稳定性。

至少比较：

1. `../q1_test/outputs/q1/result1_1.xlsx` 可行确定性基线；当前审计为 `feasible=True, certified=0`，只能作代理和比较基线，不能称已认证最优方案；
2. `lambda=0` 风险中性方案；
3. 唯一选出的风险方案。

半价超产敏感性才使用 `../q1_test/outputs/q1/result1_2.xlsx`，且同样注明其当前未认证状态。另做联合不利压力情景：低销量、低亩产、高成本、不利售价同时出现。再将均匀分布替换为同区间三角分布，并更换至少5个随机种子。

若主要作物面积、期望利润或下尾利润的变异系数超过5%，增加情景数；若分布改变导致推荐结构明显翻转，停止交付并返回建模手重新建立鲁棒模型合同。

## 15. `result2.xlsx` 的安全回填

已知模板页眉页脚会触发 `openpyxl` 解析预警，禁止使用普通 `openpyxl.save()` 整体保存覆盖模板。

正确流程：

1. 字节复制只读模板为 `.candidate.xlsx`。
2. 将XLSX作为ZIP读取，仅修改7个 `xl/worksheets/sheet*.xml` 的 `<sheetData>` 中目标数值单元格。
3. 保留既有单元格样式索引，只增加或替换数值节点。
4. 除被改的7个工作表XML外，其余ZIP成员逐项SHA-256必须完全相同。
5. 对工作表XML移除 `<sheetData>` 后做规范化比较；合并区域、页眉页脚、打印设置、尺寸和关系必须一致。
6. 回读全部面积并与内存方案比较。
7. 只有约束、利润、CVaR、OOXML结构和Excel回读全部通过，才原子替换 `outputs/q2/result2.xlsx`。

## 16. 必备审计字段

`outputs/q2/audit_q2.csv` 至少包含：

- 最大面积守恒违约；
- 最大不适种面积；
- 最大面积—激活上界违约；
- 最大最小种植面积违约；
- 重茬违反次数；
- 滚动三年豆类最小裕度；
- 水浇地模式冲突数；
- 水浇地蔬菜模式第二季根菜启用数偏离1的最大值及违反次数；
- `max(u-Q)` 与 `max(u-D)`；
- 最大产量平衡差；
- 0-1整数性违约；
- 情景利润复算差；
- CVaR复算差；
- Excel面积回读差；
- OOXML非目标结构差异数；
- 求解状态、bound、gap和认证状态。

## 17. 图表和结果文件

正式绘图前必须加载本地 `科研可视化工具` Skill，先对各图数据执行数据剖析并为每张图写明“核心结论—证据链—图型—最终尺寸—统计口径”的图表契约。至少形成9张逻辑图，每张输出SVG、至少300 DPI PNG和灰度预览。

原始数据类：

1. 各不确定参数的年度变化区间；
2. LHS样本边际分布与覆盖；
3. 原始情景与缩减情景的均值/分位数对比。

过程类：

1. 情景生成—缩减—随机MILP—样本外验证流程；
2. 风险前沿与唯一方案选择；
3. 场景数或求解上下界收敛图。

结果类：

1. 最终年度种植面积热力图；
2. 三种基线方案的样本外利润分布；
3. 期望利润、下尾利润和压力利润对比。

文件名使用 `raw_q2_*`、`process_q2_*`、`result_q2_*`。

所有正式图必须调用

```text
<SKILL_ROOT>/tools/figure/scripts/export_figure.py
```

或其中的 `export_figure()` 导出，不得直接把普通 `savefig` 产物冒充出版级图。作者每次导出后运行：

```powershell
python "<SKILL_ROOT>/tools/figure/scripts/check_figure.py" "D:\时光归墟\赛事\数模\practice_1\q2_test\outputs\q2\figures" --strict
```

并实际打开彩色PNG与灰度预览，在论文预计尺寸下检查缺字、裁切、遮挡、颜色、尺度、图例和面板层级。P2独立质检还必须运行：

```powershell
python "<SKILL_ROOT>/references/roles/编程手/scripts/figure_audit.py" "D:\时光归墟\赛事\数模\practice_1\q2_test\outputs\q2\figures" --questions q2 --strict
```

两条命令退出码非0或实际读图发现问题时，必须改绘图代码、重新导出和复审；不得直接修位图或关闭门禁。

正式输出至少包括：

```text
outputs/q2/result2.xlsx
outputs/q2/scenario_summary.csv
outputs/q2/risk_frontier.csv
outputs/q2/out_of_sample_profits.csv
outputs/q2/out_of_sample_metrics.csv
outputs/q2/audit_q2.csv
outputs/q2/repro_q2.json
outputs/q2/logs/
outputs/q2/figures/
doc/Q2_建模实现报告.md
```

## 18. 一键入口与终端进度

实现：

```powershell
cd "D:\时光归墟\赛事\数模\practice_1\q2_test"
python scripts/run_q2.py --seed 2024 --raw-scenarios 1000 --reduced-scenarios 30 --beta 0.90 --lambda-grid 0:1:0.1 --out-sample 5000 --mip-gap 0.001 --time-limit 600
```

使用全局 `run_problem.ps1 -ConfigPath <path>` 启动，启动器固定UTF-8环境并透传退出码。

建议进度节点：输入5%、清洗10%、情景生成20%、缩减30%、各 `lambda` 求解30%—75%、样本外评估85%、Excel审计92%、图表和报告98%、完成100%。进度百分比表示流水线阶段，不伪装成MILP内部收敛比例。

## 19. P2最终门禁与回报

代码、结果、图表和参数冻结后，作者先从问题2工程目录 `D:\时光归墟\赛事\数模\practice_1\q2_test` 启动完整复现和清单自检；随后派发一个**未参与实现和修正**的质检 Subagent 执行独立 P2。独立 P2 必须在同一问题2工程目录亲自运行唯一复现命令和上述 `figure_audit.py --questions q2 --strict`，并实际查看彩色图与灰度预览。只有独立 P2 返回 `PASS`，才能声称编程交付完成。P2至少核对：

- 输入哈希、随机种子、依赖和唯一命令；
- P1全部指标；
- `K=20/30/50`场景收敛；
- 5000个样本外情景；
- 5个随机种子及三角分布复验；
- 9张逻辑图及SVG文字可编辑性；
- `result2.xlsx`模板结构完整；
- 复现清单中的每个输出哈希。

最终回报必须列出：P1与P2状态、唯一方案的 `lambda` 和选择理由、期望利润、下尾利润、样本外指标、压力利润、求解状态/bound/gap、全部违约量、结果路径和仍存在的阻塞。

## 20. 失败回退规则

- 公式或题意冲突：停止，返回建模手修订根目录两个权威Markdown。
- `K=30`过大：先用 `K=10/20` 和热启动取得可行解，不得静默删除情景风险目标。
- 没有可行整数解：输出IIS/conflict或最小失败实例，不得放松农业硬约束。
- 概率假设导致结构翻转：停止交付，回到建模手另立鲁棒模型合同。
- 未达MIP gap：如实报告可行解、bound和gap，不称全局最优。
- Excel结构审计失败：保留旧正式结果和候选文件，禁止覆盖模板或正式工作簿。

本实现仅供学习、研究与参赛队伍复核。正式使用前需由队伍成员人工检查数据、模型、代码、表格、图表和结论，并遵守竞赛规则与学术诚信要求。
