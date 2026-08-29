# 问题 3 下游 Agent 权威实施合同

> 角色：编程手。目标：实现并验证 2024 CUMCM C 题问题 3。数学模型已经冻结；不得自行改成另一套模型、虚构参数来源或把未认证结果写成全局最优。

## 1. 必读文件

按顺序读取：

1. `../使用指南.md`
2. `../doc/2024_C题_农作物的种植策略.md`
3. `../题目分析报告.md` 第 12、13 章
4. `../术语表格.md`
5. `doc/Q3_尝试解答.md`
6. `doc/Q3_编程手实现指导.md`
7. 本文件

发生冲突时，题面与原始附件优先，其次是根目录第13章，再其次是本目录文档。发现模型不可实现时，携证据返回建模手，不得静默改公式。

## 2. 工作范围

必须实现：只读输入和哈希校验、Q2边际LHS、因子t-Copula七年相关重排、交叉价格弹性需求、豆类前茬互补线性化、尾部与相关结构保护的情景缩减、均值—下尾CVaR随机MILP、三级字典序、风险前沿、四组消融、共同随机数样本外比较、表格、候选Excel、图表、日志、报告和复现清单。

禁止：

- 修改 `../doc/C题/`、`../q1_test/`、`../q2_test/`；
- 给 `x,y,r` 增加情景下标；
- 把模拟载荷、弹性和增产率称为附件估计值；
- 将 `R_lat` 当成Spearman矩阵；
- 省略季次索引后重复计算价格趋势；
- 用普通 `openpyxl.save()` 覆盖模板；
- 在没有有限incumbent时输出正式方案；
- 使用冒烟结果生成正式图表；
- 未达到门槛时退出码仍返回0。

## 3. 冻结模型参数

基准：

```text
seed=2024
nu=5
correlation_scale=1.0
temporal_rho=0.5
elasticity_scale=1.0
gamma=0.03
beta=0.90
eta=0.5
lambda=0.0:1.0:0.1
```

载荷、弹性和边际区间见 `doc/Q3_尝试解答.md`。弱中强扫描只能从配置读取，禁止散落在代码中。

## 4. 输入保护

运行前验证：

| 输入 | SHA-256 |
|---|---|
| `../doc/C题/C题.pdf` | `C7B5E58BFF4189B8AFBA5505F7BFF7D4F08280FC291C51EF3F46134EBBF74F9A` |
| `../doc/C题/附件1.xlsx` | `5E98BF5E1B247624397E57E74759DA293BD72005DAA0E2DF3710A0DAC0E9EF6A` |
| `../doc/C题/附件2.xlsx` | `869081A3AB47D3BF8D0955106B622AAF0FD2C068FADA7948DA69B20EBF1D00CE` |
| `../doc/C题/附件3/result2.xlsx` | `6A1BA9FC28D14D0A4A795E5F0B7261FB6E32165517AFEE62BCD1931ABA5BEE8A` |
| `../题目分析报告.md` | `C5DAE0CFEB81344F49E2D2D6ED704A7368C56AFD2D1A1B343609EC99499AFB17` |
| `../术语表格.md` | `ED61B2FC173DA28A0FF0010749454D5027C5F18099EF235DA412BFE6B5ABE61B` |
| Q2 `selected_plan.csv` | `ACAE51363E165A007A3EBAE76DE1470C226673003552AEEFEEFE4D37A1FFC491` |
| Q2 `audit_q2.csv` | `699571C4DB10987E68E2D1220C65820CA775CEBCF9F7999291B2BD03B70DC251` |
| Q2 `repro_q2.json` | `E5854369C453922C1F4306012A563C04B3F1962EBEBF0083B16E12038260FF7A` |

根目录两个模型文档若经建模手修订，更新哈希并使旧M1失效；不要为了绕过哈希检查硬编码忽略。

## 5. Q2基线状态

Q2基线农业约束可行，但风险前沿不完整且未认证。任何输出只能使用标签 `q2_feasible_baseline`，不能使用 `q2_optimal` 或 `q2_certified`。Q3代码应从 `selected_plan.csv` 读取面积，不依赖可能被Excel占用的Q2工作簿。

## 6. 开发顺序

1. 路径、哈希、只读输入；
2. Q2边际生成单元测试；
3. `dependency.py` 与 Kendall 审计；
4. `elasticity.py` 与方向测试；
5. `b,w` 微型线性化测试；
6. Q3最小MILP与独立利润复算；
7. P1作者测试；
8. 未参与实现的独立P1；
9. 检查点、热启动和断电恢复；
10. 正式风险前沿；
11. 四组消融和样本外评估；
12. 图表、报告、复现清单；
13. 独立P2。

P1未PASS，不得运行全量计算和正式出图。

## 7. 关键数学断言

程序必须主动失败而不是打印警告后继续：

```text
min_eigenvalue(R_lat) >= -1e-10
diag(R_lat) == 1
max_abs_kendall_error <= 0.05
all marginal values within declared bounds
sum_{h != i}|e_ih| <= |e_ii|
e_ii < 0
max_abs(w - x*b) <= 1e-6
scenario weights >= 0 and sum == 1
all agriculture violations <= 1e-6
profit and CVaR recomputation differences <= 1e-4 yuan
```

## 8. 求解与断电恢复

Q2证明 SciPy/HiGHS 在正式规模上可能无法给所有lambda找到可行解。应优先使用支持MIP start的合法求解器；若只能用HiGHS，先降K并验证可行性，不能直接运行数小时后才发现前沿为空。

每个lambda完成后原子写入：

```text
outputs/q3/checkpoints/lambda_<value>.json
outputs/q3/checkpoints/lambda_<value>_plan.npz
```

检查点记录输入哈希、配置哈希、求解器版本、incumbent、bound、gap、三级状态和方案哈希。恢复时只有全部哈希一致才能跳过，否则重算该点。

## 9. 图表和报告

正式图必须由真实正式结果生成，原始/过程/结果每类至少3张，输出SVG与300 DPI PNG。图表数据源和生成命令进入复现清单。

报告必须披露所有模拟参数并非附件估计、Q2基线未认证、每个lambda状态与gap、三类敏感性、共同随机数比较以及失败点和回退条件。

## 10. 交付判断

只有 `audit_q3.csv` 全部硬门槛通过、风险前沿满足既定完整性规则、结果文件哈希一致且P2为PASS时，才允许退出码0和“完成”表述。

若仅有可行方案但未认证，保留结果并返回退出码2；若无可行方案，返回3。不要删除失败日志。
