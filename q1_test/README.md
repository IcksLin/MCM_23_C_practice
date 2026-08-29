# 问题1：确定性种植 MILP

## 状态与口径

已实现两种滞销口径的建模、求解、字典序减少碎片、审计、Excel、敏感性和绘图。当前产物可行，但未完成最优性认证。

- 情形1：超出预计销量的产量不产生收入，按浪费处理。
- 情形2：超额产量按正常售价的50%销售。
- 预计销量由2023年种植面积与亩产的代理量推算，不伪造销量字段。

## 配置与运行

当前结果镜像位于 `configs/current_result_mirror/`，绑定20秒实验参数、输入哈希、代码哈希和参考结果哈希。

```powershell
.\run_problem.ps1 -ConfigPath q1_test\configs\current_result_mirror\config.yaml
```

## 当前结果

`doc/results/q1/audit.csv` 是冻结审计来源；镜像已真实复现相同利润和Excel回读结果到 `doc/results/q1/reproduced/`。旧 `outputs/q1/repro.json` 属600秒历史实验，不再作为当前镜像依据。

| 情形 | incumbent | 正向理论上界 | 修正 gap | 状态 |
|---|---:|---:|---:|---|
| 滞销浪费 | 34,229,493.25 | 41,156,748.42 | 20.24% | 可行，未认证 |
| 超额半价 | 42,570,196.63 | 63,924,760.06 | 50.16% | 可行，未认证 |

gap 必须来自 primary 利润阶段；字典序激活数 gap 不是利润 gap。

## 产物

`outputs/q1/result1_1.xlsx`、`result1_2.xlsx`、`audit.csv`、`yearly_stats.csv`、`sensitivity_*.csv`和 `figures/`。统一对照见 `../doc/三问实验基准与对照.csv`。

## 工程结构

```text
q1_test/
├─ algorithms/
│  ├─ io_data.py, preprocess.py       # 附件读取、清洗和编码
│  ├─ model.py, solve.py              # 两种情形MILP与字典序求解
│  ├─ validate.py                     # 硬约束和利润独立复算
│  ├─ export_excel.py                 # result1_1/result1_2回填与回读
│  └─ plots.py                        # 原始/过程/结果图
├─ scripts/
│  ├─ run_q1.py                       # 正式主入口
│  ├─ p1_test.py, pipeline_test.py    # 门禁与流水线测试
│  └─ explore_*.py, build_test.py      # 数据探查/构建测试
├─ configs/current_result_mirror/     # 当前结果独立Config、manifest
├─ outputs/data_cleaning/             # 清洗后可追溯数据
├─ outputs/q1/                        # 方案、审计、敏感性、图表和日志
├─ doc/
└─ run.ps1                            # 转发到全局Config启动器
```

## 产物说明

| 文件 | 含义 | 交付判定 |
|---|---|---|
| `result1_1.xlsx` | 情形1的7年种植面积 | 候选，最优性未认证 |
| `result1_2.xlsx` | 情形2的7年种植面积 | 候选，最优性未认证 |
| `audit.csv` | 两情形的可行性、incumbent、理论界和gap | 当前最新数值来源 |
| `yearly_stats.csv` | 年收入、成本、利润、产量、销量和剩余 | 论文统计表来源 |
| `sensitivity_*.csv` | `eta/delta/demand_scale`扫描 | 稳定性论证 |
| `repro.json` | 参数、环境、哈希和求解统计 | 当前与audit版本不一致，待重跑冻结 |
