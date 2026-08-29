# 问题3当前结果镜像

复现命令：

```powershell
.\run_problem.ps1 -ConfigPath q3_test\configs\current_result_mirror\config.yaml
```

Q2基线和Q3发现阶段结构均冻结在 `frozen_inputs/`，精炼不再依赖普通输出目录。输出写入 `doc/results/q3/reproduced/`。K=30仅用于条件精炼，正式风险统计采用未缩减OOS。
