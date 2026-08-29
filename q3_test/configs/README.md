# Q3 配置说明

## 当前推荐

当前结果统一使用 `current_result_mirror/config.yaml`。它显式绑定冻结的Q2基线和Q3发现结构，不依赖 `outputs/q3/selected_plan_q3.csv`。

```powershell
.\run_problem.ps1 -ConfigPath q3_test\configs\current_result_mirror\config.yaml
```

- `q3_default.yaml`：完整正式前沿，耗时最长。
- `q3_quality_candidate.yaml`：固定 Q2 启用模式的快速连续优化，仅供候选对照。
- `q3_paper_candidate.yaml`：释放整数结构、小 K 搜索新种植组合，再做大样本评估。
- `q3_paper_refine.yaml`：固定上述新结构，用 `K=30`、3个 lambda 精炼风险与连续面积。

其余YAML保留为历史实验配置，不作为当前结果复现入口。

`q3_paper_refine.yaml` 中的 gap 是固定整数结构后的条件 gap；全整数发现 gap 另见 `outputs/q3/gap_summary_q3.csv`。
