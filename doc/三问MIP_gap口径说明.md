# 三问 MIP gap 口径说明

对最大化阶段，SciPy/HiGHS 内部求解的是负目标最小化。统一换算为：

\[
LB=-f_{inc},\qquad UB=-f_{dual},\qquad
gap=\frac{\max(0,UB-LB)}{\max(|LB|,10^{-9})}.
\]

第三级“最少激活数”是最小化目标，其 gap 不得冒充利润 gap。

- Q1 应报告 primary 利润阶段 gap。
- Q2 应报告 Stage 1 风险目标 gap，不得使用 Stage 3 激活数的 `dual_bound=544`。
- Q3 全整数发现阶段 gap 为 `31.4722%`。
- Q3 `K=30` 精炼阶段 gap 为 `0`，但只表示固定当前 `y/r/b` 结构后的条件最优，不得表述为全整数全局最优。
