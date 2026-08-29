# -*- coding: utf-8 -*-
"""Q3 算法包 — 相关性、替代性与互补性下的随机MILP。

模块职责：
  paths             — 路径配置与哈希校验
  io_data           — 只读附件解析 + Q2基线读取
  preprocess        — 数据清洗 + ModelData + 作物类别映射
  dependency        — 因子t-Copula七年相关重排
  elasticity        — 交叉价格弹性矩阵
  scenarios         — Q3边际LHS → 相关重排 → 弹性修正
  scenario_reduction — 分层PAM缩减 + Kendall审计
  model             — 含b/w豆类前茬互补的Q3随机MILP
  solve             — 求解器探测 + 检查点 + 三级字典序
  risk              — CVaR复算 + 风险前沿资格 + 唯一方案选择
  evaluate          — Q2/Q3配对比较 + 四组消融
  validate          — 约束审计 + 相关/弹性/互补扩展检查
  export_ooxml      — result3.xlsx结构保持式原子导出
  plots             — 9张逻辑图（原始3+过程3+结果3）
"""
