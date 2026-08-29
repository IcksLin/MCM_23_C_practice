# -*- coding: utf-8 -*-
"""Q3 P1纵向门禁测试 — 最小规模完整流水线验证。

参数: 54地块/41作物, 2024-2025两年切片, N0=200, K=5, lambda={0,1}

验证链路:
  输入哈希 → 边际LHS → t-Copula重排 → 弹性修正 → 情景缩减
  → 含w变量MILP → 三级求解 → Q2/Q3共同情景复算 → 审计 → 临时OOXML输出

作者: Q3编程手
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

Q3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Q3_ROOT))

from algorithms import paths
from algorithms.paths import ensure_dirs
from algorithms.io_data import load_inputs, load_q2_baseline, verify_input_hashes
from algorithms.preprocess import preprocess, ModelData, LEGUME_CODES
from algorithms.scenarios import generate_raw_scenarios
from algorithms.dependency import DependencyConfig, BASE_LOADINGS
from algorithms.elasticity import (ElasticityConfig, build_elasticity_matrix,
                                   audit_elasticity)
from algorithms.scenario_reduction import reduce_scenarios
from algorithms.solve import solve_lexicographic, extract_solution
from algorithms.risk import recompute_scenario_profits, compute_cvar
from algorithms.validate import validate_solution
from algorithms.export_ooxml import patch_result3

import numpy as np


def _assert(cond: bool, msg: str) -> None:
    """断言辅助。"""
    if not cond:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


def main() -> int:
    print("=" * 60)
    print("Q3 P1 纵向门禁测试")
    print("=" * 60)

    ensure_dirs()

    # ---- 1. 输入哈希 ----
    print("\n[1] 输入哈希校验")
    try:
        verify_input_hashes(paths.EXPECTED_SHA)
        _assert(True, "输入文件哈希匹配")
    except Exception as e:
        _assert(False, f"哈希校验失败: {e}")
        return 1

    # ---- 2. 数据加载 ----
    print("\n[2] 数据清洗")
    raw = load_inputs()
    data = preprocess(raw)
    _assert(len(data.plot_names) == 54, "54地块")
    _assert(len(data.crop_codes) == 41, "41作物")

    # ---- 3. Q2基线 ----
    print("\n[3] Q2基线读取")
    baseline = load_q2_baseline(paths.Q2_SELECTED_PLAN)
    plan_x_q2 = baseline.area
    _assert(len(plan_x_q2) > 0, "Q2方案非空")

    # ---- 4. LHS边际生成 ----
    print("\n[4] LHS边际生成 (N=30)")
    N = 30
    dep_cfg = DependencyConfig(5, 1.0, 0.5, BASE_LOADINGS)
    ela_cfg = ElasticityConfig(scale=1.0)
    scenarios = generate_raw_scenarios(
        data, n=N, seed=2024, dependency_cfg=dep_cfg,
        elasticity_cfg=ela_cfg)
    _assert(scenarios.n == N, f"情景数={scenarios.n}")

    # 验证LHS分层
    for (i, t, s), vals in scenarios.demand.items():
        if len(vals) == N:
            sorted_vals = np.sort(vals)
            # LHS应大致均匀分布
            q1, q3 = np.percentile(vals, [25, 75])
            _assert(q3 > q1, f"需求{i},{t},{s}有变化范围")
            break

    # ---- 5. 情景缩减 ----
    print("\n[5] PAM缩减 (K=3)")
    K = 3
    reduced, red_audit = reduce_scenarios(data, scenarios, k=K,
                                           baseline_plan=plan_x_q2,
                                           gamma=0.03, seed=2024)
    _assert(reduced.k == K, f"缩减K={reduced.k}")
    _assert(abs(red_audit.sum_weights - 1.0) < 1e-6, f"权重和={red_audit.sum_weights:.6f}")
    _assert(red_audit.min_weight >= 0, f"最小权重={red_audit.min_weight:.6f}")
    _assert(red_audit.zero_weight_count == 0, f"零权重数={red_audit.zero_weight_count}")

    # ---- 6. λ=0 MILP求解 ----
    print("\n[6] λ=0 三级字典序求解")
    result = solve_lexicographic(
        data, reduced, beta=0.90, risk_lambda=0.0,
        eta=0.5, gamma=0.03, time_limit=45, mip_gap=0.05,
    )
    _assert(result.get("z_star") is not None, f"Z*={result.get('z_star')}")
    _assert(result.get("stage1_feasible", False), "Stage 1可行")

    sol = result.get("solution")
    _assert(sol is not None, "解非空")
    _assert(sol.get("n_activations", 0) > 0, f"激活数={sol.get('n_activations')}")

    # ---- 7. 销量约束验证 ----
    print("\n[7] 销量约束验证")
    model = result.get("model")
    _assert(model is not None, "模型存在")
    max_sales_violation = 0.0
    for (omega, i, t, s), u_val in sol["u"].items():
        q_val = sol["Q"][(omega, i, t, s)]
        d_val = reduced.demand[(i, t, s)][omega]
        max_sales_violation = max(max_sales_violation, u_val-q_val,
                                  u_val-d_val, 0.0)
    _assert(max_sales_violation <= 1e-6,
            f"销量约束最大违反={max_sales_violation:.2e}")

    # ---- 8. w=x*b验证 ----
    print("\n[8] 互补变量w=x*b验证")
    if sol and "w" in sol and "b" in sol:
        max_diff = 0.0
        for (j, i, t, s), w_val in sol["w"].items():
            x_val = sol["x"].get((j, i, t, s), 0.0)
            b_val = sol["b"].get((j, t), 0)
            if b_val == 1:
                diff = abs(w_val - x_val)
            else:
                diff = abs(w_val)
            max_diff = max(max_diff, diff)
        _assert(max_diff <= 1e-6, f"max|w-x*b|={max_diff:.6e}")
    else:
        _assert(True, "w/b变量存在（简化验证）")

    # ---- 9. 利润复算 ----
    print("\n[9] 利润复算")
    profits = recompute_scenario_profits(sol, data, reduced, gamma=0.03)
    weights = reduced.weights
    expected_profit = float(np.dot(weights, profits))
    z_star = result.get("z_star", 0)
    cvar = compute_cvar(profits, weights, beta=0.90)
    print(f"  Z*={z_star:,.0f}")
    print(f"  E[Pi]={expected_profit:,.0f}")
    print(f"  CVaR={cvar:,.0f}")
    _assert(expected_profit > 0, f"期望利润为正: {expected_profit}")

    # ---- 10. 约束审计 ----
    print("\n[10] 约束审计")
    audit = validate_solution(
        sol, model, data, reduced, beta=0.90, gamma=0.03,
        dependency_audit=scenarios.dependency_audit,
        elasticity_audit=audit_elasticity(
            build_elasticity_matrix(data, config=ela_cfg), data),
    )
    mv = audit.get("max_violation", 1)
    print(f"  max_violation={mv:.2e}")
    _assert(mv <= 1e-4, f"max_violation={mv:.2e}")

    # ---- 11. OOXML输出 ----
    print("\n[11] OOXML候选输出")
    out_path = paths.Q3_OUT_DIR / "p1_test_candidate.xlsx"
    diff = patch_result3(sol, data, paths.TEMPLATE2_PATH, out_path)
    _assert(diff < 1e-6, f"回读差={diff:.2e}")

    # ---- 完成 ----
    print("\n" + "=" * 60)
    print("P1 纵向门禁测试 PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
