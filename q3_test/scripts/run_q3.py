# -*- coding: utf-8 -*-
"""Q3主入口 — 相关性、替代性与互补性下的随机MILP求解。

流水线:
  1. 配置校验 + 输入哈希校验
  2. 数据清洗 + Q2基线读取
  3. Q3情景生成 (LHS → t-Copula → 弹性)
  4. 情景缩减 (PAM k-medoids + Kendall审计)
  5. 风险前沿 (11个lambda三级字典序)
  6. 唯一方案选择 (膝点规则)
  7. 样本外评估 + Q2/Q3配对比较
  8. 四组消融
  9. 约束审计 + Excel候选 + 图表 + 报告 + 复现清单

退出码:
  0 = 完成，全部硬门槛通过
  1 = 输入/依赖/模型/审计/导出错误
  2 = 有可行方案但前沿/认证未达标
  3 = 无可行方案

用法:
  python scripts/run_q3.py --config configs/q3_default.yaml --figures --reports
  python scripts/run_q3.py --raw-scenarios 200 --reduced-scenarios 5 --time-limit 30 --allow-uncertified

作者: Q3编程手
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import hashlib
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

# 确保algorithms包可导入
Q3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Q3_ROOT))

from algorithms import paths
from algorithms.paths import ensure_dirs
from algorithms.io_data import load_inputs, load_q2_baseline, verify_input_hashes
from algorithms.preprocess import preprocess, ModelData
from algorithms.scenarios import generate_raw_scenarios
from algorithms.scenario_reduction import reduce_scenarios
from algorithms.solve import solve_lexicographic, detect_solver, extract_solution
from algorithms.risk import (recompute_scenario_profits, compute_cvar,
                             select_unique_plan, check_frontier_complete)
from algorithms.dependency import DependencyConfig, BASE_LOADINGS
from algorithms.elasticity import ElasticityConfig, build_elasticity_matrix, audit_elasticity
from algorithms.validate import validate_solution
from algorithms.export_ooxml import patch_result3


def _print_progress(pct: int, msg: str) -> None:
    """打印进度条。"""
    bar = "[" + "#" * (pct // 3) + "." * ((100 - pct) // 3) + "]"
    print(f"\r{bar:34s} {pct:3d}%  {msg}", flush=True)


def _config_hash(config: dict) -> str:
    """计算配置哈希（用于检查点验证）。"""
    s = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _load_config(config_path: Path) -> dict:
    """加载YAML配置。"""
    try:
        import yaml
    except ImportError:
        return {}
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="2024 C题 问题3 求解主程序")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--seed", type=int, default=2024, help="随机种子")
    parser.add_argument("--raw-scenarios", type=int, default=2000,
                        help="原始情景数")
    parser.add_argument("--reduced-scenarios", type=int, default=30,
                        help="缩减情景数")
    parser.add_argument("--out-sample", type=int, default=5000,
                        help="样本外评估情景数")
    parser.add_argument("--beta", type=float, default=0.90, help="CVaR置信水平")
    parser.add_argument("--lambda-grid", type=str, default="0:1:0.1",
                        help="lambda网格 start:stop:step")
    parser.add_argument("--eta", type=float, default=0.5, help="最小面积比例")
    parser.add_argument("--gamma", type=float, default=0.03, help="豆类前茬增产率")
    parser.add_argument("--time-limit", type=int, default=600, help="求解时间限制(秒)")
    parser.add_argument("--mip-gap", type=float, default=0.001, help="MIP间隙")
    parser.add_argument("--eps-z", type=float, default=None, help="字典序Z容差")
    parser.add_argument("--eps-e", type=float, default=None, help="字典序E容差")
    parser.add_argument("--figures", action="store_true", help="生成图表")
    parser.add_argument("--reports", action="store_true", help="生成报告")
    parser.add_argument("--allow-uncertified", action="store_true",
                        help="允许未认证结果(P1/冒烟用)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="禁用检查点")
    args = parser.parse_args()

    ensure_dirs()

    # ---- 加载配置 ----
    config = {}
    if args.config:
        config = _load_config(Path(args.config))
    argv = set(sys.argv[1:])
    def choose(flag, cli_value, cfg_value):
        return cli_value if flag in argv else cfg_value
    seed = choose("--seed", args.seed, config.get("seed", args.seed))
    raw_n = choose("--raw-scenarios", args.raw_scenarios,
                   config.get("raw_scenarios", args.raw_scenarios))
    K = choose("--reduced-scenarios", args.reduced_scenarios,
               config.get("reduced_scenarios", args.reduced_scenarios))
    out_sample = choose("--out-sample", args.out_sample,
                        config.get("out_sample", args.out_sample))
    beta = choose("--beta", args.beta, config.get("beta", args.beta))
    eta = choose("--eta", args.eta, config.get("eta", args.eta))
    gamma = choose("--gamma", args.gamma,
                   config.get("complementarity", {}).get("gamma", args.gamma))
    time_limit = choose("--time-limit", args.time_limit,
                        config.get("solver", {}).get("time_limit", args.time_limit))
    mip_gap = choose("--mip-gap", args.mip_gap,
                     config.get("solver", {}).get("mip_gap", args.mip_gap))

    # lambda网格
    if "lambda_grid" in config and "--lambda-grid" not in argv:
        lambdas = [float(v) for v in config["lambda_grid"]]
    else:
        parts = args.lambda_grid.split(":")
        lam_start, lam_stop, lam_step = map(float, parts)
        lambdas = [round(lam_start + i * lam_step, 4)
                   for i in range(int((lam_stop - lam_start) / lam_step + 0.5) + 1)]
    distribution = config.get("distribution", "uniform")
    dep_raw = config.get("dependency", {})
    dep_cfg = DependencyConfig(
        df=float(dep_raw.get("df", 5)),
        correlation_scale=float(dep_raw.get("correlation_scale", 1.0)),
        temporal_rho=float(dep_raw.get("temporal_rho", 0.5)),
        loadings=BASE_LOADINGS,
    )
    ela_cfg = ElasticityConfig(scale=float(config.get("elasticity", {}).get("scale", 1.0)))

    config_hash = _config_hash({
        "seed": seed, "raw_n": raw_n, "K": K, "beta": beta,
        "eta": eta, "gamma": gamma, "lambdas": lambdas,
        "time_limit": time_limit, "mip_gap": mip_gap,
    })

    print("=" * 60)
    print("Q3 随机MILP求解 — 相关性·替代性·互补性")
    print("=" * 60)
    print(f"  原始情景: {raw_n}, 缩减: {K}, 样本外: {out_sample}")
    print(f"  beta={beta}, eta={eta}, gamma={gamma}")
    print(f"  lambda网格: {lambdas}")
    print(f"  时间限制: {time_limit}s, MIP gap: {mip_gap}")
    print(f"  配置哈希: {config_hash}")
    print()

    try:
        # ---- 1. 输入校验 ----
        _print_progress(5, "输入哈希校验")
        verify_input_hashes(paths.EXPECTED_SHA)

        # ---- 2. 数据加载 ----
        _print_progress(10, "数据清洗")
        raw = load_inputs()
        data = preprocess(raw)

        # ---- 3. Q2基线 ----
        _print_progress(12, "Q2基线读取")
        baseline = load_q2_baseline(paths.Q2_SELECTED_PLAN)
        plan_x_q2 = baseline.area

        # ---- 4. 求解器探测 ----
        _print_progress(15, "求解器探测")
        caps = detect_solver()
        print(f"  求解器: {caps.backend}")
        print(f"  MIP start: {caps.supports_mip_start}")
        print(f"  对偶界: {caps.supports_bound}")

        # ---- 5. 情景生成 ----
        _print_progress(20, f"Q3情景生成 (N={raw_n})")
        scenarios = generate_raw_scenarios(
            data, n=raw_n, seed=seed, distribution=distribution,
            dependency_cfg=dep_cfg, elasticity_cfg=ela_cfg)
        elasticity_audit = audit_elasticity(build_elasticity_matrix(data, config=ela_cfg), data)

        # ---- 6. 情景缩减 ----
        _print_progress(30, f"PAM缩减 (K={K})")
        reduced, red_audit = reduce_scenarios(data, scenarios, k=K,
                                               baseline_plan=plan_x_q2,
                                               gamma=gamma, seed=seed)
        print(f"  权重和: {red_audit.sum_weights:.6f}")
        print(f"  最小权重: {red_audit.min_weight:.6f}")
        print(f"  尾部代表数: {red_audit.min_profit_layer_reps}")
        print(f"  原始Kendall最大误差: {scenarios.dependency_audit.max_kendall_error:.4f}")
        print(f"  缩减Kendall最大误差: {red_audit.max_kendall_error:.4f}")
        if (scenarios.dependency_audit.max_kendall_error > 0.05
                or red_audit.max_kendall_error > 0.15
                or not red_audit.kendall_direction_consistent):
            raise RuntimeError("相关情景原始/缩减 Kendall 门禁未通过")

        # ---- 7. 风险前沿 ----
        _print_progress(35, "风险前沿求解")
        frontier_points = []
        ckpt_dir = None if args.no_checkpoint else paths.CKPT_DIR

        for li, lam in enumerate(lambdas):
            pct = 35 + int(40 * li / len(lambdas))
            _print_progress(pct, f"lambda={lam:.1f} 三级字典序求解")
            result = solve_lexicographic(
                data, reduced, beta=beta, risk_lambda=lam,
                eta=eta, gamma=gamma, time_limit=time_limit,
                mip_gap=mip_gap, eps_z=args.eps_z, eps_e=args.eps_e,
                ckpt_dir=ckpt_dir, config_hash=config_hash,
            )
            z_star = result.get("z_star")
            e_star = result.get("e_star")
            n_act = result.get("n_activations")
            final_result = result.get("result")
            solver_status = getattr(final_result, "solver_status", "unknown") if final_result else "unknown"
            mip_gap_val = getattr(final_result, "mip_gap", float("nan")) if final_result else float("nan")
            lex_complete = result.get("lex_complete", False)

            if z_star is not None:
                # 复算利润和CVaR
                sol = result.get("solution")
                if sol:
                    profits = recompute_scenario_profits(sol, data, reduced, gamma)
                    weights = reduced.weights
                    cvar = compute_cvar(profits, weights, beta)
                    expected_profit = float(np.dot(weights, profits))
                else:
                    profits = np.array([])
                    cvar = float("nan")
                    expected_profit = float("nan")

                frontier_points.append({
                    "lambda": lam,
                    "z_lambda": z_star,
                    "expected_profit": expected_profit,
                    "cvar": cvar,
                    "n_activations": n_act,
                    "solver_status": solver_status,
                    "status": solver_status,
                    "mip_gap": mip_gap_val,
                    "lex_complete": lex_complete,
                    "result": final_result,
                    "solution": sol,
                })
                print(f"  lambda={lam:.1f}: Z={z_star:,.0f}, E[Pi]={expected_profit:,.0f}, "
                      f"CVaR={cvar:,.0f}, gap={mip_gap_val:.4f}, {solver_status}")
            else:
                frontier_points.append({
                    "lambda": lam,
                    "z_lambda": None,
                    "expected_profit": None,
                    "cvar": None,
                    "n_activations": None,
                    "solver_status": solver_status,
                    "status": solver_status,
                    "mip_gap": float("nan"),
                    "lex_complete": False,
                    "result": final_result,
                    "solution": None,
                })
                print(f"  lambda={lam:.1f}: {solver_status}")

        # ---- 8. 方案选择 ----
        _print_progress(78, "唯一方案选择")
        selected = select_unique_plan(frontier_points)
        if selected is None:
            print("  风险前沿无具备资格的点（需有限可行解且三级字典序完成）")
            return 3
        sel_lam = selected.get("selected_lambda")
        print(f"  选中 lambda: {sel_lam}")

        sel_point = None
        for p in frontier_points:
            if p.get("lambda") == sel_lam:
                sel_point = p
                break

        if sel_point is None or sel_point.get("solution") is None:
            print("  无可行方案!")
            return 3

        # ---- 9. 样本外评估 ----
        _print_progress(82, f"样本外评估 (N={out_sample})")
        oos_scenarios = generate_raw_scenarios(
            data, n=out_sample, seed=seed + 999, distribution=distribution,
            dependency_cfg=dep_cfg, elasticity_cfg=ela_cfg)
        q3_profits = recompute_scenario_profits(
            sel_point["solution"], data, oos_scenarios, gamma)
        q2_profits = recompute_scenario_profits(
            {"x": plan_x_q2, "y": {}, "r": {}, "b": {}, "w": {}},
            data, oos_scenarios, gamma)
        sel_point["solution"]["paired_sample_count"] = int(out_sample)
        print(f"  Q3样本外均值: {np.mean(q3_profits):,.0f}")
        print(f"  Q2样本外均值: {np.mean(q2_profits):,.0f}")
        print(f"  差异: {np.mean(q3_profits) - np.mean(q2_profits):,.0f}")

        # ---- 10. 约束审计 ----
        _print_progress(88, "约束审计")
        frontier_complete = check_frontier_complete(frontier_points, tuple(lambdas))
        audit = validate_solution(
            sel_point["solution"], sel_point["result"].model if sel_point.get("result") else None,
            data, reduced, beta=beta, gamma=gamma,
            frontier_complete=frontier_complete,
            selected_certified=bool(sel_point.get("lex_complete")),
            elasticity_audit=elasticity_audit,
            dependency_audit=scenarios.dependency_audit,
        )
        print(f"  feasible={audit.get('feasible', False)}")
        print(f"  max_violation={audit.get('max_violation', 0):.2e}")
        print(f"  certified={audit.get('certified', False)}")

        # ---- 11. Excel候选 ----
        _print_progress(92, "Excel候选生成")
        try:
            readback_diff = patch_result3(
                sel_point["solution"], data,
                paths.TEMPLATE2_PATH, paths.RESULT3_CANDIDATE)
            print(f"  回读差: {readback_diff:.2e}")
        except Exception as e:
            print(f"  Excel导出失败: {e}")
            readback_diff = float("inf")

        # 可机读核心产物（不依赖图表/报告开关）
        pd.DataFrame([{k: v for k, v in p.items()
                       if k not in ("result", "solution")} for p in frontier_points]).to_csv(
            paths.RISK_FRONTIER_Q3, index=False, encoding="utf-8-sig")
        pd.DataFrame({"scenario": np.arange(out_sample),
                      "q2_profit": q2_profits,
                      "q3_profit": q3_profits,
                      "difference": q3_profits - q2_profits}).to_csv(
            paths.PAIRED_PROFITS_Q2_Q3, index=False, encoding="utf-8-sig")
        pd.DataFrame([audit]).to_csv(paths.AUDIT_Q3, index=False, encoding="utf-8-sig")

        # ---- 12. 图表 ----
        if args.figures:
            _print_progress(95, "生成图表")
            try:
                from algorithms.plots import generate_figures
                generate_figures(
                    data, scenarios=scenarios, reduced=reduced,
                    frontier_points=frontier_points,
                    selected_plan=sel_point["solution"],
                    q2_plan={"x": plan_x_q2},
                    q3_profits=q3_profits, q2_profits=q2_profits,
                    out_dir=paths.FIG_DIR,
                )
            except Exception as e:
                print(f"  图表生成失败: {e}")

        # ---- 13. 复现清单 ----
        _print_progress(98, "复现清单")
        repro = {
            "seed": seed,
            "raw_scenarios": raw_n,
            "reduced_scenarios": K,
            "out_sample": out_sample,
            "beta": beta,
            "eta": eta,
            "gamma": gamma,
            "lambdas": lambdas,
            "selected_lambda": sel_lam,
            "time_limit": time_limit,
            "mip_gap": mip_gap,
            "config_hash": config_hash,
            "solver_backend": caps.backend,
            "supports_mip_start": caps.supports_mip_start,
            "frontier_complete": frontier_complete,
            "selected_certified": audit.get("certified", False),
            "q3_mean_profit": float(np.mean(q3_profits)),
            "q2_mean_profit": float(np.mean(q2_profits)),
            "timestamp": time.time(),
        }
        with open(paths.REPRO_Q3, "w", encoding="utf-8") as f:
            json.dump(repro, f, ensure_ascii=False, indent=2)

        # ---- 14. 退出码 ----
        _print_progress(100, "完成")
        if audit.get("certified", False):
            return 0
        elif audit.get("feasible", False):
            return 2
        else:
            return 3

    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
