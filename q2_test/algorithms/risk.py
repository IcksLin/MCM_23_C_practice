# -*- coding: utf-8 -*-
"""CVaR recomputation and risk-frontier knee-point selection (AGENT.md section 10).

Functions:
  recompute_scenario_profits — independent profit recomputation for a fixed plan
  compute_cvar — compute LCVaR_beta from scenario profits
  select_unique_plan — deterministic knee-point selection on the risk frontier
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .preprocess import ModelData
from .scenario_reduction import ReducedScenarioSet


def recompute_scenario_profits(plan_x: dict, scenarios, data: ModelData) -> np.ndarray:
    """Recompute per-scenario profit independently from solver values.

    plan_x: dict (j,i,t,s) -> area (mu), common across scenarios.
    scenarios: ReducedScenarioSet or ScenarioSet.

    Returns: (K,) array of scenario profits.
    """
    K = scenarios.k if hasattr(scenarios, "k") else scenarios.n
    profits = np.zeros(K)
    for omega in range(K):
        # production by (i,t,s)
        Q = {}
        for (j, i, t, s), area in plan_x.items():
            if area <= 0:
                continue
            q_omega = scenarios.yield_.get((j, i, t, s), np.zeros(K))[omega]
            Q[(i, t, s)] = Q.get((i, t, s), 0.0) + q_omega * area
        # revenue: u = min(Q, D)
        revenue = 0.0
        for (i, t, s), q_tot in Q.items():
            d = scenarios.demand.get((i, t, s), np.zeros(K))[omega]
            u = min(q_tot, d)
            p = scenarios.price.get((i, t, s), np.zeros(K))[omega]
            revenue += p * u
        # cost
        cost = 0.0
        for (j, i, t, s), area in plan_x.items():
            if area <= 0:
                continue
            c = scenarios.cost.get((j, i, t, s), np.zeros(K))[omega]
            cost += c * area
        profits[omega] = revenue - cost
    return profits


def compute_cvar(profits: np.ndarray, weights: np.ndarray,
                 beta: float = 0.90) -> float:
    """Compute LCVaR_beta (lower-tail CVaR) from scenario profits.

    LCVaR_beta = E[Pi | Pi <= VaR_beta]
    where VaR_beta is the beta-quantile of the profit distribution.

    For discrete scenarios with weights:
    1. Sort by profit ascending.
    2. Find the point where cumulative weight >= 1-beta.
    3. CVaR = weighted average of profits below/at the VaR threshold.
    """
    K = len(profits)
    if K == 0:
        return 0.0
    # sort by profit ascending
    order = np.argsort(profits)
    sorted_p = profits[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    # find VaR: the profit level where cumulative weight just reaches 1-beta
    tail_mass = 1.0 - beta
    # find the index where cumulative weight >= tail_mass
    idx = np.searchsorted(cum_w, tail_mass)
    if idx >= K:
        idx = K - 1
    # CVaR = weighted average of the tail
    # weight of the partial point at the boundary
    if cum_w[idx] > tail_mass and idx > 0:
        partial = tail_mass - cum_w[idx - 1]
        tail_weights = sorted_w[:idx].copy()
        tail_weights = np.append(tail_weights, partial)
        tail_profits = sorted_p[:idx + 1]
    else:
        tail_weights = sorted_w[:idx + 1]
        tail_profits = sorted_p[:idx + 1]
    total_tail = tail_weights.sum()
    if total_tail > 0:
        cvar = float(np.average(tail_profits, weights=tail_weights))
    else:
        cvar = float(sorted_p[0]) if K > 0 else 0.0
    return cvar


def pareto_nondominated(frontier: pd.DataFrame, tol: float = 1e-8) -> pd.DataFrame:
    """Return finite points not weakly dominated in both profit objectives."""
    required = ["lambda", "expected_profit", "lower_tail_cvar"]
    missing = [c for c in required if c not in frontier.columns]
    if missing:
        raise ValueError(f"frontier missing columns: {missing}")
    df = frontier.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    keep = []
    for idx, row in df.iterrows():
        other = df.drop(index=idx)
        dominated = (
            (other["expected_profit"] >= row["expected_profit"] - tol)
            & (other["lower_tail_cvar"] >= row["lower_tail_cvar"] - tol)
            & ((other["expected_profit"] > row["expected_profit"] + tol)
               | (other["lower_tail_cvar"] > row["lower_tail_cvar"] + tol))
        ).any()
        keep.append(not dominated)
    return df.loc[keep].sort_values("lambda").reset_index(drop=True)


def select_unique_plan(frontier: pd.DataFrame) -> str:
    """Deterministic knee-point selection on the risk frontier.

    frontier columns: lambda, expected_profit, lower_tail_cvar, n_activations

    Selection rules (AGENT.md section 10):
    1. Normalize expected_profit and lower_tail_cvar to [0,1].
    2. For each non-endpoint, compute perpendicular distance to the line
       connecting the two endpoints.
    3. If max distance >= 0.02: select the point with max distance.
       Ties (within 1e-9): prefer larger lambda, then higher expected_profit,
       then fewer activations.
    4. If max distance < 0.02: select the smallest lambda whose lower_tail_cvar
       reaches 99% of the frontier max. Ties: higher expected_profit, fewer
       activations.

    Returns: the lambda value as a string (for dict key lookup).
    """
    df = pareto_nondominated(frontier)
    n = len(df)
    if n == 0:
        return None
    if n == 1:
        return str(df.loc[0, "lambda"])

    # normalize to [0,1]
    ep = df["expected_profit"].values
    cv = df["lower_tail_cvar"].values
    ep_n = (ep - ep.min()) / (ep.max() - ep.min() + 1e-15)
    cv_n = (cv - cv.min()) / (cv.max() - cv.min() + 1e-15)

    # endpoints
    p0 = np.array([ep_n[0], cv_n[0]])
    p1 = np.array([ep_n[-1], cv_n[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-15:
        return str(df.loc[0, "lambda"])

    # perpendicular distances for non-endpoints
    distances = np.full(n, -1.0)
    for i in range(1, n - 1):
        pi = np.array([ep_n[i], cv_n[i]])
        # perpendicular distance to line p0-p1
        d = abs(np.cross(line_vec, pi - p0)) / line_len
        distances[i] = d

    max_dist = distances.max()
    max_idx = int(np.argmax(distances))

    if max_dist >= 0.02:
        # knee point selection
        # check ties within 1e-9
        tied = np.where(distances >= max_dist - 1e-9)[0]
        if len(tied) > 1:
            # prefer larger lambda, then higher expected_profit, then fewer activations
            best = tied[np.lexsort((
                df.loc[tied, "n_activations"].values,
                -df.loc[tied, "expected_profit"].values,
                -df.loc[tied, "lambda"].values,
            ))[0]]
        else:
            best = max_idx
        return str(df.loc[best, "lambda"])
    else:
        # no clear knee: select smallest lambda with cvar >= 99% of max
        cvar_max = df["lower_tail_cvar"].max()
        threshold = 0.99 * cvar_max
        candidates = df[df["lower_tail_cvar"] >= threshold]
        if len(candidates) == 0:
            candidates = df[df["lower_tail_cvar"] == df["lower_tail_cvar"].max()]
        # smallest lambda, then higher expected_profit, then fewer activations
        best_row = candidates.sort_values(
            ["lambda", "expected_profit", "n_activations"],
            ascending=[True, False, True]
        ).iloc[0]
        return str(best_row["lambda"])
