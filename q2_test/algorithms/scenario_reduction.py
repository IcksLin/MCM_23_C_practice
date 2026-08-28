# -*- coding: utf-8 -*-
"""Scenario reduction via PAM k-medoids with tail protection (AGENT.md section 9).

Pipeline:
  1. Fix Q1 result1_1 plan, compute proxy profit for all N raw scenarios.
  2. Stratify scenarios into 10 equal-frequency profit layers.
  3. Bottom layer gets at least ceil(0.1*K) representatives.
  4. PAM k-medoids with L1 distance on standardized rate-of-change vectors.
  5. Representatives are actual samples (not centroids).
  6. Weights = cluster_size / N, sum to 1.

Usage:
    from algorithms.scenario_reduction import reduce_scenarios
    reduced = reduce_scenarios(data, raw_scenarios, k=30, baseline_plan=plan)
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.distance import cdist

from .preprocess import ModelData
from .scenarios import ScenarioSet, compute_proxy_profit


@dataclass
class ReducedScenarioSet:
    """Reduced scenario set with K representative scenarios."""
    indices: np.ndarray       # (K,) indices into original ScenarioSet arrays
    weights: np.ndarray       # (K,) probability weights, sum to 1
    demand: dict              # (i,t,s) -> array(K,)
    yield_: dict              # (j,i,t,s) -> array(K,)
    cost: dict                # (j,i,t,s) -> array(K,)
    price: dict               # (i,t,s) -> array(K,)
    k: int
    n_original: int
    proxy_profits: np.ndarray  # (K,) proxy profit per representative


def _compute_all_proxy_profits(scenarios: ScenarioSet, data: ModelData,
                                plan_x: dict) -> np.ndarray:
    """Compute proxy profit for all N scenarios with a fixed plan."""
    n = scenarios.n
    profits = np.zeros(n)
    # vectorized: pre-compute production Q per (i,t,s) across all scenarios
    for scenario_idx in range(n):
        profits[scenario_idx] = compute_proxy_profit(
            scenarios, data, plan_x, scenario_idx)
    return profits


def _pam_kmedoids(distances: np.ndarray, k: int, seed: int = 2024) -> tuple:
    """Memory-safe PAM-style k-medoids on a precomputed distance matrix.

    Args:
      distances: (N, N) pairwise distance matrix.
      k: number of medoids.
      seed: for reproducibility.

    Returns:
      (medoid_indices, assignments) where:
        medoid_indices: (k,) indices of selected medoids
        assignments: (N,) cluster assignment for each point
    """
    n = distances.shape[0]
    rng = np.random.default_rng(seed)

    if k >= n:
        return np.arange(n), np.arange(n)

    # BUILD initialization: start from the most central point, then add the
    # point farthest from its nearest existing medoid.
    medoids = [int(np.argmin(distances.sum(axis=1)))]
    for _ in range(1, k):
        # choose point farthest from nearest medoid
        min_dist_to_medoids = distances[:, medoids].min(axis=1)
        next_medoid = int(np.argmax(min_dist_to_medoids))
        if next_medoid in medoids:
            # fallback: random non-medoid
            candidates = [i for i in range(n) if i not in medoids]
            if not candidates:
                break
            next_medoid = int(rng.choice(candidates))
        medoids.append(next_medoid)

    # Alternate assignment/update is the standard scalable k-medoids
    # refinement.  It avoids the O(k*n^2) Python swap loop that made the
    # N=1000 formal configuration impractical.
    max_iter = 50
    for _ in range(max_iter):
        medoid_arr = np.asarray(medoids, dtype=int)
        assignments = np.argmin(distances[:, medoid_arr], axis=1)
        updated = medoid_arr.copy()
        for m_idx in range(len(medoid_arr)):
            members = np.flatnonzero(assignments == m_idx)
            if members.size == 0:
                nearest = distances[:, medoid_arr].min(axis=1)
                updated[m_idx] = int(np.argmax(nearest))
                continue
            within = distances[np.ix_(members, members)]
            updated[m_idx] = int(members[np.argmin(within.sum(axis=1))])
        # Deduplicate rare coincident updates deterministically.
        seen = set()
        for pos, value in enumerate(updated.tolist()):
            if value in seen:
                nearest = distances[:, updated].min(axis=1)
                for candidate in np.argsort(nearest)[::-1]:
                    if int(candidate) not in seen:
                        updated[pos] = int(candidate)
                        break
            seen.add(int(updated[pos]))
        if np.array_equal(updated, medoid_arr):
            break
        medoids = updated.tolist()

    # final assignments
    medoid_arr = np.array(medoids)
    dist_to_medoids = distances[:, medoid_arr]
    assignments = np.argmin(dist_to_medoids, axis=1)

    return medoid_arr, assignments


def _build_change_vectors(scenarios: ScenarioSet, data: ModelData,
                          proxy_profits: np.ndarray) -> np.ndarray:
    """Build standardized rate-of-change feature vectors for PAM.

    For each scenario, compute the parameter deviation from the mean,
    normalized by standard deviation.  Uses a subsample of key parameters
    for tractability.
    """
    n = scenarios.n
    features = []

    # demand deviations (subsample key crop-season-year)
    for (i, t, s), vals in sorted(scenarios.demand.items()):
        base = data.D.get((i, s), 0.0)
        if base > 0:
            features.append((vals - base) / (base + 1e-10))

    # yield deviations (subsample to keep dim manageable)
    count = 0
    for (j, i, t, s), vals in sorted(scenarios.yield_.items()):
        base = data.q.get((j, i, s), 0.0)
        if base > 0:
            features.append((vals - base) / (base + 1e-10))
            count += 1

    # price deviations
    for (i, t, s), vals in sorted(scenarios.price.items()):
        base = data.p.get((i, s), 0.0)
        if base > 0:
            features.append((vals - base) / (base + 1e-10))

    if not features:
        # fallback: use proxy profit as the only feature
        return proxy_profits.reshape(-1, 1)

    feat_matrix = np.array(features).T  # (n, D)
    # standardize
    mean = feat_matrix.mean(axis=0)
    std = feat_matrix.std(axis=0)
    std[std < 1e-10] = 1.0
    return (feat_matrix - mean) / std


def reduce_scenarios(data: ModelData, scenarios: ScenarioSet, k: int,
                     baseline_plan: dict, seed: int = 2024) -> ReducedScenarioSet:
    """Reduce N raw scenarios to K representatives via tail-protected PAM.

    Args:
      data: ModelData.
      scenarios: raw ScenarioSet (N scenarios).
      k: target number of representative scenarios.
      baseline_plan: (j,i,t,s) -> area from Q1 result1_1, for proxy profit.
      seed: random seed for PAM.

    Returns:
      ReducedScenarioSet with K scenarios and weights.
    """
    n = scenarios.n

    # 1. compute proxy profits for all scenarios
    proxy_profits = _compute_all_proxy_profits(scenarios, data, baseline_plan)

    # 2. identify the exact lowest-profit decile for CVaR tail protection
    sorted_idx = np.argsort(proxy_profits)
    bottom_layer = sorted_idx[:max(1, math.ceil(0.1 * n))]

    # 3. bottom decile gets at least ceil(0.1*k) representatives
    min_bottom = math.ceil(0.1 * k)

    # 4. build feature vectors and L1 distance matrix
    features = _build_change_vectors(scenarios, data, proxy_profits)
    # scipy.cdist computes the city-block metric without materialising an
    # n*n*d temporary tensor (which exceeded 2 GB in the formal run).
    distances = cdist(features, features, metric="cityblock")

    # 5. Profit-stratified k-medoids.  Full runs use ten equal-frequency
    # layers; small smoke tests use min(10, k) non-empty layers.
    n_layers = min(10, k, n)
    layers = [part for part in np.array_split(sorted_idx, n_layers) if len(part)]
    allocation = np.full(len(layers), k // len(layers), dtype=int)
    allocation[:k % len(layers)] += 1
    medoids = []
    for layer_no, (layer_idx, layer_k) in enumerate(zip(layers, allocation)):
        local_dist = distances[np.ix_(layer_idx, layer_idx)]
        local_medoids, _ = _pam_kmedoids(
            local_dist, int(layer_k), seed=seed + layer_no)
        medoids.extend(layer_idx[local_medoids].tolist())
    medoid_indices = np.asarray(medoids, dtype=int)

    # 6. Tail protection is inherent in stratification, but keep an explicit
    # assertion because the bottom decile is the key CVaR evidence.
    bottom_set = set(bottom_layer.tolist())
    bottom_count = sum(int(m) in bottom_set for m in medoid_indices)
    if bottom_count < min_bottom:
        local_dist = distances[np.ix_(bottom_layer, bottom_layer)]
        tail_medoids, _ = _pam_kmedoids(local_dist, min_bottom, seed=seed + 999)
        protected = bottom_layer[tail_medoids]
        replace_positions = np.argsort(proxy_profits[medoid_indices])[::-1]
        for pos, tail_idx in zip(replace_positions, protected):
            if int(tail_idx) not in set(medoid_indices.tolist()):
                medoid_indices[int(pos)] = int(tail_idx)

    # Recompute assignments after the final medoid order is frozen.  The
    # probability at position m therefore belongs to parameters at index m.
    assignments = np.argmin(distances[:, medoid_indices], axis=1)

    # 7. compute weights = cluster_size / N
    weights = np.zeros(len(medoid_indices))
    for m_idx in range(len(medoid_indices)):
        weights[m_idx] = np.sum(assignments == m_idx) / n
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("invalid reduced-scenario weights")

    # 8. extract reduced scenario parameters
    idx = medoid_indices
    demand = {key: vals[idx] for key, vals in scenarios.demand.items()}
    yield_ = {key: vals[idx] for key, vals in scenarios.yield_.items()}
    cost = {key: vals[idx] for key, vals in scenarios.cost.items()}
    price = {key: vals[idx] for key, vals in scenarios.price.items()}

    return ReducedScenarioSet(
        indices=idx,
        weights=weights,
        demand=demand, yield_=yield_, cost=cost, price=price,
        k=len(idx), n_original=n,
        proxy_profits=proxy_profits[idx],
    )
