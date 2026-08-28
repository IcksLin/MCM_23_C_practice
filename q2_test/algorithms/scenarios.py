# -*- coding: utf-8 -*-
"""Latin Hypercube scenario generation for Q2 (AGENT.md section 4).

Generates N0 raw scenarios of uncertain parameters:
  - Demand: wheat/corn (codes 1-5) compound growth g∈U(0.05,0.10);
           other crops level shock ε∈U(-0.05,0.05)
  - Yield:  ε∈U(-0.10,0.10) per (j,i,t,s)
  - Cost:   deterministic c_t = c_2023 * 1.05^k  (not random)
  - Price:  grain=const, vegetable=(1.05)^k, morel=(0.95)^k,
           mushroom random decrease g∈[0.01,0.05] compound

Distribution: uniform (primary) or triangular (verification, same ranges).
Seed: 2024 default.

Usage:
    from algorithms.scenarios import generate_raw_scenarios
    scenarios = generate_raw_scenarios(data, n=1000, seed=2024)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import triang

from .preprocess import ModelData, GRAIN_CODES, RICE_CODE, MUSHROOM_CODES, MOREL_CODE, crop_price_group


@dataclass
class ScenarioSet:
    """Container for generated scenarios.

    Attributes:
      weights: (N,) probability weights, all 1/N for raw scenarios.
      demand: dict (i, t, s) -> np.ndarray(N,) expected sales in jin.
      yield_: dict (j, i, t, s) -> np.ndarray(N,) yield in jin/mu.
      cost:   dict (j, i, t, s) -> np.ndarray(N,) cost in yuan/mu (deterministic).
      price:  dict (i, t, s) -> np.ndarray(N,) price in yuan/jin.
      n: number of scenarios.
      seed: random seed used.
      distribution: "uniform" or "triangular".
    """
    weights: np.ndarray
    demand: dict = field(default_factory=dict)     # (i,t,s) -> array(N,)
    yield_: dict = field(default_factory=dict)     # (j,i,t,s) -> array(N,)
    cost: dict = field(default_factory=dict)       # (j,i,t,s) -> array(N,)
    price: dict = field(default_factory=dict)      # (i,t,s) -> array(N,)
    n: int = 0
    seed: int = 2024
    distribution: str = "uniform"


WHEAT_CORN_CODES = frozenset((6, 7))


def _lhs_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw one randomized Latin-hypercube dimension on ``[0, 1)``.

    Each of the ``n`` equiprobable strata is sampled exactly once.  Calling
    this function independently for every uncertain parameter is equivalent
    to constructing a high-dimensional randomized LHS, but avoids allocating
    one very wide ``n x d`` matrix (Q2 has more than seven thousand yield
    dimensions).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return (rng.permutation(n) + rng.random(n)) / n


def _lhs_sample(rng: np.random.Generator, lo: float, hi: float, n: int,
                distribution: str) -> np.ndarray:
    """Map one LHS unit dimension to the requested bounded distribution."""
    if not hi > lo:
        raise ValueError(f"invalid sampling interval [{lo}, {hi}]")
    u = _lhs_unit(rng, n)
    if distribution == "uniform":
        return lo + (hi - lo) * u
    if distribution == "triangular":
        # Symmetric triangular distribution with its mode at the midpoint.
        return triang.ppf(u, 0.5, loc=lo, scale=hi - lo)
    raise ValueError(f"unsupported distribution: {distribution}")


def generate_raw_scenarios(data: ModelData, n: int = 1000, seed: int = 2024,
                           distribution: str = "uniform") -> ScenarioSet:
    """Generate N raw scenarios via LHS or independent sampling.

    For high-dimensional parameter spaces (D > 200), uses independent
    stratified sampling per dimension (equivalent to LHS marginally).
    For D <= 200, uses scipy.stats.qmc.LatinHypercube for full LHS.

    Args:
      data: ModelData from preprocess.
      n: number of scenarios to generate.
      seed: random seed.
      distribution: "uniform" or "triangular".

    Returns:
      ScenarioSet with arrays of shape (n,) per parameter.
    """
    if distribution not in {"uniform", "triangular"}:
        raise ValueError("distribution must be 'uniform' or 'triangular'")
    rng = np.random.default_rng(seed)

    def sample(lo: float, hi: float) -> np.ndarray:
        return _lhs_sample(rng, lo, hi, n, distribution)

    # ---- demand generation ----
    # wheat/corn (codes 1-5): compound growth g~U(0.05,0.10)
    # other crops: level shock ε~U(-0.05,0.05) relative to 2023
    demand = {}
    is_pairs = sorted({(i, s) for (j, i, s) in data.suit.keys()})
    for (i, s) in is_pairs:
        d_base = data.D.get((i, s), 0.0)
        if d_base <= 0:
            continue
        if i in WHEAT_CORN_CODES:
            d_vals = np.full(n, d_base, dtype=float)
            for t in data.years:
                # One new annual rate per future year: 2024 contains exactly
                # one factor, 2025 two factors, ..., 2030 seven factors.
                d_vals = d_vals * (1.0 + sample(0.05, 0.10))
                demand[(i, t, s)] = d_vals.copy()
        else:
            for t in data.years:
                # Other crops fluctuate independently around the 2023 level;
                # they do not accumulate a random walk.
                demand[(i, t, s)] = d_base * (1.0 + sample(-0.05, 0.05))

    # ---- yield generation ----
    # ε~U(-0.10,0.10) per (j,i,t,s)
    yield_ = {}
    for t in data.years:
        for (j, i, s) in sorted(data.suit.keys()):
            q_base = data.q.get((j, i, s), 0.0)
            if q_base <= 0:
                continue
            shock = sample(-0.10, 0.10)
            yield_[(j, i, t, s)] = q_base * (1.0 + shock)

    # ---- cost generation (deterministic) ----
    # c_t = c_2023 * 1.05^k
    cost = {}
    for t in data.years:
        k = t - 2023
        factor = 1.05 ** k
        for (j, i, s) in sorted(data.suit.keys()):
            c_base = data.c.get((j, i, s), 0.0)
            cost[(j, i, t, s)] = np.full(n, c_base * factor)

    # ---- price generation ----
    # grain: p_t = p_2023 (constant)
    # vegetable: p_t = p_2023 * 1.05^k (deterministic)
    # morel: p_t = p_2023 * 0.95^k (deterministic)
    # other mushroom: random decrease g∈[0.01,0.05] compound
    price = {}
    for (i, s) in is_pairs:
        p_base = data.p.get((i, s), 0.0)
        if p_base <= 0:
            continue
        grp = crop_price_group(i)
        if grp == "mushroom":
            p_vals = np.full(n, p_base, dtype=float)
            for t in data.years:
                p_vals = p_vals * (1.0 - sample(0.01, 0.05))
                price[(i, t, s)] = p_vals.copy()
        else:
            for t in data.years:
                k = t - 2023
                if grp == "grain":
                    price[(i, t, s)] = np.full(n, p_base)
                elif grp == "vegetable":
                    price[(i, t, s)] = np.full(n, p_base * (1.05 ** k))
                elif grp == "morel":
                    price[(i, t, s)] = np.full(n, p_base * (0.95 ** k))

    return ScenarioSet(
        weights=np.full(n, 1.0 / n),
        demand=demand, yield_=yield_, cost=cost, price=price,
        n=n, seed=seed, distribution=distribution,
    )


def scenario_to_dataframe(scenarios: ScenarioSet, data: ModelData,
                          scenario_idx: int = 0) -> pd.DataFrame:
    """Extract one scenario's parameters as a tidy DataFrame.

    Columns: scenario, year, plot, land_type, crop, season,
             demand, yield, cost, price
    """
    rows = []
    for t in data.years:
        for (j, i, s) in sorted(data.suit.keys()):
            ptype = data.plot_type[j]
            dem = scenarios.demand.get((i, t, s), np.array([0.0]))[scenario_idx]
            yld = scenarios.yield_.get((j, i, t, s), np.array([0.0]))[scenario_idx]
            cst = scenarios.cost.get((j, i, t, s), np.array([0.0]))[scenario_idx]
            pri = scenarios.price.get((i, t, s), np.array([0.0]))[scenario_idx]
            rows.append({
                "scenario": scenario_idx,
                "year": t,
                "plot": data.plot_names[j],
                "land_type": ptype,
                "crop_code": i,
                "crop_name": data.crop_names.get(i, ""),
                "season": s,
                "demand": dem,
                "yield": yld,
                "cost": cst,
                "price": pri,
            })
    return pd.DataFrame(rows)


def compute_proxy_profit(scenarios: ScenarioSet, data: ModelData,
                          plan_x: dict, scenario_idx: int = 0) -> float:
    """Compute proxy profit for one scenario given a fixed plan.

    plan_x: dict (j,i,t,s) -> area (mu).
    Uses conservative u = min(Q, D).
    """
    profit = 0.0
    # group production by (i,t,s)
    Q = {}
    for (j, i, t, s), area in plan_x.items():
        if area <= 0:
            continue
        q = scenarios.yield_.get((j, i, t, s), np.array([0.0]))[scenario_idx]
        Q[(i, t, s)] = Q.get((i, t, s), 0.0) + q * area

    for (i, t, s), q_tot in Q.items():
        d = scenarios.demand.get((i, t, s), np.array([0.0]))[scenario_idx]
        u = min(q_tot, d)
        p = scenarios.price.get((i, t, s), np.array([0.0]))[scenario_idx]
        profit += p * u

    # subtract cost
    for (j, i, t, s), area in plan_x.items():
        if area <= 0:
            continue
        c = scenarios.cost.get((j, i, t, s), np.array([0.0]))[scenario_idx]
        profit -= c * area

    return profit
