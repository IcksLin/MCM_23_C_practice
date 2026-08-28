# -*- coding: utf-8 -*-
"""Q2 algorithms package — scenario-based stochastic MILP.

Modules:
  paths         — relative path configuration
  io_data       — read-only Excel ingestion (附件1/2 + result2 template)
  preprocess    — data cleaning & ModelData assembly (inherits Q1 logic)
  scenarios     — Latin Hypercube scenario generation
  scenario_reduction — PAM k-medoids with tail protection
  model         — mean-CVaR stochastic MILP assembly
  solve         — three-stage lexicographic optimization
  risk          — CVaR recomputation & knee-point frontier selection
  validate      — constraint audit
  export_ooxml  — safe result2.xlsx patching via ZIP/XML
  plots         — 9 required figures (raw/process/result)
"""
