# -*- coding: utf-8 -*-
"""algorithms package: MILP modelling & solving for 2024 C-problem Q1.

Modules:
    paths      - portable path configuration
    io_data    - read-only Excel ingestion -> RawData
    preprocess - clean, inherit, build ModelData (suitability / D / history)
    model      - assemble sparse MILP (scipy.optimize.milp / HiGHS)
    solve      - primary + lexicographic solving
    validate   - numerical & constraint audit
    export_excel - write result1_1.xlsx / result1_2.xlsx from templates
    plots      - 9 raw/process/result figures (svg+png)
"""
