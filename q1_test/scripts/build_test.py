# -*- coding: utf-8 -*-
"""Assemble the full MILP (no solve) to verify model.py builds cleanly."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from algorithms import io_data, preprocess, model as model_mod

raw = io_data.load_inputs()
md = preprocess.preprocess(raw)
print("ModelData ready.")

for sc in (1, 2):
    t0 = time.time()
    m = model_mod.build_model(md, scenario=sc, eta=0.5, demand_scale=1.0,
                              stage="primary")
    dt = time.time() - t0
    print(f"\n=== scenario {sc} primary built in {dt:.2f}s ===")
    print(f"n_vars={m.n}  n_ub={m.A_ub.shape[0]}  n_eq={m.A_eq.shape[0]}")
    print(f"  x_vars={len(m.x_idx)} y_vars={len(m.y_idx)} "
          f"r_vars={len(m.r_idx)} Q_vars={len(m.Q_idx)} u_vars={len(m.u_idx)}")
    print(f"  n_binary={int((m.integrality == 1).sum())}")
    print(f"  c nnz={int((m.c != 0).sum())}  finite ub={int(np.isfinite(m.ub).sum())}")
    print(f"  ub<0 (forced-zero vars)={int((m.ub == 0.0).sum())}")
    # check for inf in b
    print(f"  b_ub finite: {np.isfinite(m.b_ub).all()}  "
          f"b_eq finite: {np.isfinite(m.b_eq).all()}")

# lex stage
m_lex = model_mod.build_model(md, scenario=1, eta=0.5, demand_scale=1.0,
                               stage="lex", z_star=1.0e7, delta=0.0)
print(f"\n=== lex stage built ===  n_ub={m_lex.A_ub.shape[0]} "
      f"n_eq={m_lex.A_eq.shape[0]}  n_vars={m_lex.n}")
print("OK: model assembles for both scenarios and lex stage.")
