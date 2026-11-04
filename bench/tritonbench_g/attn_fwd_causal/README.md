# attn_fwd_causal

[Kernels.lean](Kernels.lean) keeps the original [Python helper and caller](attn_fwd_causal.py) function bodies. It also pastes the complete caller from the existing [integer quotient/remainder candidate](attn_fwd_causal_quotient_remainder.py); both callers bind the same original helper. [jit-correspondence.json](jit-correspondence.json) checks all four body pairs, parameter classes, call bindings and helper bindings. [candidate-provenance.json](candidate-provenance.json) limits the candidate to the integer `off_h` expression.

Runtime integers, storage dtypes and the host's fixed `STAGE = 3` stay outside the pasted bodies. The optional frontend specializes the helper at each closed constexpr call, captures runtime arguments once, separates local workspaces and preserves tuple return assignment. The two actual loops retain their original bounds, masks, explicit casts, FP16 dot result, pointer advances and final store. The two previous hand-inlined whole quotations and three unused Lean-only parameters have been removed.

[Correct.lean](Correct.lean) states the independent D128 causal base-two softmax-weighted sum over `min(N, row + 1)` keys. It covers the full legal grid, typed outputs, access safety and exterior memory frame. [FPEquiv.lean](FPEquiv.lean) compares the complete actual runtime programs. The public entries retain their existing input and numerical domains.

This migration is pending. All 78 owned roots, the shared frontend and both regressions have passed locally, together with both actual public Real/FP theorems and prepared trust/specification audits. Shared-support cleanup and exact-current official independent replay are still required. Compilation and body comparisons alone do not complete this task. No GPU run is claimed.

Some older general mathematical helpers are still referenced by `attn_fwd_triton`; their finite-penalty formula is not the current causal kernel specification. Their remaining uses and inaccurate source descriptions must be cleaned up before final acceptance of this migration.
