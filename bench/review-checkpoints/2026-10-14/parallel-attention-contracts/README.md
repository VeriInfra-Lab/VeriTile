# Parallel attention public-contract correction

Both complete source surface definitions and the Python file are unchanged.
The lowered AST now retains the original identity-carrier casts, explicit
no-TF32 dot options and unspecified load padding. The eight independent guarded
input references use the initial state's address-indexed `undef` values rather
than inventing zero padding. The backward masked `dz` formula is included.

`public.diff` preserves the full before/after public statements: the forward
hypotheses remain unchanged, while the backward global zero-undef premise is
removed. `references.diff` preserves the complete eight formula changes.
`changes.diff` and `original.lean.txt` allow review against the exact baseline.
The final source is captured by the separate formal build receipt.

A registered full-execution regression has arbitrary padding input `u` and
proves both actual output stores equal `u⁴`; it supplies no desired output as
an assumption. Real arithmetic, the existing stride/trip-count conditions,
host/helper boundaries and the named-surface/frame review boundaries remain
explicit. Native bitwise refinement and GPU execution are not claimed.
