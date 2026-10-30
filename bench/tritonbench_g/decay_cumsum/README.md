# Decay cumulative scans

- `Kernels.lean`: the full forward scan, Q/K preparation and reverse gradient
  scan from `decay_cumsum.py`, with their index-commuted candidates.
- `Correct.lean`: six real specifications, explicit finite IO windows and
  independent formulas. Execution and loop proofs are in `Proofs/`.
- `FPEquiv.lean`: three complete FP program equivalences. The only change is
  `i_k * BK` to `BK * i_k` inside the lane mask; all three print `none`.

The forward formula is `1.44269504 * sum(g[0..t])`. Preparation computes
`q[t] * exp2(g[t]) * scale` and `k[t] * exp2(g[last] - g[t])`.
Backward computes the corrected dq/dk streams and the reverse cumulative sum
of `dq[t] * q[t] - dk[t] * k[t]`, all from initial input arrays.
These are per-program real formulas after precision erasure.

Storage supports fp16, bf16 and fp32. Preparation and backward permit
independent data and gate storage. Source fp32 accumulators/conversions,
exp2 operations, loop order and final casts are retained. Preparation loads
all lanes of its final gate row without a mask; the IO read window includes
these extra cells. Backward retains its final pointer decrements and updates
dq/dk arrays in place.

The reverse source lowers to a signed loop. `BackwardLoop.lean` proves its
iteration-by-iteration correspondence with an explicitly written internal Nat
reference: for positive `BT`, both visit `BT-1, ..., 0`. The bridge deletes only
the dead `j`, `__rev_j` and cached `offs` registers when comparing final states;
all memory, failures, program IDs and other registers are preserved. This bridge
uses the existing `BT > 0` real-contract premise. A regression separately checks
that the actual signed source executes zero iterations at `BT = 0`.

The real contracts require `BK <= DK`, and preparation/backward also require
`BT > 0`. Their explicit region separation conditions protect input values
from earlier stores. Each contract also requires a finite disjoint allocation,
bounds for actual accesses and a zero unspecified-value oracle, and proves
successful full flat execution, active output formulas and preservation of
all other tagged cells. These conditions are visible at the real entry and
in its `Implements` definitions. Shapes and strides remain symbolic.

Finite real witnesses cover every storage pair with two iterations, nonzero
program IDs and a partial lane mask. Separate abstract FP executions from
empty registers witness that the full typed programs and candidates can run;
they are not GPU execution measurements. The FP structural equivalences
preserve complete states and failures and need no numerical atoms or extra
numerical-domain restrictions. No host launch, concurrent GPU execution,
IEEE rounding or fixed-width index-overflow guarantee is claimed.

Current validation and exact source snapshots are recorded in the dated
acceptance checkpoint. `FPDecayCumsumSource` checks precision/storage for all
source variants; `DecayCumsumReverse` checks the signed reverse-loop shape,
zero iterations at `BT = 0`, finite two-iteration execution for all storage pairs,
and a complete typed execution. Independent comparator replay remains a
separate environment-dependent acceptance gate; historical replay logs are
not evidence for a changed source snapshot.
