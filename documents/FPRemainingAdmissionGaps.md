# Current FP rule usage and proof boundaries

The FP examples are complete for their stated sources and domains. There are
19 example groups and 21 public FP specifications: 10 groups use experimental
rules and 9 use only strict-model or structural proofs. At specification level,
the split is 10 with experimental rules and 11 without them; HyperConnectionsWidth
has one strict scalar specification and one experimental matrix specification.
See the [example index](../bench/examples/README.md) for the individual entries.

The [task ledger](../bench/tritonbench_g/task_progress.json) tracks 183 tasks:
**183 accepted actual kernel pairs: 182 completed and 1 completed with deferred
numerical evidence**. Of those 183 pairs, 178 use strict-model/structural proofs,
4 use the existing FP32 DIV-MUL-RCP admission (`apply_penalty`,
`softmax_triton1`, `softmax_triton2`, `softmax_triton3`), and `matmul_leakyrelu`
depends on one deferred local dot/add-to-dotAcc rule. The deferred rule is `MATMUL-LEAKYRELU-FP32-K32-DOT-ADD-TO-DOTACC`, instance
`matmul_leakyrelu.fp32.tf32.k32.live_acc.all_words.v1`. Both gates are `NOT_RUN`,
with no evidence artifact; the complete program proof explicitly depends on
future local evidence/admission. See its
[precision, applicability domain and experiment plan](../bench/repairs/tritonbench_g/matmul_leakyrelu/KernelOnly/NUMERICAL_VALIDATION.md).
The current auxiliary experiment queue is empty. Program-proof gaps cannot be
reclassified as numerical deferrals. Per-task formulas, domains, actual program pairs and proof scope are recorded in
the [final contract review](../bench/validation/contracts.json). Current build
and replay results are checked independently by CI.

## Available catalogue and actual use

[CurrentAdmission](../VeriTile/Triton/Float/CurrentAdmission.lean) classifies the
78 distinct historical instances as 42 strict-model proofs and 36 retained
experimental instances. The 36 are a catalogue pool, not a count of relations
used by every proof or enabled by every rule API.

The 183 accepted task-pair statements and the 21 example specifications together use
**12 of those 36 admitted experimental instances**, plus the separate deferred
local rule above. The four admitted-rule task consumers use the same FP32
DIV-MUL-RCP instance already used by the examples. The other 24 are unused
by these main statements; that does not make them unused throughout the library.
For example, the libdevice LOG-MUL-GUARDED variant has public API and test uses.
Imports, unused `R` fields and complete `Rules.assumptions` lists do not count as
actual proof use. This inventory is not a proof that the used assumptions are minimal.

The following is the complete admitted main-statement relation list; the pending
dot/add-to-dotAcc relation is listed separately above. P32 means
input/compute/accumulator/output = FP32/FP32/FP32/FP32. These are the local
relation's precision fields; FloatDTypeSoftmax still performs its actual FP32
load before widening to the FP64 operands of its local relation.

| Experimental relation | Precision | Operand guards and implementation |
| --- | --- | --- |
| ADD-ASSOC | P32 | `a`, `b`, `c` finite |
| ADD-ZERO | P32 | `a` finite, including both zero signs |
| CANCEL | P32 | `a`, `b` finite |
| MUL-ASSOC | P32 | `a`, `b`, `c` finite |
| MUL-DISTRIB | P32 | `a`, `b`, `c` finite |
| MUL-RCP-CANCEL | P32 | `a` finite and nonzero |
| DIV-MUL-RCP | P32 | `a`, `b` finite; `b` nonzero; ordinary division |
| DIV-MUL-RCP | FP64/FP64/FP64/FP32 | Same guards; equality includes the final FP32 conversion |
| EXP-SUB | P32 | `a`, `b` finite; `libdevice.exp` |
| LOG-EXP-GUARDED | P32 | `a` finite; `libdevice.log` with `libdevice.exp` |
| LOG-EXP-GUARDED-INTRINSIC | P32 | `a` finite; `tl.log` with `libdevice.exp` |
| LOG-MUL-GUARDED-INTRINSIC | P32 | `a`, `b` finite and positive; `tl.log` |

## Conditions retained by the proofs

- `Correct` proves independent Real formulas using Mathlib. `FPEquiv` preserves
  typed FP operations, casts, FMA distinctions, order, masks and stores.
- Strict certificates apply in their named model and configuration. Empty
  experimental `R` can still use the fixed strict software theory; it does not
  assert the identities for arbitrary opaque operations or prove arbitrary
  `Models` exist. The printer separates strict certificates from experimental use.
- Experimental relations retain their exact precision, implementation and
  operand guards. Their conditional `≡[R]` conclusions are not unconditional
  IEEE bit equalities or whole-kernel error bounds. Finite input leaves alone
  do not establish finite intermediate sums, reciprocals or rewrite operands.
- ADD-ZERO remains experimental on the original finite domain: nearest-even
  addition gives `(-0) + (+0) = +0`. The domain has not silently excluded -0.
- COUNT-ZERO and bounded COUNT-SUCCESSOR now have strict certificates. Welford
  retains `0 < N <= 2^24`; LayerNorm retains `N <= 2^24` and covers empty output
  rows. An unrestricted successor-conversion identity is not available.
- Guarded log-exp returns its input only when `0.5 < abs(a) <= 80`; otherwise it
  keeps the original log-exp computation. Guarded log-product keeps the product
  log when `0.5 <= fp32(a*b) <= 2` and splits outside that interval. These branches
  and inactive-argument masks belong to the candidate program, not extra input
  restrictions or permission to use the corresponding unconditional identity.
- Local relations are connected to successful kernel executions, typed outputs
  and memory frames. Whole softmax, Welford, LayerNorm and reduction equalities
  are derived conclusions, never added as atomic numerical assumptions.

## Unsupported substitutions and historical evidence

The selected LogExp API does not admit its unconditional log-product or log-exp
candidates. Its guarded variants retain their fallbacks. A historical catalogue
row cannot enable a different backend, domain or candidate graph. The measured
`tl.exp` EXP-SUB and guarded log-exp variants failed the configured bias gate;
the accepted shift/cancellation variants use `libdevice.exp`. Ordinary division
and `div_rn` also remain distinct.

StableLogSumExp proves the fallback-preserving candidate against its unchanged
direct reference. Its [older unconditional target](../bench/examples/StableLogSumExp/Proofs/Unconditional.lean)
has a checked logical countermodel satisfying the selected rules and domain;
this is not a GPU failure. Older main-table-only algebraic countermodels do not
establish a missing rule in today's combined strict and experimental theory.

Use the [rewrite-rule guide](FloatingPointRewriteRules.md) for relation definitions
and the [special-value guide](FloatingPointSpecialValues.md) for model boundaries.

For a changed proof, inspect `#print_fp_assumptions` on its actual public entry,
match each used experimental fragment to its precision/backend/domain evidence,
and retain the kernel connection. Reuse matching strict certificates without
repeating their numerical experiments. Do not delete a retained catalogue
instance merely because these main statements do not use it.
