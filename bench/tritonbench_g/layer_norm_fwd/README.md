# layer_norm_fwd

- Source file: `layer_norm_fwd.py` (upstream `data/TritonBench_G_v1/layer_norm_fwd.py`)
- Corpus: TritonBench-G v1
- Size: 218 lines, 1 `@triton.jit` kernel(s)
- Source pin: upstream commit `603e28a`, SHA256
  `27dfccf23671d5b9c6f4b749cef39ddf69079793c486a833cbae9c57fe3a4de4`.
- Status: source imported; selected fp32 AST audit and lowering checks pass;
  full-state fp32 FP equivalence and complete original/candidate real correctness are proved.
  Finite memory execution, seven-output IO and full frame are also proved.
  A complete typed nonempty execution witness is proved in an explicit
  `WithBot ℝ` consistency algebra. Other storage profiles and the native
  scalar/index ABI bridges remain pending, including for selected fp32 storage.
  This task is **not complete**.

`Kernels.lean` contains the full source macro, retaining all ten constexpr
flags, both seed loads and all outputs. Its candidate changes only the order
of the integer factors in the initial X row-pointer increment. `Proofs/Formula.lean`
defines independent mathematical formulas over the input snapshot, including
seven-round canonical Philox dropout, padding, both affine outputs and all
side outputs. `Proofs/Contracts.lean` specifies the exact tagged memory result and
frame for every output. `Proofs/Real.lean` proves that the complete original source successfully executes
and establishes this initial-snapshot contract. `Proofs/FP.lean` proves equality
of complete fp32 executions, including failures, all flags and every state field;
`FPEquiv.lean` exposes this through the public specification API.

The selected-source audit metadata in `Kernels.lean` selects fp32 for the ten
floating storage arguments and retains all ten symbolic flags. Its complete AST
comparison preserves every branch, both signed seed loads, and all stores; the
only materialization is the checked identity binding for `BLOCK_N`. This audit
passes the 24 mechanical checks on the actual port entry. It does not cover
every storage/native profile or replace proof replay.

The selected fp32 storage profile does not establish the native runtime-scalar
ABI. The Lean parameters `eps` and `dropout_p` are mathematical reals, and
strides, `M` and `N` are unbounded naturals. The native samples explicitly use
fp32 scalar arguments and i32 indices. Relating Python float conversion to the
modeled scalar values, and proving that native fixed-width index arithmetic
matches the natural-number addresses, are still open obligations. Finite
allocation bounds alone do not prove the latter.

The source **loads int64 seeds and then casts to uint32**; treating the seed
pointer itself as uint32 changes the wrapper ABI. The typed frontend preserves this ABI and implements the cast modulo `2^32`. Dropout keeps
the strict `random > dropout_p` comparison and the eagerly evaluated division.

`scripts/native_lowering.py` compiled six complete-source profiles with installed
Triton 3.8.0 for SM90, including both dual-input dropout and every optional
output. The receipt is `native_lowering_sm90_triton_3_8_0.json`; it records two
int64 loads followed by i64-to-i32 truncations in dual-input dropout profiles.
This is compiler admission evidence, with no GPU launch or numerical proof.

`Proofs/SeedSource.lean` checks that the nested random seed expression retains
the signed int64 load before the modular cast. `Proofs/Execution.lean`,
`Proofs/Scale.lean` and `Proofs/Memory.lean` prove initial execution phases and
exact tagged scatter behavior. `Proofs/FirstStream.lean` proves successful
execution of the actual first twelve source statements, independent first-stream
values, and the exact boolean mask/frame result. `Proofs/PhaseAudit.lean` checks
27 phase theorems with `#axiomsClean`. Full source composition is proved separately in `Proofs/Real.lean`.

`Proofs/NormalizationExecution.lean` proves successful execution of the actual
normalization branch and both scalar stores, with independent mean, variance
and Rstd formulas and exact tagged frame. Its 13 audit gates pass. `Proofs/NormPrefix.lean` derives its combined-input tile
from the actual preceding second-stream and residual statements.

`Proofs/SecondStream.lean` executes the complete second input body, including
its distinct rowscale and signed seed addresses, second mask and addition;
its 18 audit gates pass. `Proofs/Residual.lean` proves successful residual
load/add and ResidualOut storage. The 24 gates in
`Proofs/InputPreservationAudit.lean` also prove that active input regions retain
their original tagged cells under `Domain.ReadWriteSeparated`, so later loads
match the initial snapshot. The separation condition permits aliases among
inputs and imposes no condition on inactive arguments. The complete first eighteen statements pass 21 `Proofs/PrefixAudit.lean` gates.
The affine proofs retain all remaining source statements, including unspecified
masked coefficient lanes and both output stores. `Proofs/RealAudit.lean` passes
36 gates for complete original execution and its independent seven-output
contract, with no intermediate-execution premises. `Proofs/CandidateAudit.lean` passes seven gates for the candidate’s independently
executed changed prefix, full source composition and the same independent
contract.

`Proofs/RealIO.lean` proves successful finite memory execution and the complete
seven-output contract for both source versions. `Proofs/Safety.lean` derives
all active access bounds from the launch row and actual allocation extents;
trace safety is not a public premise. `Proofs/FlatContract.lean` preserves the
full tagged frame, including relocated pointers. The flatten bridge requires
zero `undef` explicitly; omitted-other loads remain in the source. The 40 gates
in `Proofs/IOAudit.lean` pass.
`Proofs/RealSurface.lean` also exposes the same independent contract through
the standard `RealizesFrame_without_Rounding` API, with explicit successful
algorithm projection; its four audit gates pass.

`Proofs/TypedExecution.lean` proves complete typed executions from empty
registers at PID 1, with negative int64 seeds, `N=2`, `BLOCK_N=4`, every optional
feature enabled and both normalization modes. It executes the actual source
loads, random generators, eager operands, reductions and stores. Its explicit
`WithBot ℝ` consistency algebra is not an IEEE or native GPU model. The 14 gates
in `Proofs/TypedAudit.lean` pass; `FPEquiv.lean` exposes the ordinary nonempty
theorem alongside the universal fp32 equivalence specification.

See `source_scope.md` for the remaining obligations and `source_pin.json` for
the exact imported artifact.
