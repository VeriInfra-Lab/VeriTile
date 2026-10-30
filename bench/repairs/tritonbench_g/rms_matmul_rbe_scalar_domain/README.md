# RMS QKV scalar-domain candidate

The current public entries describe the complete Q→K→V kernel for every caller
supplied launch schedule. The quoted Python/kernel source and checked Tensor
contract are unchanged by this IO-format revision. The historical original
source defects and the separate scalar-domain candidate remain distinct.

`Correct.lean` has one public `Spec.Real` specification: the shared
`ArrayProgramIO.ImplementsOn` judgment connects named input arrays to the
independent RMS-normalized matrix product and adjacent-pair rotary formula.
`Misc/NativeCorrect.lean` has one ordinary native theorem using the same shared
judgment and an independent ordered-word reference. It is a source-to-native
observation, with no invented optimization or generic native=Real assertion.

The mathematical projection is
`sum_k(X[k]*RMS[k]*W[n,k]) / sqrt(sum_k X[k]^2/K + EPS)`.
Q/K additionally rotate adjacent pairs at
`(start+row)/THETA^((pair%head_dim)/head_dim)`; V is the plain projection.

## Public IO and unchanged scope

- `IO.lean` names all nine X/W/RMS input roles, finite read footprints, three
  output channels, source precision branches, actual invocations and runtime
  EPS/THETA register arguments. The input arrays may alias one another.
- `Target.lean` defines the mathematical and native functions from logical
  arrays. Neither target calls the interpreter or uses a source execution
  recurrence. Both have noncircularity audits.
- Every scheduled launch runs all three phases in order, including empty
  schedules. The original even-N condition and input-versus-output region
  separation remain visible in the public specifications.
- Output regions and strided output addresses may overlap. The observable
  output mask selects the final writer in actual program/phase/lane order,
  retaining its FP16 or E5 cell tag. `IOEvents.covered` proves that every active
  write has such a final writer at the same address; overwritten accesses
  remain covered by the safety proof and the exact exterior frame.
- FP32 X values and accumulation, FP16 or signed-INT8 W/RMS input storage,
  distinct E5 RMS/weight conversion paths, FP16 Q output and FP16/E5 K/V output
  remain source-selected. Real is the mathematical interpretation.
- The INT8 input condition applies only to the named W/RMS arrays. The old
  execution helper's stronger condition on unrelated integer cells is
  discharged by localized byte-conversion execution proofs, not made into a
  hidden assumption or an experimental TODO.

The shared judgment requires successful execution, all final typed outputs and
an exact exterior cell frame independently of allocation capacities. For every
capacity assignment covering the declared finite accesses, it also proves
actual Structural trace safety through the evaluated prefix, loop, epilogue,
Q/K/V successors and complete schedule. Declaring a window alone is not used
as a substitute for source-access safety.

## Proof organization

`Proofs/IOBinding.lean` connects the named arrays to the existing independent
scalar references. `IOByte*.lean` reuse the established loop, epilogue and
numerical recurrences while localizing valid-byte checks to actual operands.
`IOEvents.lean`, `IOJob.lean`, `IOGrid.lean`, `IOCoverage.lean` and `IOOutput.lean`
prove ordered writes, coverage, successful execution, typed results and frames.
`IOSafety*.lean` prove actual load/store safety; `IOResults.lean` composes the
shared IO judgments. `Correct.lean` contains the public Real specification and
`Misc/NativeCorrect.lean` retains the ordinary native theorem, with surface/axiom audits.

`Misc/KernelResults.lean` retains the previous complete raw grid statements
as ordinary helper theorems. `Misc/TensorBridge.lean` retains the exact existing
Tensor applications, physical memory/frame, pointer bounds and reshape-storage
results. These reuse the identified source execution foundation. Neither main
imports the Tensor bridge, and no helper imports a public entry or reader.
`Misc/Reader.lean` checks both interfaces.

## Existing scalar and numerical boundaries

The checked candidate admits positive finite Python EPS values, including
`True`, with integer EPS below `2**64`; THETA is a Python float in
`[2**-126, (2-2**-23)*2**127]`. These existing dispatch constraints belong to
the Tensor contract. Runtime scalar values remain runtime values in the
kernel IO; no literal specialization is introduced. Mathematical EPS/THETA
use their original scalar values. Native EPS may round to zero or infinity.

The native primitive boundary retains scalar operations/library functions,
format conversions, scalar literal materialization, one fixed K8 TF32 MMA
lane, the existing zero-conversion laws, reduction tree and selected rotary
FMA order. It admits arbitrary floating words, including nonfinite values.
No new numerical assumptions or numerical experiments were added here.

Historical validation includes 22 actual-wrapper offline scalar dispatch
cases, 35 candidate scalar-contract cases, 13 adapter cases plus 18 rejects,
and eight manual SM80 structural-signature controls. Those finite compiler
controls do not claim arbitrary compiler conformance. This revision performs
targeted Lean builds and audits; it adds no independent GPU run or numerical
replay. Compiler, allocator and launch-runtime internals are outside the
per-kernel proof scope.

## Complete implementation pair (2026-10-21)

`FPEquiv.rms_qkv_index_commuted_equiv` compares the actual complete typed
Q/K/V source with the existing final output-row index commutation
`pid_m * BLOCK_SIZE_M` → `BLOCK_SIZE_M * pid_m`. The two programs have ordinary
readable quotations for every FP16/E5 weight and output branch. EPS/THETA are
explicit `tl.param` FP32 external registers supplied by `IO.job`.
`Misc/SourcePreservation.readable_source_preserved` proves complete original
program identity with the previous generated source, including ports and all
precision scopes. The Python candidate is `rms_matmul_rbe_index_commuted.py`.

The main pair uses the shared array IO implementation syntax and an empty
experimental rule set. `Proofs/Pair.lean` proves callee, Q/K/V and full-schedule
execution equality and actual access-safety equivalence, then reuses the
existing successful source execution and exterior-frame theorem. All final
typed output cells agree. It preserves arbitrary schedules (including empty),
all input/output alias behavior admitted above, runtime scalar values, all
floating words and every original cast/operation order. The rewrite uses only
natural-number multiplication commutation; no floating algebra rule is needed.

`FPEquiv.native_word_instantiation` instantiates these same two typed sources
with the existing scalar/instruction word model. The main pair is the
`native=false` typed-source view; it does not equate that view to the distinct
`native=true` selected K8/FMA lowering. The latter's independent ordered-word
results remain ordinary optional theorems in `Misc/NativeCorrect.lean`.

The all-word candidate application `FPEquiv.native_word_instantiation` is optional
and lives in `Misc/CandidateEquiv.lean`; the primary entry contains only the
complete generic Q/K/V equivalence specification.
