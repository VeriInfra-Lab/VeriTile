# Scalar-cast-only Llama feed-forward repair

This independent sibling changes exactly twelve runtime dimension/stride casts
from `x.to(tl.int64)` to `tl.cast(x, tl.int64)`. Triton's default binder can
specialize unit scalars to Python integers; the explicit language cast accepts
those integers. The original 25-file task and accepted source-defect packet are
preserved. `pid.to(tl.int64)`, every wrapper check and all other Python remain
unchanged.

The public Real specification is `LlamaFFScalarCast.Correct.kernel_correct` in
`Correct.lean`. The optional source-to-native theorem
`LlamaFFScalarCast.FPEquiv.kernel_equiv` is retained in `Misc/NativeCorrect.lean`;
the complete original/candidate implementation pair is `FPEquiv.source_equivalence`. These results retain
the complete existing contract: both FP16 and signed-int8/E5M2 storage, arbitrary
half/E5M2 bit patterns, positive arbitrary K/N, unit dimensions and strides,
M/N/K tails, and empty leading dimensions. Configuration stays 16×16×64, four
warps, two stages and EPS=1e-6. No divisibility or finite-value condition was
introduced.

`Kernels.lean` contains four complete ordinary `triton` quotations using the
shared DSL: original and candidate for each storage branch. Explicit signed
scalar casts remain unchanged; signed ceiling division is written directly as
`(a+b-1)//b`. The previous adapter is retained only in
`Misc/LegacyQuotation.lean`, outside the main dependency closure. Complete
original/candidate AST identities against the previous quotations are checked
by reflexivity in the campaign preservation artifact. `check_source.py` compares both quotation bodies with the
selected Python AST. Its normalizations are parameter/SSA names, static
storage-branch selection, shape tuple syntax, explicit signed cdiv expansion and
the implicit output half cast.

The execution proof evaluates the actual shared prelude, loop, normalization,
address arithmetic, masks and store. The source interpretation proves the
independent mathematical result

```
rho[m] = 1 / sqrt(sum(k<K) X[m,k]^2 / K + EPS)
u[m,n] = rho[m] * sum(k<K) X[m,k] * RMS[k] * W1[n,k]
v[m,n] = rho[m] * sum(k<K) X[m,k] * RMS[k] * W3[n,k]
Y[m,n] = u[m,n] / (1 + exp(-u[m,n])) * v[m,n].
```

This ideal Real interpretation erases rounding and explicitly maps nonfinite
half/E5M2 payloads to zero. It is distinct from the all-word native model and
does not claim native values equal Real arithmetic.

The native execution substitutes two local live-carry dot stages, the observed
direct E5M2-to-half weight conversions, and the scalar FMA/sigmoid gate. Each dot is an explicit four-K16 chain;
the selected 64-lane butterfly reduction is explicit. `NativeReference.lean`
independently flattens the K16 instruction stream. Half multiplication rounds
before either projection. The source quotation retains W1/W3 conversion through
FP32 and then half. The selected native compiler contracts those conversions
to direct E5M2-to-half instructions; the native model and independent native
target use those direct instructions. RMS also converts directly to half. No
equality between direct and two-step exceptional-value conversions is assumed.
Libdevice integer power uses both FTZ multiplies.
The final gate retains the negative-raw-accumulator FMA, projection order and
half narrowing. No whole quotation/program/event/output binding is a premise.

The native carrier stores raw 16-bit half words injectively in 32-bit cells and
uses raw 8-bit E5M2 payloads. Exact little-endian encode/decode, half readback,
full-grid coverage and the complete relevant byte frame are proved, including
all input regions and output holes. `TensorView.Output` records only actual
returned Tensor metadata and coordinate observations. The `TensorBridge`
applications give the result at its `[batch, sequence, n]` physical coordinates. Signed index and
formed-offset bounds are derived from the unchanged public checks. The work is
kernel-only; it adds no allocator, host orchestration or unified-heap premise.

Ordinary interpretation boundaries remain explicit: truthful Tensor metadata
and coordinates, the audited Python/quotation match, the selected compiler's
local fusion/fragment routing, and the raw scalar/K16 instruction meanings in
`Words.Instructions`. That record has only scalar operations and one fixed-K16
result lane; it has no reduction, matrix, result, execution or frame callback.
No primitive experiment is required to use the structural theorems. These
theorems do not assert an unmeasured blanket SM90 exceptional-payload profile.

Validation uses the required v2 Lean build wrapper and recursive source/import
artifact pinning. Ten actual Triton 3.7.1 default-binder SM90 compiles pass:
M/N/K = 1/1/1, 2/16/64, 17/19/63, 17/19/65 and 17/19/129, in both storage
branches. Unit specialization remains enabled. Both empty calls skip launch.
All ten retained PTX artifacts have sixteen fixed m16n8k16 instructions and eight
terminal FMAs. These representative compiles are not exhaustive compiler or
resource validation. Tensor allocations, GPU executions and numerical
experiments are all zero.

The corrected frozen delivery is in
`.veritile-cache/task-deliveries/llama_ff_triton_scalar_cast/delivery-v2.json`.
The first-round packet and its source/artifact snapshot are preserved in
`delivery.json` and `round001/`.
Root owns candidate registration and selection; this sibling does not edit the
task ledger, Lake configuration or shared core.

The public kernel interfaces use named typed input arrays and shared IO judgments, as described below. `Misc/RawCorrect.lean` and `Proofs/RawNative.lean` retain the raw geometry/state and byte-memory results. `Misc/TensorBridge.lean` retains the complete `correct`, `kernel_correct`, `fp_equiv` and `kernel_equiv` Tensor applications (under `TensorBridge`), together with metadata corollaries, derived from those shared execution results.


## Public IO organization (2026-10-20)

`Correct.lean` now exposes one `ArrayProgramIO.ImplementsOn` specification with
an independent decoded RMSNorm–SwiGLU formula. `Misc/NativeCorrect.lean` retains one ordinary
`ArrayProgramIO.ObservesOn` theorem for the existing native half-word
result. Both prove actual source access safety, successful complete-grid
execution, every output dtype/value, and the exact exterior cell frame.

`IO.lean` lists the actual source/compiled stages, seven named typed input
views, finite active read footprints, output coordinates, empty schedule, and
no private scratch. Its simultaneous half/int views retain the original raw
`ByteValues` scope; the int8 condition is a logical input domain, not a new
floating or experimental assumption. It also carries the logical array
adapters to the independent finite Real target. `Misc/NativeReference.lean`
retains the optional ordered native instruction expression, and its input
connection and observation result are in `Misc/NativeInputBinding.lean` and
`Misc/NativeIOResults.lean`.

The native path retains the existing compiler boundary: original prelude,
compiled K loop, original normalization, selected scalar gate, and original
masked half store. `Proofs/GateSource.lean` reifies only that already selected
gate and proves exact equality to `gateExecution` for every state;
`Proofs/SourceGateSite.lean` certifies the original ordered source operand site
by reflexivity. The gate-only backend supplies the existing scalar ports and
no program/loop/output callback. No source-to-native arithmetic equality or
additional optimization is claimed.

`Misc/RawCorrect.lean` and `Proofs/RawNative.lean` retain the previous complete
statements, including arbitrary original native bytes, finite two-byte half
stores, every exterior byte, and exact empty-launch identity. `Misc/TensorBridge.lean`
retains all previous Tensor statements and uses those same execution results.
`Misc/PhysicalIOBinding.lean` connects the original byte readers to the named IO
arrays and both independent output formulas; `Proofs/IOBounds.lean` supplies
concrete satisfiable capacities. Main entries and supporting proofs do not
import the optional Tensor bridge. `Misc/ProofAudit.lean` audits helper results.

Python source, authoritative `Kernels.lean`, input contract and numerical
parameters are unchanged. This organization/build work runs no GPU experiment
or independent numerical replay and introduces no numerical assumption.

## Implementation-pair review (2026-10-21)

The historical output-row index candidate is now instantiated on the selected
repaired source in both FP16 and int8/E5M2 storage branches. The candidate changes
`pid_m * BM` to `BM * pid_m` only in the output-row address calculation. It retains
all twelve explicit signed scalar casts, four K masks, conversions, floating
operations and their order. The independent actual Python candidate is
`llama_ff_index_commuted.py`; no performance measurement is claimed.

`PairSource` proves the full source states and actual access safety agree for
arbitrary selected FP primitive models, using exact signed `Int.mul_comm`.
`PairExecution` lifts this through every CTA in the complete original grid.
`PairIO` proves both schedules safe and successful, all observed output cells
equal and both complete exterior typed-cell frames. `FPEquiv.source_equivalence`
therefore publishes `original ≡[[]] candidate` on exactly the retained `byteDomain`,
including both storage branches, arbitrary payloads and empty leading dimensions.
No floating rewrite, new numerical condition or experimental rule is used.

The retained source-to-native theorem remains separate in
`Misc/NativeCorrect.lean`; compiler-local observations keep their original names,
domains and modeling boundaries. The independent decoded Real specification
remains unchanged. The standard-frontend migration is connected to all four
previous complete source bodies by exact AST equality, and the new Python/Lean
source checker matches every complete quotation to its actual Python branch.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/NativeCorrect.lean`
- `Misc/PhysicalIOBinding.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.

## Reader headline separation (2026-10-22)

`Correct.lean` retains the complete original kernel's independent decoded Real
headline, and `FPEquiv.lean` retains the existing full-grid source pair. The
optional raw-state application `RawCorrect.kernel_correct` is now authoritative
in `Misc/RawCorrect.lean`; its previous Proofs path is a compatibility import.
The Tensor bridge imports that optional result explicitly. The reader no longer
imports the raw-state application. Declaration names, statements, proofs, source
programs, domains and numerical assumptions are preserved.
