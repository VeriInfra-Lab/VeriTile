# softmax_optimize: complete row IO and checked Tensor applications

The selected source is `softmax_optimize_repaired.py` (SHA256
`b019cf887fd78db7708f38955364d6409f4a126ca2db0aba2b0ec154d304251c`).
This revision leaves that Python source, its guards, four loops, FP32 computation,
fusion-enabled launch, four warps and three stages unchanged. `Kernels.lean`
contains the exact Python function bodies for the original, existing candidate,
and `prev_multiple_of` helper. All four loops remain visible in each body.
`with_triton_signature` supplies the FP32 pointer and signed runtime-parameter
profile outside `triton { ... }`; the invocation binds `N` outside that body.
`M` remains a declared but unused source parameter. No extra `cols` assignment,
Lean metadata, antiquotes, removed casts or rewritten helper appears in the body.

`jit-correspondence.json` compares all three complete bodies as Python ASTs and
checks their external profiles. The existing Python modules and wrappers are
unchanged. The candidate still swaps only the four signed row products.
`Proofs/SourceCandidate.lean` proves equality to the complete proof-only
integer-commuted view; `KernelRewrite` connects actual full execution, including
failures. The independent mathematical softmax and existing native statements
use those same quotations.

Proofs use the source register `N` and the actual caller-binding order.
The helper's original `tl.cdiv(N, TILE_N)` is retained: its evaluated signed
integer expression is connected to the previous-multiple formula under the
existing positive tile-width condition. This does not add an input restriction
or a floating arithmetic identity.

The domain remains rank-two, contiguous, strided FP32 CUDA Tensors with no
unresolved negative bit, numel below 2^31 and aligned nonempty data pointers.
Storage offsets and irrelevant singleton strides are admitted. Empty matrices
return fresh empty output without a launch. All FP32 words, including NaNs and
infinities, are admitted. Output contracts describe the ordinary separate
contiguous output buffer; they do not narrow the input domain.

The original source can produce spurious NaNs from inactive or negative-infinite
lanes, and at zero columns its tail starts at a negative offset. The selected
repair retains the documented `new_m == -inf` and `inp == -inf` source guards,
widened integer addresses and empty-input branch.

## Public kernel results

- `TensorBridge.softmax_kernel_correct` constructs successful execution of every
  launched row and proves ordinary mathematical softmax
  `exp(x[row,column]) / sum(exp(x[row,j]))`. It reads the original Tensor's
  logical elements, writes physical output Tensor indices, preserves raw cells
  outside the row and preserves the input. Empty columns perform no launch.
- `TensorBridge.softmax_kernel_fp32` constructs successful executions of the source
  and genuine index-commuted candidate, proving equality of their complete
  states, every observed output word and exterior raw-cell frame. Both runs
  use the same native primitive operations and compiler reduction/operand/FMA
  profile. No floating commutativity or split-multiply/add-versus-FMA law is
  assumed.

`softmax_optimize_index_commuted.py` is a real candidate with exactly four
changes: `pid_m * N` becomes `N * pid_m` in the four loop offsets. Its floating
statements, helper, comparisons, masks, casts and stores are identical. The DSL
candidate performs those same integer rewrites. `KernelRewrite` reuses the
original contextual signed-loop rewrite proof, eliminates the empty numerical
assumption table and proves equality of actual `FP.Structural.exec` results.

`FPExecution`, `FPComplete` and `FPFrame` independently construct successful
execution of the actual typed prelude, both input loops, guarded updates,
reductions and both output loops. Signed full-loop multiples and the single
masked tail establish the exact row frame. Thus the result does not rely on
equality of failed runs or a manually assembled floating graph.

`TensorBinding.source_load` connects the source storage cell to the proved
native byte address and then to the original Tensor logical coordinate. The
real proof composes these equalities inside its softmax operands. The native
proof composes them to identify the actual physical provenance leaves with a
logical Tensor view before invoking the successful typed execution theorem.
Actual storage laws provide bounds and the input span, including offset views.

The native model reuses `softmax_flaggems.Proofs.Native`'s `Term`, `Shape`,
valid reduction trees, primitive `Instructions`, and `Schedule`/`Fusion`
observer. A small syntax-only `NativeOrder.arrange` adapter also records the
observed mass-add operand reversal before the shared observer selects FMA and
multiplication order. Compiler choices inspect operation shapes, not numerical words.
Explicit source precision, loads/casts, operand order and eligible adjacent
FMA contractions remain represented. The two actual source comparisons use
observed equality to the concrete negative-infinity word.

Ordinary Tensor storage laws, Python/DSL correspondence, the chosen native
compiler profile, primitive instruction conformance and FP32 word packing are explicit modeling
boundaries. These are not deferred numerical experiments or kernel-output
premises. The proof does not identify FP32 output with the mathematical Real
formula, nor assert that arbitrary different compiler profiles agree.

The historical host and independent graph results remain optional in
`Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`. Neither public entry
imports those wrappers. Their local `Profile` describes exp lowering as the
FP32 log2(e) multiply (`0x3fb8aa3b`) followed by `ex2.approx`; historical compiler
observations are supplementary evidence rather than assumptions about whole
kernel output.

## Validation

```sh
python bench/repairs/tritonbench_g/softmax_optimize/check_contract.py
python bench/repairs/tritonbench_g/softmax_optimize/check_index_candidate.py
```

The historical CPU checks reported 18 admissions, 14 rejections and three
counterexamples, plus 16 candidate complete-storage comparisons. These are
historical observations, not fresh verification results for this migration.
Fresh source checks, actual Lean replay, trust audits and any unavailable
optional checks are recorded in the current `bench/review-checkpoints/` receipt.
The body drift tests reject missing casts/guards, changed loop bounds, changed
pointer dtype profiles, changed masks/stores and a modified helper formula.
No CUDA or new numerical experiment is implied by these source/proof checks.

Original Python SHA256:
`230ad985c624f38c13f81b6d82f7cba854ad6ddc418c983caf55e4fee84c4d38`.

## Public IO and supporting proofs

`IO.lean` declares the actual selected source, one N-element input row, one
N-element output row, no scratch, and the independent formula
`exp(x[j]) / sum_i exp(x[i])`. `Correct.softmax_row_correct` uses the shared
`UKernelIO.RegionalImplements` judgment. The row is complete even when
`Config.tile N < N`; all full/tail iterations of both passes remain covered.
The conditions are exactly separate input/output regions and positive N. No
restriction on the raw program ID or N relative to the tile was added.

`Proofs/IOSafetyBodies`, `IOSafetyLoops` and `IOSafety` prove actual source
load/store safety from those row bounds, following its evaluated predecessor
states. `Proofs/IOCorrect` composes that proof with the existing complete
`SourceIO.selected_tile_io` execution, all outputs and raw-cell exterior frame.
The historical raw result, including unchanged PIDs, grid dimensions and undef
memory, remains in `Proofs/KernelRealResults.lean`.

`FPEquiv.softmax_source_equiv` is the standard whole typed-source equivalence
for the original and index-commuted programs, with an empty numerical
assumption table. The complete successful native execution to one common
state and its full row frame remain in `Proofs/KernelNativeResults.lean` under
the original reduction, primitive instruction and scheduling model.

`TensorBridge.softmax_kernel_correct` and `TensorBridge.softmax_kernel_fp32`
retain their complete Tensor statements, storage observations and empty-column
no-launch behavior. They use the same underlying program proofs. Public and
supporting proof imports do not depend on a reader entry or TensorBridge.
Source, Python guards, precision and native modeling boundaries are unchanged.
The public-IO pass performed targeted Lean builds and axiom/surface checks;
it added no numerical assumptions or GPU experiment.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
