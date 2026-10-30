# Persistent matmul scalar-cast candidate

This independent sibling changes exactly nine runtime scalar conversions from
`value.to(tl.int64)` to `tl.cast(value, tl.int64)`. Triton's real binder may
specialize scalar one to a Python integer; the functional cast accepts that
value. Tensor-valued PID/arange casts, both wrapper definitions, validation,
device context, empty/K=0 behavior, and launch choices are preserved.

| Storage | BM | BN | BK | GROUP_M | stages | warps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 128 | 256 | 64 | 8 | 3 | 8 |
| E4M3FN | 128 | 256 | 128 | 8 | 4 | 8 |

The historical benchmark and predecessor repair remain available. This scalar-cast
version is the selected task source in the current corpus.

## Evidence

The following historical diagnostics were recorded under the former workspace
cache. They are not fresh validation of the literal-body migration, and their
original cache files are not required to build these sources or proofs:

* `source-delta.json`: nine replacements; reversing them reproduces the entire
  frozen Python module byte for byte and AST for AST.
* `binder-report.json`: eight actual JIT-binder/offline SM90 compilations;
  both original unit-size and 2×2 witnesses, plus K=1 and irregular-tail
  controls for both storage profiles. Scalar-one constexpr specialization is
  retained. The report pins TTIR, TTGIR, LLVM IR, and PTX artifacts.
* `cpu-report.json`: 22 production-AST diagnostic cases and nine guard
  rejections, including persistent iterations, final groups, ordinary
  strides/offsets, broadcast, A/B aliasing, device restoration, empty output,
  K=0, and whole input-storage frames.

The binder compilations do not execute a GPU. The CPU facade uses NumPy/Torch
diagnostic arithmetic and is not evidence of native floating-point equality.

## Formal chain

`Kernels.lean` contains the original Python function body verbatim in four
`triton { ... }` quotations: FP16/E4M3 original and row-index candidate. Caller
pointer types, runtime scalars and constexpr metadata stay outside each body.
`jit-correspondence.json` checks every statement and the external call metadata;
`check_pair_source.py` checks the complete Python modules and permits only the
one row-index product commutation. The tuple casts, Boolean ceiling expressions,
signed nested loops, live dot and output dtype branch remain visible.

The operational proofs use the shared `FP.Structural.run`, signed dynamic
ranges, masked stores and `FP.Structural.exec`. Original scalar assignments and
control registers are established by the prefix and preserved across both
loops. No equality to the historical rewritten AST is assumed. The full Real,
FP and IO judgments are re-proved for the copied source itself.

The proof files separate prefix, tile setup, inner K loop, output store,
persistent outer loop, and complete grid composition. `Targets.result` is
defined from caller physical storage and mathematical row/column coordinates.
It contains no PID, execution, or handwritten event interpreter.

`Public.invoke` preserves the checked wrapper's post-allocation empty,
zero-K, and launch branches. `TensorBinding.Laws` gives ordinary actual object,
metadata, pointer, and independent logical-index observations. `Physical`
transports the executed region result to that Tensor's backing storage.

The primary mathematical endpoint is `kernel_correct` (independent Real K-sum).
`kernel_fp_equiv` is an ordinary auxiliary theorem in `Misc/NativeCorrect.lean`
against the independent native instruction reference. The shared `ProgramIO`
retains two strided logical matrix inputs, one typed matrix output, the complete
persistent launch schedule, and no private scratch. `Correct` states the
independent finite K-sum. `NativeCorrect` keeps the
existing four ordered ISA stages per K block, incoming FP32 accumulators,
masked padding and final FP16/FP8 conversion for every input word.

The shared `Implements` judgment proves actual source access safety, successful
execution, every typed output and the complete exterior frame. `Proofs/`
connects logical channel bindings to the existing full-grid theorem and checks
the reached load/store addresses through both source loops. Input A/B aliases,
arbitrary admitted strides, empty matrices and K=0 retain their original
scope. Raw allocation and stride premises remain explicit; raw state/read/run
propositions have moved to `Proofs/KernelRealResult.lean` and
`Proofs/KernelNativeResult.lean` without changing those theorem statements.
Raw allocations carry only storage identity, offsets, capacities and pointers.

The source-to-native theorem uses the same local scalar-conversion and named
MMA ports as before. These are primitive-model boundaries, not added numerical
rewrite assumptions. This format revision runs no GPU or numerical experiment
and does not change the previously recorded deferred validation status.

`TensorBridge.correct` and `TensorBridge.fp_equiv` retain the complete original
checked-wrapper applications as ordinary theorems, including the exact returned
Tensor, every logical output, the entire output-storage exterior, all physical
input-storage elements, and the unchanged empty and zero-K branches.

See `SOURCE_BINDING.md` and `NUMERICAL_BOUNDARY.md` for the exact boundaries.
Build receipts and the final delivery verifier determine acceptance; this
README alone is not a build or completion certificate.

`FPEquiv.source_pair_equiv` now compares the actual complete selected source with
`matmul_persistent_triton_reordered.py`, under an empty experimental rule set.
It transports the existing predecessor's row-index multiplication
canonicalization to the selected common row calculation:
`start_m = pid_m * BLOCK_SIZE_M` becomes `start_m = BLOCK_SIZE_M * pid_m`.
This puts the static tile width first for integer address-expression matching;
no measured speed improvement is claimed. The complete Python AST and ordinary
quote differences contain exactly this one operation, with validation, both
storage configurations, casts, masked pointers, dot carry, scheduling and
wrapper branches preserved.

The main comparison uses the shared complete `ProgramIO` syntax. Under the
original input binding and logical IO bounds it concludes both actual trace
safety judgments, successful executions, equal complete output cells and both
exact exterior memory frames. `Proofs/PairIO.lean` proves the source scatter
frame independently of a numerical target or zero-conversion identity.

`Proofs/Pair.lean` proves exact signed-integer evaluation equality, then lifts
it through the whole tile body, dynamic persistent range and complete PID grid.
Successful execution is constructed separately using the existing source loop
and register invariants, without assuming a whole result or a numerical
zero-conversion identity. Final states agree in every typed memory cell and
register, and reached trace-safety conditions are equivalent.
`Pair.implements_transfer` preserves all independently proved logical outputs
and exact raw frames. Floating operations and input-word domains are unchanged;
there are no strict or experimental floating rewrite dependencies.

The optional `Misc/CandidateCorrect.lean` applies this source equality to
`Public.invoke`, including the actual returned output object, all physical
memory observations and the empty/K=0 no-launch branches. The existing Real,
native and Tensor auxiliary statements and their full domains remain intact.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/Audit.lean](Misc/Audit.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
