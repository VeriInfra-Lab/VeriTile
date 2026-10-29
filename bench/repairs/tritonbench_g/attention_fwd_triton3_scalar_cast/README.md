# `attention_fwd_triton3_scalar_cast`

This separate candidate repairs the ordinary-window `NK=1` JIT failure by replacing `N_CTX.to(tl.int64)` with `tl.cast(N_CTX, tl.int64)`. The admission guard also changes the CUDA grid-y bound from `Z*H <= 65536` to `Z*H <= 65535`. These are the only Python changes from the pinned predecessor. `NK=1` remains admitted. No total-grid-product condition or segmented launch is added.

The delivery is a **kernel certificate**: complete modeled quotation execution, independent Real and native results, actual shared Tensor storage observations, all padded O/M/L stores, and memory frames. The optional returned-Tensor copy corollaries are not prerequisites for the kernel result. Host dispatch, resource delivery, compiler correctness and Torch implementation verification are outside this certificate.

## Public entries

| Entry | Result |
| --- | --- |
| `correct` | Shared `AlgorithmProgramIO.Implements` for the actual complete ordinary grid and finite six-input Real formula; source access safety, all active padded M/L/O outputs and exact frame. `RealIO.lean` exposes source, windows, masks, carry aliases and zero-undef convention. |
| `Auxiliary.whole_real` / `Auxiliary.canonical_flat_io` | The unchanged canonical flattened ordinary-grid theorem, plus an explicit finite-input snapshot/formula connection. Existing `Flat.program` transports actual safe source execution with `FlatAlloc.exec_flatten`. |
| `TensorBridge.complete_real_tensors` | The same Real execution starts from a physical Real heap at the actual six shared Tensor addresses. Returned kernel O workspace elements at ordinary query coordinates equal the independent Real target. `RealTensor.six_inputs` binds each of the six input Tensor roles. |
| `FPEquiv.key_mask_equiv` | Actual original versus key-tail-mask CSE, `original ≡[[]] candidate`; both complete safe ordinary grids, all padded FP32 O/M/L graph cells and both exterior frames. |
| `fp_equiv` | Shared `ArrayProgramIO.Observes` for the complete actual native grid: six arbitrary named input arrays, finite source-free access windows, owner-specific native words and exact typed-cell frame. `NativeIO.lean` exposes source precision and raw scale ABI. |
| `Auxiliary.fp_equiv` | The unchanged complete raw-byte statement: publication, output words, exact byte frame, supported graphs and center provenance under the original nine same-port primitive laws. |
| `TensorBridge.fp_equiv` | The exact complete native Tensor application, including actual padded O/M/L Tensor elements and their physical frame. |
| `Native.six_inputs` | Exact initial FP32 words from every actual shared Tensor role, including continuation O/M/L. |
| `TensorBinding.logical_lane` / `padded_tensor_readback` | Every store lane, including BM-padded query rows, has a derived bounded logical O/M/L Tensor index. No coordinate or numerical-output equality is assumed. |
| `Bytes.tensor_denominator_preserved` | END preserves every logical L word, including signed zero and NaN payload bits. |

Lean namespace prefix: `VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast`. The filenames identify the modules; the mathematical specification `correct` and ordinary native theorem `fp_equiv` are in that root namespace. `Misc/Auxiliary.lean` indexes the preserved operational results; `Misc/TensorBridge.lean` retains all five complete application statements.

## Targets and complete execution

`TARGET.md` fixes the targets and domain before this implementation. The Real target is the predecessor's independent initial-array formula `Whole.result`, with its documented bottom/zero representation and total Real division and logarithm. It is not an IEEE approximation theorem. `RealTensor.logicalValue` interprets an ideal Real heap at actual shared Tensor addresses, separately from the raw-bit native interpretation.

The native target is `Native.targetBytes`, defined from the independent six-input recurrence in `Reference.lean`, its complete grid fold, and explicit native instruction evaluation. It contains no quotation, execution function, or correctness theorem. It is not defined by the completed source execution. The native recurrence retains raw continuation, signed window bounds, both window masks, all-masked behavior, zero iterations, empty grids, exact store order, and every BM-padded row. It preserves the actual FP32 scale ABI word separately from the Real parameter, including negative zero.

`Prelude`, `Window`, `Body`, `Range`, `Epilogue`, `Program`, and `Grid` prove successful execution of every statement in the readable quotation and its complete grid. The grid threads actual predecessor memory through every program and resets registers. Generic backend operations in this internal execution proof do not become arbitrary dot/reduction assumptions in the public numerical interface.

`NativeExpr` stages the native source graph and `NativeEval` interprets it using exactly the inherited `Raw.Ops` ports: FP32 add, subtract, multiply, divide, maximum, exp2, log2, FMA, and one coordinate of a fixed-K8 TF32 matrix instruction. Dots are finite consecutive K8 instruction chains. Reductions use explicit ordered trees, with coverage of each key lane. Physical consumers retain their individual owner routes and M/L byte writers.

The only floating comparison in the actual GPU source is `tmp != tmp`. Its staging token carries an alpha node; evaluation performs raw-bit NaN classification independently at each owner. `NativeTags` proves the required center provenance for raw M inputs, every recurrence step, normalized M stores, and the complete grid. `NativeSupported` and `NativeReachability` prove that reachable graphs contain no unsupported callback, only the three ordinary source literal values, and only reductions whose width equals BN. The generic evaluator's unmatched cases are therefore unreachable for the completed candidate.

## Explicit model boundaries

The same nine primitive/ISA ports on each side of `Raw.LocalLaws` are a model boundary. No whole-dot equality, reduction reassociation, tensor-output equality, successful-execution assumption, native-to-Real equality, or pending numerical-transformation experiment is required.

`Raw.Schedule` supplies matrix kinds, source/compiler fusion choice, ordered trees, owner routes, and byte writers. The theorem works for the supplied structural schedule. Realization of that schedule by a particular compiler/device is external; the theorem does not claim that arbitrary schedules describe actual compiler output. The selected primitive meanings must match the intended FP32/TF32 ISA and exp2/log2 implementation.

`TensorBinding.Admission` uses actual `Torch.FP32.Tensor` objects, ordinary Tensor API storage/index laws, checked metadata, and pointer/allocation bindings. Alignment and absence of negative views are derived from the checks. Native publication reads completed source memory, writes exactly active physical bytes, and preserves exterior bytes. Tensor/allocator APIs, launch resource delivery and synchronization are ordinary external boundaries. The scale input is the actual bound FP32 ABI word; there is no claim that the Python wrapper classifies every possible Python scalar argument.

Candidate-local `TensorReturn.API`, `TensorBridge.returned_tensors`, and `Native.returned_output` additionally describe the ordinary END slice/contiguous/FP32 copy through storage-word copying and actual Tensor identities. They contain no kernel-result law. They are optional host API consequences, not part of the kernel-only acceptance prerequisite.

## Evidence and preservation

Evidence is under `.veritile-cache/kernel-evidence/attention_fwd_triton3_scalar_cast/`:

- `source-fix-controls.json`: actual candidate wrapper captures real CPU Tensor arguments under explicit CUDA-admission/context/submission facades; actual source heuristics and the installed real JIT binder compile ordinary-window constexpr `NK=1` and runtime `NK=2` for SM90. The guard admits grid-y 65535 and rejects 65536 before submission. TTIR, TTGIR, LLVM IR, and PTX are pinned. No GPU or kernel interpreter executes.
- `quotation-controls.json`: all normalized statements of both GPU functions match the readable quotation after checked helper inlining and explicit representation adapters. Integer widths and the raw-scale interpretation remain separately documented boundaries. This static comparison is not a compiler proof.
- `native-source-audit.json`: floating-comparison, literal, reduction and dot source audit, complementing the Lean reachability proofs.
- `delivery.json` and `candidate-manifest.json`: final source, proof, receipt, evidence and dependency hashes, with the exact reviewed local-first build driver.
- `baseline-manifest.json` and `preservation.json`: all 86 predecessor files and their snapshots, plus both frozen defect diagnoses, remain byte-for-byte unchanged. The old accepted diagnosis does not become a correctness acceptance for this separate candidate.

This IO-format pass preserves Python, `Kernels.lean`, `Contract.lean`, numerical boundaries and historical evidence. Targeted builds and source/dependency hashes are recorded under `.veritile-cache/public-io-format-20261020/attention_fwd_triton3/`. No new independent replay or GPU measurement is claimed.

The source-independent formulas and IO definitions formerly in the historical original `Correct.lean` were moved, with identical names and definition bodies, to `bench/tritonbench_g/attention_fwd_triton3/Proofs/MathematicalSpecification.lean`. The historical public specifications still import them. The inherited repair geometry imports that helper directly, so supporting proofs no longer import reader-facing Correct/FPEquiv entries.

`Correct.lean` contains one shared-IO specification; `Misc/NativeCorrect.lean` retains the ordinary single-source native theorem. Both include statement-surface, axiom and independent-formula audits. Actual native source access safety is proved in `SourceSafetySteps`, `SourceSafetyGeometry`, `SourceSafetyTail` and `SourceSafety`; it follows the real branches and signed loop, including zero iterations and checked/unchecked KV tails. `NativeBindings` connects finite windows to those accesses. `NativeLocality`, `NativeFrame` and `NativeImplementation` reuse the existing complete source execution to prove all native outputs and their frame.

The native array interface retains arbitrary decoded input arrays, with no allocation claim for unused suffixes. Source and target share only the independent array values, geometry and original primitive model. Raw FP32 input words are the original `.input region word` instance of this symbolic interface; the preserved raw-byte theorem also retains reachability/provenance facts for that instance. This introduces no numerical-transformation assumption.

All complete admitted Tensor applications remain `TensorBridge.whole_real`, `TensorBridge.correct`, `TensorBridge.fp_equiv`, `TensorBridge.complete_real_tensors` and `TensorBridge.returned_tensors`. Their statements are unchanged; they use the same established full execution foundation through the preserved RawKernel endpoints. No arbitrary-flat-state extension or additional layout restriction was introduced.

The prior single-source native results remain ordinary theorems in
`Misc/NativeCorrect.lean`, with their original names, domains and proofs. The
main pair is now `FPEquiv.key_mask_equiv`, comparing the original complete
quotation with `Quoted.keyMaskCSE`. The historical reciprocal-normalization
pair was considered first; its FP16 storage does not preserve this selected
FP32 entry. The new candidate has a direct common-subexpression-elimination
purpose: each uneven-key loop iteration computes
`key_mask = (tl.arange(0, BLOCK_N) + start_n) < NKV_CTX` once, then reuses it for
both score masking and probability zeroing. This removes the repeated index
generation, addition and comparison. No performance improvement is claimed
without measurements.

Both complete programs are ordinary `triton` quotations in `Kernels.lean`.
The original complete quotation, `Kernels.source`, Python, Contract and scalar
cast/grid-y repair are unchanged. All floating operands, FP32 storage and
casts, fixed-K8 semantics, compiler fusion choices, raw scale ABI, sliding and
complement windows, tail branches, signed loop, continuation, zero iterations,
ordinary grid order, and padded L/M/O store order remain unchanged.

`CSEParts`, `CSEBody`, `CSERange`, `CSEProgram` and `CSEGrid` prove the actual
candidate execution against the existing independent six-input recurrence.
`CSESafety` proves its actual accesses from the unchanged windows; `CSEIO`
connects both source schedules to the shared named-array interface. The main
pair proves exact equality of the native symbolic output graph, which retains
the source precision and ordered primitive/ISA nodes. Consequently its
owner-specific word observations agree for every supplied native interpretation
and raw scale word. Six arbitrary input arrays, NaNs, signed zero, all padded
outputs and both full typed-cell frames remain covered; no finite-word premise
or numerical rewrite is introduced. `R=[]` records an empty experimental rule
set within this fixed source interpretation.

The existing independent Real and Tensor results retain their original scopes.
Campaign validation is recorded in
`.veritile-cache/finish-pending-105-20261021/attention_fwd_triton3/`; it adds no
host, hardware, GPU or numerical-experiment obligation.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.
