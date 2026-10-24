# Scalar ABI checked attention candidate

This separate candidate adds one Python guard: reject a raw positive Python
integer `sm_scale >= 2**64` before Triton's argument binder. It preserves the
original scalar object. The three JIT kernel bodies, their launch options,
Tensor checks, ordinary shapes and supported head dimensions are unchanged.
The original defect diagnosis and historical acceptance are separate records.

The public selected precision is the wrapper's FP32 branch, with the explicit
FP16 probability, score-gradient and output casts retained. The wrapper's FP16
compatibility branch remains present; these new certificates do not certify it.

## Public proof surface

The bound native auxiliary theorem is `FPEquivBound.kernel_native` in
`Misc/NativeCorrect.lean`; see
`README-v2.md` for the current finite shared IO organization. The unbound
`FPEquiv.kernel_native` below is also an ordinary theorem in
`Misc/NativeCorrect.lean`, retaining its original generic structural scope.

- `Correct.kernel_real`: complete forward, preprocess and backward execution
  of the shared readable DSL, causal-attention output, independent Q/K/V VJP
  tensors, and the complete typed-memory frame.
- `TensorBridge.tensors_real`: the same result through actual `Torch.FP32.Tensor`
  storage offsets and logical indices, with returned object identities and
  the scalar-address storage frame.
- `FPEquiv.kernel_native`: the shared DSL interpreted with scalar provenance,
  native reductions, fixed-K8 MMA seeds/order, explicit casts, division and
  exponential instruction graphs; complete execution yields an independent
  input-only native target and frame.
- `TensorBridge.Unbound.tensors_native`: actual Tensor object identities, storage readback
  and frame for the native result. `TensorStorage.wordInput_logical` connects
  source input words to the existing Torch FP32 logical-value API.

`Misc/NativeCorrect.lean` is imported explicitly by optional consumers.
`FPEquiv.lean` and `FPEquivBound.lean` declare no main two-program equivalence
specification and do not import the optional native entry.

The Real theorem quantifies arbitrary real input values and every complete
legal CTA permutation. Saved normalizers, maxima, preprocessed gradients and
Delta are derived by execution. Initial scratch is arbitrary; only DQ is
zero-initialized. The Tensor entry performs that initialization and imposes no
execution, final-output, mathematical-gradient, or native-target premise.

`SourceKernels.lean` contains the shared readable quotations; `Proofs/FPSource.lean` is a compatibility import preserving full declaration names. The Real and
native proofs both execute these through `FP.Structural.exec`. The targets in
`RealTensors.lean` and `NativeTarget.lean` do not reference execution or the
source loop recurrence. Native `stores` is an internal execution lemma, not the
public specification target. `NativePipeline.lean` proves the complete native
chain. `TensorStorage.Objects` contains only Tensor API/shape/device facts and
fresh-view address separation; it allows read-only input aliasing. The incoming
DO view denotes the wrapper's ordinary contiguous copy.

Raw integer `1` selects constexpr-one lowering. Float `1.0` retains runtime
scale multiplication and backward FMA. Loaded word `1` remains `Stage.ready`,
so it cannot masquerade as a scalar literal. Preprocess retains product order
`1,0,2,3`, the nested FMAs, and its exact shuffle tree. Coverage excludes padding
without reordering a floating-point reduction.

## Evidence and boundaries

The authoritative candidate evidence lives in
`.veritile-cache/kernel-evidence/triton_attention_scalar_abi_checked/`.
`delivery.json` is the frozen manifest; parent integration and acceptance remain
separate. All candidate Lean source receipts and objects are hash-checked.
Public theorems pass axiom auditing with only the standard Lean base.

`scalar-abi.json` records 9 accepted and 11 rejected scalar controls using the
actual source JIT binders and extracted scalar-only guard. It does not run the
Tensor host wrapper. `scalar-native-profiles/receipt.json` records four offline
compilations of actual source: integer-one/runtime-one × forward/backward,
Triton 3.7.1, CUDA sm90, D16, one block. TTIR/TTGIR/LLIR/PTX are retained.
No GPU execution or native numerical experiment is claimed.

The native proof is parametric in scalar/fixed-K8 native instruction semantics,
scalar literal conversion and reduction owners. Ordinary correspondence of
the Python source, raw scalar conversion, compiler-selected instruction profile,
Tensor API and actual runtime to that interpretation remains an external
binding boundary. The bounded compiler observations do not establish universal
compiler conformance. There are no numerical equivalence leaves, FP-to-Real
claims, result assumptions, or newly deferred numerical experiments in these
certificates. Actual execution resources and grid-product overflow remain
outside this repair; no per-device resource guard was added. The storage bridge
is a scalar-address view interpretation, not a new byte-heap or host/compiler
verification project.

`FPEquiv.grid_equiv` proves the complete actual `original ≡[[]] shifted`
forward/preprocess/backward pipeline. The selected `FPEquivBound.lean` imports
this primary. The candidate changes only the preprocessing row base from
`program_id(0) * 128` to `program_id(0) << 7`, an integer address strength
reduction with no measured performance claim. Its complete ordinary quotation
is in `CandidateKernels.lean`; forward/backward reuse the unchanged original
source transparently. `triton_attention_shifted.py` contains the full executable
pipeline with the same ABI, launch flow and fixed-128 preprocessing guard.

`Proofs/Pair.lean` keeps the existing MaterializedProgramIO and shared
MaterializedInvocation runner, encoding, per-store replica lowering and
observation. Its task-local implementation view changes only actual invocation
lists. The proof requires both complete schedules to be source-safe and
successful from the same exact input cells, agrees on Out/DQ/DK/DV and retains
both complete exterior frames including the original scratch footprint.
Arbitrary complete schedules for all three stages and the original zero-initial
DQ domain are retained. FP32 inputs/operations, internal FP16 casts, reductions,
scalar specialization, owner choices and fixed instruction ports are unchanged.
All word patterns remain admitted. The shift is strictly exact; there are no
experimental numerical rules or pending numerical experiments for this pair.

`check_pair_source.py` restores only the preprocess shift/static assertion and
name, then checks exact AST equality of the complete Python module. The source
relocation, candidate and pair/public builds plus fresh joint mathematical,
program-pair and optional native/Tensor entry check passed. Historical
single-source native/scalar-binding certificates retain their prior scope.


Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/ReviewEntryV2.lean](Misc/ReviewEntryV2.lean), [Misc/ReviewEntry.lean](Misc/ReviewEntry.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.

[Misc/RealFacts.lean](Misc/RealFacts.lean) retains the optional
`Correct.attention_vjp` mathematical proof alias. The complete
forward/preprocess/backward schedule remains the main kernel functionality.
