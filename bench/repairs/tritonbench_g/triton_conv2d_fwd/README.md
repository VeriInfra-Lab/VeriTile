# triton_conv2d_fwd repair

The repair adds `other=0.0` to the two masked dot-operand loads. The original omission permits undefined channel-tail and spatial-padding values to contaminate an active result. `Proofs/Witness.lean` checks two admissible undefined-value witnesses: channel-tail result 32 instead of 1, and padded-spatial result 288 instead of 32. These are semantic counterexamples, not GPU observations.

The mathematical target is grouped cross-correlation, with dilation one and no bias:

```
Cg = C/G, Og = O/G
OH = (H + 2*PH - KH)//SH + 1
OW = (W + 2*PW - KW)//SW + 1
Y[n,o,y,x] = sum(c<Cg, h<KH, w<KW,
  Xpad[n,(o//Og)*Cg+c,y*SH-PH+h,x*SW-PW+w] * W[o,c,h,w])
```

The original Python remains unchanged. The repaired kernel preserves every pointer expression and mask, the h/w/channel loop order, 128×32×32 tiles, both precision flags and defaults, default compiler options, and the direct grid `(ceil(N*OH*OW/128), ceil(O/32), G)`. Redundant group-local feature programs remain. The wrapper retains its fresh FP32 `torch.empty` and returns that object. It adds validation and an empty-batch return; it performs no copy, normalization, snapshot, segmented launch, or copyback. The kernel docstring now gives the actual grouped weight shape `[O,C/G,KH,KW]`.

## Checked contract

- Input and weight are resolved rank-four, positive-stride FP32 CUDA tensors on the same current CUDA device. The wrapper preserves the original current-device launch behavior.
- Noncontiguous positive-stride layouts and arbitrary actual storage offsets are admitted. Read-only input and weight may share storage or be the same object.
- N may be zero. C, H, W, O, KH, KW, SH, SW, and G are positive; PH/PW are nonnegative; C and O are divisible by G. The exact weight shape and positive OH/OW are checked.
- Scalar integer arguments use strict Python `int` classification; precision flags use strict `bool` classification.
- Every accessed four-byte element must lie in the queried actual backing storage. Empty input has no input-storage access.
- The explicit conservative signed32 envelope bounds scalar arguments, padded coordinates, and every formed pointer intermediate. It is checked on actual dimensions and strides.
- For nonempty calls, CUDA grid Y and Z must be at most 65535. Grid X follows from the signed envelope. There is no segmented compatibility launcher.
- A successful fresh `torch.empty` supplies the output allocation identity, dense metadata, storage bounds, and physical separation from the input and weight allocations. Allocation/launch resource success and the Torch storage ABI remain ordinary runtime boundaries.

`Contract.lean` records the metadata observations and executable guard. `Proofs/Contract.lean` proves acceptance/rejection equivalence, derives source safety, and proves the unchanged grid fits the checked limits. The normalized `Shape` record is bound to actual queried shapes/strides, group products, and exact positive-denominator floor-division formulas.

## Kernel-only formal results

`Kernels.lean` now contains readable quotations of the repaired shared typed DSL. `KernelSource.kernel` selects the FP16 constexpr specialization of the same Python kernel. Both arithmetic interpretations execute this selected source. The historical handwritten circuit declarations remain unchanged; the previous public files and all initial task sources are preserved under the delivery's hashed `initial-snapshot/`.

The `KernelHalf*` and `KernelDefault*` chains prove successful shared-interpreter execution through every h/w/channel loop trip, explicit zero-filled loads, optional FP16 operand casts, two-argument dot, outer accumulator add, final cast, and masked stores. `KernelBinding`, `KernelGrid`, and `KernelLaunch` derive exact coordinates, addresses, unique stores, full original-grid coverage, evolving-state execution, input preservation, and the complete region frame. The redundant group-local feature programs and empty-batch grid remain intact.

`KernelOperands` proves that every active typed-pointer address equals the original complete signed source address at the actual strides. Negative inactive spatial terms need not have the same typed-pointer representation; they are masked and never read. The unchanged `Lowering` proofs retain the original signed32 formed-expression and physical-address safety facts.

`KernelNumeric.reference` is an independent recurrence over initial logical arrays, with no source AST, register state, output storage or execution result. Its internal generic numerical interface is instantiated by the public entries. No arbitrary whole-dot callback is a public premise.

For ideal Real arithmetic, `KernelReal.reference_real` proves the independent grouped finite sum above. The half conversion is ideal identity and the fixed K32 operation is a closed finite sum. `Correct.lean` exposes `correct_real` as `Spec.Real ((ConvIO.realIO s flags).Implements (ConvIO.mathematical s))`. The shared IO judgment proves full source-grid safety, successful execution, logical output values and the exterior frame. `TensorBridge.correct_real` retains the separate actual-Tensor readback statement. It is an exact ideal-Real statement, not a floating-point error bound.

For native arithmetic, each source dot remains a pending node until its following accumulator addition. The lowering interpretation supplies the old accumulator to the first emitted instruction, matching the observed compiler carry absorption. This preserves the source dot/add structure without asserting equality between rounded split and fused arithmetic. `KernelNative.lower32` expands to 32 scalar FMAs, four K8 WGMMA lane steps, or two K16 steps. FP16 conversion is explicit and takes precedence over TF32; TF32 input treatment belongs to the K8 primitive because the checked PTX emits no separate TF32 conversion. All padded numerical operations remain, including for NaNs, infinities, subnormals and signed zero.

`Misc/NativeCorrect.lean` retains the ordinary theorem `fp_equivalent` as the shared `ProgramIO.Implements` relation from the same complete DSL grid to the independent native target. The exact actual-Tensor readback, every output byte, output-exterior frame and preservation of both input backing allocations remain in `TensorBridge.fp_equivalent`. Its only numerical ports are one scalar FMA, one scalar FP32-to-FP16 RN conversion, and one fixed output lane of the literal SM90 `m64n32k16.f32.f16.f16` or `m64n32k8.f32.tf32.tf32` instruction. Each lane port takes only its fixed-length operand row/column and scalar carry. No configurable dot, convolution, output tensor or heap port occurs.

The ordinary theorem `fp_default` in `Misc/AuxiliaryFPEquiv.lean` specializes the same kernel to its production-default flags and closes its reference to the explicit IEEE RN-even FP32 FMA circuit. Only the local scalar FMA equation is required; unused conversion/tensor ports do not affect this result. Its complete contract and name are unchanged; `Misc/Review.lean` imports this auxiliary result explicitly.

`KernelTensor` binds actual shared `VeriTile.Torch.FP32.Tensor` objects, their storage identities, offsets, physical indices and queried metadata. Its element capacity is the original untyped byte capacity divided by four; unused tail bytes remain admitted and remain part of the full backing frame. Independent review identified and corrected an initial overstrong exact-divisibility assumption before freeze, using a legal one-FP32-element view over a five-byte backing allocation. `KernelInputs` proves actual input/weight Tensor readback. `KernelStorage.image_initial` proves the initial physical image is the caller heap; `image_write` and `fold_projection` prove each DSL store is four concrete byte writes. `KernelStores.program_bytes`, retained as `KernelNativeResult.physical_execution`, applies that connection to every actual program and preceding state. There is no assumed final-byteheap agreement. Existing successful-allocation metadata remains an ordinary boundary; no new host or allocator implementation proof is introduced.

The native result is an explicit SM90 lowering model. Actual binder/offline-compiler profiles support the selected graph and layout mapping; they do not prove universal compiler correctness or arbitrary architecture conformance. Primitive numerical semantics, successful allocation/launch, and the Torch storage ABI remain stated model boundaries. No GPU execution, native numerical experiment, or native-to-Real equality is claimed.

## Validation and provenance

The delivery pins all 50 task Lean sources, dependency-order build receipts, resolved import objects, and public axiom/noncircularity audits. The canonical repaired Python and admission domain are unchanged from the initial snapshot. Importing the shared Triton parser requires qualifying the existing Boolean literals in the older Contract/contract-proof/software files as `Bool.true`/`Bool.false`; normalized snapshot comparison checks that these edits change no guard or numerical expression.

Inherited evidence is retained and identified separately: 32 actual-wrapper/Triton CPU-interpreter cases, 13 rejected-domain cases, and 12 actual-JIT-binder SM90 offline profiles. The original omitted-`other` witnesses remain checked: channel-tail result 32 instead of 1, and spatial-padding result 288 instead of 32. These are semantic counterexamples, not GPU observations.

New controls execute 16 layout/flag combinations through the actual source wrapper and Triton CPU interpreter: input spatial transpose, weight spatial transpose, channels-last grouped tensors with nonzero offsets, and a 144-spatial-element case exercising both row fragments and a tail CTA. They check complete outputs, full input backing preservation, metadata and freshness. Only CUDA residency/current-device observations are adapted. Two negative controls demonstrate that a dense-NCHW reinterpretation gives different values; they reject a modeling shortcut and do not identify a defect in this selected kernel.

The same 16 metadata fixtures go through the actual JIT binder and offline SM90 compilation with default compiler options. Their TTIR/TTGIR/LLVM/PTX artifacts are pinned. The symbolic PTX checker uses the actual strides, spans and pointer offsets and verifies operand provenance, masks and zeros, half conversions, shared descriptors, row fragments, carry chains and complete stores. All 16 new profiles pass. The updated checker also passes the 12 inherited dense profiles. Each report rejects the three fragment/store-swap negative controls. Floating instructions remain uninterpreted throughout these provenance checks.

`check_source_binding.py` verifies the exact Python AST repair and wrapper preservation, unchanged admission expressions after Boolean qualification, selected readable quotations, explicit zero fills, conversion branches and source structure. These checks are source-binding evidence, not a verified Python-to-DSL translator.

`check_storage_tail_metadata.py` checks 12 actual admission-function cases over 5-, 6-, and 7-byte untyped allocations, including same-object input/weight aliasing. They confirm that the unchanged guard admits unused storage tail bytes. These controls adapt only CUDA residency/current-device observations and perform no kernel or numerical execution.

Focused reproduction commands:

```bash
python bench/repairs/tritonbench_g/triton_conv2d_fwd/check_kernel_layouts.py
python bench/repairs/tritonbench_g/triton_conv2d_fwd/check_kernel_profiles.py
python bench/repairs/tritonbench_g/triton_conv2d_fwd/check_source_binding.py
python bench/repairs/tritonbench_g/triton_conv2d_fwd/check_storage_tail_metadata.py
python .veritile-cache/task-deliveries/triton_conv2d_fwd/build-task.py
```

The frozen packet is `.veritile-cache/task-deliveries/triton_conv2d_fwd/delivery.json`. It records exact reproduction arguments, evidence scope, source/build/import hashes, and the retained initial snapshot. Independent acceptance is recorded separately by the root-assigned reviewer.

## Public IO organization

`IO.lean` exposes FP32 input `[N,C,H,W]`, weight `[O,C/G,KH,KW]`, and output `[G,N*OH*OW,O/G]` channels. Named affine windows retain every input/weight stride; output uses the original NCHW address. All logical lanes are active, output is a distinct region, and global scratch is empty. Overlapping read-only coordinates remain admitted through simultaneous input binding. `ConvIO.invocation` fixes the actual quoted source, backend, PID ABI and unchanged full grid, including redundant feature programs and empty batches.

The original arbitrary `KernelNumeric.Value` interpretation is retained: ordinary reals/words embed as scalar payloads, while existing symbolic dot payloads are not silently excluded. `logicalScalars` applies their existing independent numeric interpretation. `logicalRead` extends only finite logical inputs; it never reads an execution state or output. The mathematical and native targets reuse the independent grouped sum and ordered instruction graph, including all casts, padded operations and precision flags.

`Proofs/KernelHalfSafety.lean` and `KernelDefaultSafety.lean` prove the actual quotation's access safety through the original h/w/channel loops and actual predecessor states. `LogicalInputs` identifies every active source pointer with a declared strided tensor coordinate. `PublicSafety` connects these results to the original complete schedule and exact frame. `LogicalReference` proves formula congruence from logical inputs alone; `PublicResult` composes it with the existing `launch_reference` execution proof.

`Correct.lean` has one Real IO specification. `Misc/NativeCorrect.lean` retains the all-flags native IO theorem and `Misc/AuxiliaryFPEquiv.lean` retains its default specialization. `FPEquiv.grid_equiv` compares the complete original against a fixed-128 spatial-index strength-reduction candidate with `≡[[]]`. The native results are source-to-formula observations; the inherited scalar/conversion/fixed-MMA primitive laws remain explicit, with no new numerical assumption or experiment TODO.

All previous raw statements are preserved as ordinary theorems in `Proofs/KernelRealResult.lean` and `KernelNativeResult.lean`, including every-program byte-store projection. `Misc/TensorBridge.lean` retains its exact previous statements and proofs through the same underlying execution/binding foundation. `Misc/Reader.lean` audits the supporting results without importing public entries; `Misc/Review.lean` audits the main and Tensor entry points. Both main import closures are independent of TensorBridge.

This format pass changes no Python, kernel source, contract, dtype, flags or native model. Targeted Lean builds and public-surface/formula-independence/axiom checks are recorded in the current format-delivery report. The historical provenance and validation evidence above remain historical; this pass adds no GPU run, numerical experiment or independent cold replay.

Public-entry review (2026-10-21): `fp_equivalent` is retained with its original
name as an ordinary theorem in `Misc/NativeCorrect.lean`. The complete primary pair is now connected in FPEquiv.lean; the native theorem remains an independent optional application.

## Full source address-optimization pair

`CandidateKernels.lean` contains readable original-precision quotations for both
FP16 and default operand paths. They replace only the fixed 128-row spatial base
`batch_height_width_pid *128` by `batch_height_width_pid <<7`. All masks, h/w/c
loops, zero fills, casts, TF32 settings and the original complete redundant
group grid remain. `triton_conv2d_fwd_shifted.py` exposes the full checked
candidate wrapper; `check_pair_source.py` verifies the exact whole-module delta.
The optimization is integer address strength reduction; no speedup is claimed.

`Proofs/Pair.lean` proves changed-statement run and safety equality for arbitrary
structural algebra/state, then lifts through the full kernel and every original
launch. `FPEquiv.grid_equiv` includes both successful safe schedules, complete
output cells and both exterior frames. Its experimental rule set is empty and
it imposes no numerical restriction on original word/symbolic payloads.
The independent Real formula, accepted shape/layout domain and optional native
model correspondences retain their earlier scope.
