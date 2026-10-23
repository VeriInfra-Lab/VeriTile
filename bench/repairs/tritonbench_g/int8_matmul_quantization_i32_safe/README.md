# INT8 matmul quantization: signed32-safe candidate

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/Review.lean`.

Current source pair (2026-10-21): `int8_matmul_quantization_indices.py` removes the producer's redundant row `% M` (the actual wrapper fixes `BLOCK_SIZE_M=1` and launches exactly M rows), and removes both GEMM scale-load masks (the loaded offsets are already `% M` and `% N`). These reduce discrete index/predicate work; no compiler/GPU speedup is claimed. `FPEquiv.lean` exposes the actual complete producer pair `producer_indices_equiv` and the actual complete GEMM pair `gemm_scale_masks_equiv`, each `original ≡[[]] optimized`. Complete ordinary original/candidate quotations are in `Source.lean`. Full-state execution equality and safety iff preserve every floating instruction, cast, original precision, all 34 GEMM and four producer configurations, both B layouts, empty grids and all-word floating inputs. GEMM retains the same typed-zero initializer, independent per-cell split order, both complete successful safe source traces, outputs and exterior frames through the task-local `ResetPairIO` adapter. Producer retains every complete row schedule and both frames. The independent Real specifications in `Correct.lean` are unchanged; optional native/Tensor results remain under `Misc/`.

This explicitly separate candidate starts from the frozen `scalar_cast` repair. Its only Python change is the `K > 131071` rejection inside the existing `if M and N:` GEMM check. Composed calls perform that check before the producer or allocation. The 34 GEMM configurations, four producer configurations, both B layouts, all dtype specializations, zero-row/zero-column returns, standalone K=0, standalone producer domain and Torch helper axes remain unchanged. No payload, finiteness or scale guard is added.

The candidate has two independent numerical conclusions:

* The ideal Real interpretation of the complete source reaches the dense formula `sum_k(A[r,k] * B[k,n]) * row_scale[r] * column_scale[n]`.
* The native interpretation reaches the exact selected instruction graph: signed INT32 accumulation, signed32-to-FP32 conversion, FP32 multiplication by the row scale and then column scale, mandatory FP16 partial narrowing, destination widening when required, and destination-typed atomic updates in each cell's actual projected interleaving.

There is no universal native=Real theorem. FP32 overflow and FP16 split-partial overflow are permitted behaviors of the preserved source. The earlier floating witnesses delimit that unclaimed equality; they are neither new defects nor deferred numerical obligations. The canonical and scalar_cast original-domain saturation defects remain frozen and unaccepted.

Every INT8 product has magnitude at most 16384. Every distinct masked source subset is therefore signed32-safe for K≤131071. Padding is proved zero and is not counted as an active product. `NativeInteger.lean` lifts this bound through actual scalar/K32 microinstructions, fused carry, nonfused block-plus-add, and every source K-loop prefix. At K=131072 the all-(-128) product sum reaches 2^31; retained SPLIT_K=1 makes 131071 the maximal common K-only no-overflow bound.

## Proof map

`Kernels.lean`, `Contract.lean`, `Correct.lean` and `FPEquiv.lean` are the standard public entries. `Proofs/All.lean` preserves the historical candidate audit; `Misc/Review.lean` imports the current IO readers, complete composition transport and Tensor endpoints. All auxiliary proof modules in the table below are under `Proofs/`.

| Layer | Files and principal results |
|---|---|
| Exact host delta and admission | `Contract.lean`: `gemmValid_iff`, empty/K0/sharp-boundary results. `check_candidate.py`: complete AST binding, exact restoration of scalar_cast AST after removing the guard, metadata-only actual-host execution. |
| Full readable source | `Source.lean` contains the producer and all four GEMM specializations, plus the complete production AST. Five quotation identities reuse the frozen source proofs. |
| Exact discrete optimization pairs | `ProducerIndices.lean` / `ProducerIndicesIO.lean` prove full execution, safety and producer-grid transport; `GemmIndices.lean` / `GemmIndicesIO.lean` prove the same for all four GEMM precision choices. `ResetPairIO.lean` retains the original typed reset and both actual per-address source traces. `check_indices.py` verifies the complete Python AST and all five candidate quotations after exactly the three intended edits, including unchanged host/helpers/configs/guards. |
| Integer and operand connection | `IntegerBounds.lean`, `NativeInteger.lean`, `GemmValues.lean`: actual source masks/indices/strides, ordered input-only integer graph, and concrete signed32 exactness from the new guard. |
| Full program/grid | `Commit.lean`: full source execution, current-cell store/atomic, exterior frame. `Grid.lean` and `GridResults.lean`: unique physical owners, all logical events, each event at its current output, source success without a success premise, per-cell order, reset, native result and separate dense Real result. |
| Host lifecycle | `Lifecycle.lean`: literal initial/trial/post-benchmark reset and final-grid lifecycle; all 34/4 configs and empty events. `ProducerLifecycle.lean`: complete producer benchmark executions preserve input and exterior memory. |
| Original Tensor and backing | `Tensor.lean`, `Endpoints.lean`: shared original `TensorView`, address/layout/allocation laws, actual logical reads and returned handle, byte encodings, native/Real standalone endpoints. `Admission.lean` connects the caller snapshot to the actual prepared backing using input liveness and output separation, proves pointers and supplied identity, and derives composed liveness. The `Public.TensorBridge.correct` / `Public.TensorBridge.fpEquiv` endpoints require this ordinary metadata admission and conclude both actual prepared-backing source simulation and the independent post/frame. The lower `Tensor.*_complete` declarations are snapshot-semantic lemmas, not the admitted physical public API. |
| Admitted public API | `PublicAPI.lean`, `PublicComposed.lean`, `PublicProducer.lean`: source simulation directly consumes the actual post-allocation backing; standalone input liveness and Tensor laws are mandatory. Composed liveness is fixed to original backing plus both actual producer allocations, and its public frame quantifies every original live storage. Standalone producer also includes benchmark-source executions, input liveness, and input/output pointer laws. |
| Producer and composition | `ProducerTargets.lean`: full quoted producer grid to independent native/Real byte and stored-scale formulas. `ProducerTensor.lean`: original floating Tensor to actual stored INT8/FP16 backing and exact produced Tensor request. `Composition.lean`, `InputCongruence.lean`, `ComposedTensor.lean`: actual producer backing is consumed by GEMM, then masked input congruence replaces it with independent formulas over original caller tensors. Both composed endpoints retain returned-object/device/frame conclusions. |
| Unchanged Torch helper | `Endpoints.lean` reuses the shared helper contract, native local-instruction/Torch ABI, Real result, optional transpose-copy and actual returned-view theorems. Its Python function slice is byte-for-byte AST-equivalent to the frozen baseline. The inherited helper ABI remains explicit; it is not a new claim that the full candidate module has the baseline's full-AST identity. |

## Semantic and provenance boundary

No theorem assumes successful GEMM execution or a desired kernel/grid/final-memory value. DotPlan coverage is a finite structural permutation; compiler selection sees dimensions/source coordinates, not input payloads. Structural scalar profiles fix multiplication, casts, signed integer conversion and packing. Maximum, division and typed atomics retain explicit single-instruction policies; reductions retain their explicit structural trees. They are not whole-row or whole-kernel callbacks. Tensor allocation/layout/pointer facts and successful runtime dispatch are the ordinary shared external ABI boundary.

Existing quotations retain their established specializations: checked nonnegative bounded source indices are represented as Nat; INT32-to-FP32 conversion is numeric promotion; half row-scale widening and destination casts are explicit. The complete candidate AST and unchanged JIT/configuration slices are independently bound by the candidate checker. The frozen scalar_cast compiler reports and emitted artifacts supply the retained frontend and instruction evidence. No new compiler probe, GPU launch or numerical experiment was run for this candidate.

The imported 15 source-execution modules are preserved exactly. Additional shared metadata/Torch/helper/arithmetic modules are reused by import; the frozen manifest records transitive workspace sources, actual object identities, candidate direct-import receipts, retained source/probe evidence and explicitly absent inherited receipts. This candidate uses a private Lean output directory and does not modify shared core, canonical files, Lake, ledger or canonical selection.

Validation consists of the existing 62 metadata-only host cases and seven rejected mutations, a final ordered build of all 26 candidate Lean modules, combined clean-axiom/non-circular audits, and exact frozen/live/preservation hash verification. Final status is a frozen candidate for independent whole-task review, not self-acceptance.

## Current public IO format

`Correct.lean` now contains two finite tensor IO specifications: the complete
producer and the complete GEMM. `Misc/NativeCorrect.lean` retains the corresponding ordinary native
observation theorems. `ProducerIO.lean` and `GemmIO.lean` expose named input/output
channels, source identity, windows, masks, storage precision, launch geometry,
and independent formulas. Operational states and per-address execution details
are in `Proofs/`.

The producer reads one finite FP32 matrix and returns signed INT8 data and FP16
row scales. The GEMM reads finite signed INT8 A/B matrices plus the original
FP16-or-FP32 row scales and FP32 column scales; it writes the original
FP16-or-FP32 C matrix. `SignedInputs` is the exact signed-INT8 representation
condition, not a floating numerical assumption. All floating words remain
admitted. The row/column-major B choice, arbitrary reduction plans, empty grids,
all original precision branches and per-cell atomic order remain unchanged.

The GEMM uses the shared `IntegerDotPerAddressProgramIO.ImplementsResetOnCells`
relation. Its initializer is fixed finite typed zero data. Each cell trace
observes the selected cell of a complete successful source invocation, with
proved source safety, locality, complete event coverage and the exact exterior
frame. It does not replace the original arbitrary single-cell interleaving by
serial whole-CTA execution. `PublicIO.Gemm.source_at_arbitrary_prefix` explicitly
retains the stronger original theorem for every arbitrary commit prefix.

The shared `IntegerDot` accumulatorNat profile is exactly related to the
original `KernelExecution` by `SharedInteger.exec_eq` in `Proofs/IntegerDotBridge.lean`. Required Nat context
registers, unreduced source coordinates, exact accumulator assignment matching,
actual incoming carry and original control-flow fallback are retained.
`IntegerDotModeChecks.lean` checks these semantic distinctions using artificial
scalar primitives; those checks are not new production numerical laws.

`Proofs/RawCorrect.lean` and `Proofs/RawNative.lean` preserve the former entry
files verbatim, including their complete producer/GEMM composition. The
ordinary `Proofs/CompositionIO.lean` transport identifies the same produced
state, fixed `Composed.attach` aliases (`A ← q`, `AS ← scales`), shared producer
execution, and shared full GEMM source execution at every arbitrary commit
prefix. It preserves the complete original intermediate outputs, final outputs
and frames; no conjunction of two standalone IO claims is presented as a
compound source execution.

`Misc/TensorBridge.lean` retains all six original `Public.TensorBridge` endpoints,
including pointer/backing, allocation, original caller, lifecycle, output and
physical frame conditions. Only its first two imports move to the raw helpers.
It shares the same existing source execution foundation as the new IO results.
Main entry import closures exclude TensorBridge, and supporting proof modules
do not import the public entry files.

The earlier public-IO format change added no Python/kernel/contract change, new numerical law,
experimental TODO, GPU run, or independent cold replay. The preceding frozen
validation records are historical evidence; current scoped build receipts and
source-preservation checks are recorded by the public-IO-format delivery batch.

The current discrete optimization pair has passed its complete Python/Lean
source-delta check, the retained 62 host metadata cases and seven mutation
rejections, targeted source/proof/public-entry builds, and a fresh joint import
of Contract, Correct, FPEquiv and the existing optional NativeCorrect/TensorBridge
entries. Both public pair theorems and both Real specifications have standard-base
axiom footprints. It adds no numerical rewrite, experimental rule or deferred
numerical obligation. Current receipts are in the 105-campaign task delivery.
