Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

Current pair: `FPEquiv.reuse_indices_equiv` compares the selected full scalar-cast
repair with `int8_dequant_matmul_reuse_indices.py`. The candidate reuses the
first `rm/rn` index tiles and removes the identical second assignments after
A/B pointer construction. It removes redundant discrete address work; no
floating operation, integer-dot profile, mask, bias branch, output conversion
or split atomic order changes. No performance improvement has been measured.

Both full programs are ordinary `triton` quotations in `Kernels.lean`. The
runtime declaration `tl.param divfactor : tl.float32 (carrier=tl.float32)`
reads the existing FP32 ABI register through the explicit FP32-to-Real
algorithm view. It inserts no statement or rounding. The complete direct
original quotation equals the retained predecessor AST by `rfl`; the earlier
scalar-cast source defect and its diagnostic remain separate history.

The task-local pair adapter fixes the original memory interface, complete CTA
catalog, source events and each address's independent order. Its conclusions
include both complete programs' access safety, success, locality, coverage,
source-cell traces, typed observed outputs and exterior frames. The candidate
connection proves full invocation state equality and access-safety equivalence,
then reuses the existing complete source theorem. The input domain, all
native scale/bias payloads, original split-C initialization and all admitted
configurations remain. No numerical rewrite or experimental rule is used.
`check_reuse_indices.py` checks the complete Python and Lean source deltas.


# INT8 dequantized matmul: scalar-cast candidate

This separate candidate replaces exactly nine runtime scalar/stride `.to(tl.int64)` calls with `tl.cast(..., tl.int64)`. The Python program-ID casts, precision, wrapper, guards, all 418 configurations, and native constexpr-one specialization are unchanged. The predecessor and its frozen source-defect packet remain intact.

`Correct.lean` exposes one shared `IntegerDotPerAddressProgramIO.ImplementsOnCells` specification for the independent dense Real formula. `Misc/NativeCorrect.lean` exposes one optional ordinary theorem for the exact native fixed-instruction formula. Both expose finite input/output channels and the complete grouped-grid split schedule, with actual source safety, source success and the exterior frame as conclusions. `Misc/TensorBridge.lean` retains the original TensorView, raw storage, both FP16 output bytes and exterior byte observations. Native output is not identified with the Real target.

The input contract is imported unchanged. It retains unit and empty dimensions, nonempty K=0 epilogues, all split counts, optional bias, storage offsets, inert singleton strides, permitted read-only aliases, arbitrary INT32 overflow, and nonfinite or zero scales. The prepared split-K input state contains actual typed FP16 zero cells, as supplied by the existing zero hook. Allocation, autotuning, zero-hook execution, and the host return are outside these kernel-only entries.

The proof executes the readable shared DSL quotation. The shared `IntegerDotInvocation` uses the bounded `IntegerDot` interpreter for the exact integer `acc += tl.dot(a, b)` syntax and the ordinary Structural evaluator for its children and other source leaves. `Proofs/IntegerDotPlanBridge.lean` and `Proofs/IntegerDotBridge.lean` prove exact correspondence with the retained `QuoteSemantics` evaluator for every State, selected plan and integer carry. Fixed K32 operations receive the actual preceding accumulator, including native wrapping or saturation. There is no arbitrary runner, variable-length result callback or assumed source safety field.

`QuoteStages`, `QuoteLoop`, and `QuoteComplete` establish the prelude, actual loop body, loop recurrence, and every epilogue branch. `QuoteGeometry`, `QuoteOperands`, and `QuoteFactors` prove grouped program coordinates, wrapped padded operand addresses, EVEN_K and K-tail loads, and both scale loads. `QuoteRecurrence` and `QuoteValue` connect the resulting live accumulator and FP16 contribution to the independent reference, without associating or commuting native arithmetic.

`QuoteDescriptors.fullResult` proves successful execution of the entire quotation and records its final output, pointer, and mask registers together with its actual store/atomic memory fold. `Result.replay_eq` proves that replaying those descriptors reproduces that same per-program memory. `QuoteSchedule.sourceGrid` evaluates the quotation to obtain those descriptors and schedules the active physical store lanes. Completeness and uniqueness use the original grouped-grid coverage proofs; arbitrary per-cell FP16 atomic orders are retained. `sourceGrid_eq` is a proved refinement to the independent grid recurrence. There is no assumed whole quotation-to-program compiler equality.

The numerical order remains signed INT8 products into the selected INT32 fixed-K32/scalar plan; INT32-to-FP32 conversion; multiply by runtime FP32 divisor word 948045328; row scale; column scale; FP16 narrowing; optional FP32 bias narrowed to FP16 and added only on split zero; then FP16 store or atomic addition. Fixed K32 instruction realization and ordinary local scalar/memory primitive ABI laws are the remaining implementation boundaries. Compiler instruction selection and fusion remain explicit structural data.

The source audit compares all 37 top-level kernel statements after documented normalization. Ten offline SM90 compilations use Triton's actual runtime signature binder, native specialization, `_pack_args`, and `ASTSource`. They include every prior scalar-one failure, unit dimensions, K=0, K tails, EVEN_K, split 2 and 16, bias, and BK128. Successful offline compilation is not a GPU execution claim. No new numerical experiment was used.

Evidence and the frozen delivery packet are under `kernel-evidence/`. The packet pins current sources, reviewed-driver receipts, resolved dependency artifacts, the source audit, binder artifacts, and all predecessor hashes. Canonical selection and ledger updates belong to the parent review.

`IOLayout.lean` declares the six actual source regions A, B, state_x,
state_w, bias and C, their finite row-major windows and exact channel tags.
Bias is active only when enabled; C is an in-place input only for split K,
where the original FP16 zero initialization is the logical input domain.
`Mathematical.lean` contains the independent finite dense and native formulas.
`IOPrograms.lean` includes every grouped tile and split, and derives each
cell's separate event order from the original complete schedule. Zero-sized
grids, padded lanes and every K-tail branch remain covered.

`IO.lean` binds these declarations to the original quotation, primitive
profile and actual runtime divisor. The `Proofs/IO*` adapters establish
finite typed input binding, the actual source recurrence and output register,
complete-CTA cell effects, source-output locality, source-event coverage,
source-derived cell traces and the whole exterior Cell frame. The
`Proofs/IOSafety*` modules prove access safety along the actual evaluated
prelude, loop successors and epilogue. Changing other C cells is allowed in
the per-address locality proof; no whole-CTA ordering replaces the original
independent atomic orders.

`Proofs/RealKernelResult.lean` and `Proofs/NativeKernelResult.lean` retain the
exact former raw-storage endpoints. `Proofs/IOPhysical.lean` proves that each
original physical event's quotation execution is exactly the shared IO
invocation on `Quote.sourceMemory`; its binding preserves read-only storage
aliases, offsets and the original typed-default reads. `Misc/TensorBridge.lean`
retains its complete theorem statements and proof bodies, including pointer
formation, all native output words, both FP16 bytes and exterior byte frames.
Supporting proofs do not import the public reader entries.

This format change adds no numerical assumption, experiment, Python change
or kernel-source change. The fixed primitive instruction profile remains the
existing native model boundary; the finite arithmetic and source transport
facts are Lean proofs rather than deferred numerical experiments. Historical
frozen evidence, independent replay and GPU execution status are unchanged.
