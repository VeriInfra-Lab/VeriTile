# `chunk_gla_simple_min_dot_tile`

This is a separate whole-kernel candidate. The predecessor and its frozen source-defect packet remain unchanged. The Python source differs by exactly one expression:

```python
BK = max(16, min(triton.next_power_of_2(max(K, 1)), 64))
```

The inherited source docstring describes the predecessor's BK preservation; the expression above is this candidate's deliberate change. It fixes the FP32 dot inner-dimension requirement for K=0 through K=8. It adds no K/V guard, input copy, compatibility branch, changed dot precision, or arithmetic reordering. The checked FP32 input domain, all original BT values, aliases and offsets, canonical scalar strides, and empty-output route are retained. Machine equivalence to the predecessor is not claimed on its uncompiled domain.

## Source and proof chain

`check_source.py` checks byte-for-byte preservation except for that expression. It also parses both complete selected-FP32 quotations in `Kernels.lean` and compares them with their actual complete JIT ASTs. The address-CSE candidate is in `chunk_gla_simple_row_reused.py`; the original Python source remains byte-identical. Only explicit native integer widening is erased and the guarded FP32 dtype choices are resolved in that comparison. Masks, pointer formulas, loop bounds, all loads/stores, and all numerical operations remain in the comparison. `DirectReal.source_binding` binds the candidate specialization to the frozen generic execution proof at its new BK. The predecessor's shape-specific small-BK instance is never used for the candidate.

`Proofs/Shape.lean`, `Geometry.lean`, and `Layout.lean` specialize the original geometry to BK in [16,64]. The original generic mathematical and source-execution lemmas are reused. The candidate namespace reconstructs typed source execution, the two accumulator loops, all gates and masks, the final dot/scale/store, and the entire grid. It imports the frozen predecessor's `PhysicalDirect` library early to reuse the canonical automatically generated Lean equations; this prevents duplicate equation declarations and does not assume a predecessor host result.

The independent Real target is the zero-extended finite contraction of the logical Q/K/V/history/gate arrays: gated history plus causal gated QK/V contributions, then scale. Its sums are expressed in zero-padded key blocks. `IO.lean` declares all five complete input windows, the complete output and actual three-dimensional schedule. `Targets.lean` contains the state-free formulas. `Correct.whole_kernel` uses the shared `RegionalScheduledImplements` judgment: actual source access safety, successful complete schedule, every output and the exact exterior raw-cell frame. `TensorBridge.real_returned_tensor` retains the returned-pointer result. `Static.k_zero_real` proves the K=0 Real value; `Static.witness_two` proves the minimal all-ones, gate-zero, scale-one value is 2.

The native target expands all three FP32 IEEE dot sites into ordered scalar FMA graphs, retains their accumulator carries, the source casts, the exp-to-exp2/log2e lowering, and the runtime scale word. The ordinary auxiliary theorem `FPEquiv.whole_kernel` in `Misc/NativeCorrect.lean` uses shared `ProgramIO.Observes`, with arbitrary symbolic inputs and the original `ScalarGraph.lower scale` output observation. `Proofs/IONative.lean` connects actual source safety, the complete grid execution and exact FP32 output cells before applying that observation. `Tensor.complete` additionally composes signed native i64 address/mask/range and ABI proofs with actual input pointers, the returned tensor's identity and metadata, little-endian FP32 bytes, and full caller/input frames. It has no premise asserting host/launch/source success or whole-output equality. `Tensor.empty` gives an unchanged byte memory for every empty-output shape.

The prior operational statements remain ordinary theorems in `Proofs/KernelRealResults.lean` and `Proofs/KernelNativeResults.lean`. The Real scheduled result and the original independent-frame grid result share `DirectReal.perProgram`; both retain the full output formula and exterior raw-cell frame. `Proofs/IOSourceSafety.lean` checks the selected explicit-pointer source, rather than applying the older block-pointer safety theorem. `Proofs/IONativeSafety.lean` follows the actual typed source states and K-loop. Python, `Kernels.lean`, `Contract.lean`, and all returned-Tensor statements are unchanged by this public IO pass.

## Domain and model boundaries

The original checked domain requires ordinary contiguous FP32 tensors on one CUDA device, matching Q/K `[B,H,T,K]`, value `[B,H,T,V]`, gate `[B,H,T]`, and packed history `[B,H,ceil(T/BT)*K,V]`. BT is an exact Python integer in 16,32,64,128,256,512,1024. The finite builtin scale is converted to its runtime FP32 word. Nonempty launches retain the original byte-extent, pointer alignment/span, and CUDA axis checks. Output freshness is the ordinary allocation binding; input buffers may alias one another.

Numerical realization uses only the five fixed scalar families in `ScalarGraph.WordPrimitives` (literal, binary, unary, cast, FMA), with identity raw word leaves. There is no dot/reduction/output numerical port. Correct realization of these primitives and of the inspected compiler lowering is an explicit model boundary, not a numerical TODO. The Real theorem and exact native graph theorem are distinct; neither asserts arbitrary-bit native results equal exact Real arithmetic. In particular, the K=0 FP path is not universally replaced by zero in the presence of exceptional gate words.

Allocation availability, device resource capacity, scheduling/runtime/driver correctness, and compiler realization delimit applicability to physical GPU execution. No GPU execution was available or claimed. The installed SM80 compiler produces 263,168 bytes of required shared memory for B=H=T=K=V=1, BT=256, exceeding SM80's 166,912-byte per-block opt-in limit. This is recorded as a resource condition, without changing the input contract or algorithm. The original maximum score tensor at BT=1024 contains exactly 2^20 elements; the frontend rejects only larger tensors. `Static.dot_dimensions` and `Static.tensor_sizes` prove all dot lower bounds and the inclusive static tensor-size bound.

## Evidence

Candidate delivery artifacts are under `.veritile-cache/task-deliveries/chunk_gla_simple_min_dot_tile/`. The native admission probe executes the actual wrapper assertion AST, generated native binder, signature-order `_pack_args`, unchanged JIT signature/body and original four-warp config. Allocation, CUDA ownership, and submission are mocked; compilation is offline for SM80. The proof build uses the existing workspace-only Lean driver and candidate-owned outputs. Source and binary receipts, compiler artifacts, dependency pins, contract controls, preservation checks, and the model boundary inventory are frozen by the delivery packet for independent review.

The native matrix completed 12 SM80 cubin compilations, including all four original failures and their controls, K0, V1, tails, and BT through 256. The BT512 resource-only continuation was cancelled while ptxas was running; its TTIR/TTGIR/LLVM IR/PTX are retained as partial evidence. BT1024 and the remaining queued profiles were not started. Neither cancelled nor unattempted profiles are counted as compiler successes. The cancellation touched only this probe and its assembler child.

`Misc/TensorBridge.lean` holds the Real/native returned-Tensor applications and
empty-Tensor byte result. `Correct.whole_kernel` remains the primary mathematical specification.
`FPEquiv.whole_kernel` is an ordinary auxiliary native theorem in
`Misc/NativeCorrect.lean`; neither imports the bridge.

`FPEquiv.whole_kernel_equiv` compares the complete actual selected min-BK source
and an address-CSE candidate through `original c ≡[[]] rowReused c`. The candidate
replaces only `rows = i_t * BT + tl.arange(0, BT).to(tl.int64)` with
`rows = i_t * BT + o_i.to(tl.int64)`: the same `o_i` vector already supplies the
causal mask. This removes a duplicate lane-vector construction from row-address
calculation. No performance improvement has been measured.

The existing `scaleFirst` candidate remains available with its exact previous
source. It swaps the final FP multiplication's operands. Arbitrary symbolic
native inputs and arbitrary primitive interpretations do not identify those
ordered expression graphs, so that candidate is not the main comparison. The
address-CSE pair preserves the original input domain without a finite-value
guard or a new numerical law.

`Proofs/PairSource.lean` proves complete source execution equality and trace-safety
equivalence for arbitrary payload algebras, using the actually generated row-lane
vector. `PairIO.lean` changes only the actual kernel in each original invocation;
it retains the complete three-dimensional schedule, all five packed input
channels, FP32 output cells, encoding, masks and memory footprint.
`Proofs/Pair.lean` combines those equalities with the established safe successful
execution to prove every output and both full exterior frames. Its
`observed_equivalent` theorem retains `ScalarGraph.lower scale` for every runtime
FP32 scale word, including its scalar-FMA and exp2 lowering, on the same arbitrary
symbolic input domain. The main pair is stronger: its output expressions already
agree before that observation.

The selected original and CSE candidate are complete ordinary `triton` quotations.
Historical mixed-storage original/scaleFirst macros live in the task-local
`SourceCompatibility.lean`; complete function `rfl` preservation covers all
previous storage branches and both previous source families. The independent
`Correct.whole_kernel`, input contract, original Python, optional native/Tensor
statements, numerical scope, casts, dot options, all source FP operation order and
source-success/safety/output/frame scope remain unchanged. There are no new
experimental numerical obligations.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
