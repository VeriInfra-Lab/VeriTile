# Kernel-only context attention

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/Audit.lean](Misc/Audit.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

Current program-pair result (2026-10-21):
`FPEquiv.block_start_reused_equiv` proves complete selected source grids
`original ≡[[]] blockStartReused`. `Kernels.lean` directly quotes the full
original and the directly related candidate. The candidate reuses
`block_start_loc = BM * start_m` in `offs_m`, removing one repeated integer
multiplication per CTA; no performance measurement is claimed. The existing
scaleFirst source remains historical and its floating rewrite is not assumed.

`SourcePair.original_preserved` proves complete original equality with the
historical FP32 source. `same_run/same_safe` establish exact source state and
safety equality from integer commutativity and actual prefix registers.
`PairIO` fixes the original `IOData.domain`, logical windows, metadata, grid
jobs, masks and backend. `PairEquivalence` concludes both safe successful
runs, every active typed output and both complete exterior frames. The original
all-word domain, allowed Q/K/V/output aliases and incomplete metadata coverage
remain admitted; only output/start and output/length separation is retained.
Natural exp, negative-infinity causal mask, probability casts, normalized
recurrence, dot/FMA distinctions and all FP32 instructions retain their order.
There is no floating rewrite, experimental rule or finite-input restriction.
The independent Real causal softmax and its original complete-grid input
conditions remain in `Correct.lean`; optional source/native observations keep
their original names/statements in `Misc/NativeCorrect.lean`.

This package proves the unchanged checked source
`../context_attn_nopad_repaired.py` at the kernel boundary. The source SHA256 is
`cf0939e3b862aa0e05b117cb1d66e828ffa7e65e870b62764454b782df79fed1`.
The prior Python source, checked contract and existing theorem scopes are preserved.

`Contract.lean` exports ordinary same-device CUDA FP32 Q/K/V/Out Tensor and
storage laws, identical canonical `[T,H,D]` descriptors, resolved view flags,
contiguous INT32 metadata, and the source's checked capacity, interval,
alignment, storage and native-integer bounds. D is 16, 32, 64 or 128. All five
read roles may alias, including payload/metadata backing aliases. The output's
complete untyped backing is disjoint from all read backings. Input contents
and metadata come from one original physical byte heap. There is no payload
finiteness restriction, allocator condition, host-success condition, compiler
result, or source/result-equality assumption in the public contract.

`Kernels.lean` selects the original readable typed quotation with FP32 precision
and the actual ordered Python binary64 scale `1.0 / (D ** 0.5)`. It is not
replaced by an exact reciprocal square root. `FPReference.program` proves
prelude, loop and epilogue execution; `FPReference.whole` proves the complete
grid by calling `FP.Structural.exec` on that quotation for every PID.
`NativeSource.source_connection` connects the repaired signed64 PID prefix to
that execution. `Safety.projects` derives metadata extension, PID, channel,
loop and all Q/K/V/Out element/byte offsets from the ordinary guards, including
false-mask lanes and the source's K/V pointer association.

`TensorBridge.correct` retains successful execution of the Real-erased shared source and
the supplied output Tensor's logical values at every valid batch/head/time/
channel. `mathematical` is an independent natural-exponential weighted sum over
exactly `0 ≤ j ≤ t`, including the current time step. Its payload reader ranges
over unrestricted Real values at the actual Tensor addresses; it has no FP
encoder or finite-value premise. The Real store observer carries executed
Real output cells over their actual byte-address footprints; it does not
claim to encode Real values into IEEE bytes.

`TensorBridge.fp_equiv` retains successful typed source execution and exact logical output
words against an independent online-softmax recurrence. Query/key/value tiles,
negative-infinity masks, ordered max/exp/sum operations, probability and output
casts, zero/live dot seeds, and accumulator updates are explicit in that
reference. `logical_input` and `logical_metadata` bind the source to the actual
shared Tensor interfaces; `logical_output` reconstructs all 32 output bits
from the four stored bytes. `logical_coverage` covers every valid logical row.
Both Tensor bridge results preserve the entire read backing intervals and every
address outside the exact output write footprint; empty calls are identities.

The selected native model is SM80 default TF32 with fixed `m16n8k8` MMA
instruction chains, live accumulators, source FMA contractions, four ordered
physical reduction replicas, operand routing, and exp2 lowering. It uses four
warps for D ≤ 64 and eight for D = 128, with BM = BN = 128 and one stage.
`NativeInstructions.ScalarLaws` exposes eight scalar families and one fixed-K8
single-coordinate MMA family as pointwise numerical correspondences. A
whole dot, reduction, graph, execution or final-result equivalence is never
a caller premise. Compiler realization of the selected structural model,
ordinary Tensor/storage API behavior, byte transport, and race-free launch
delivery remain explicit external boundaries. Native-to-Real equality is not
asserted. Device resources and the separately tracked launcher grid-product
issue remain outside this kernel-only result.

The task-local FP/backend/native support extraction avoids the historical
`context_attn_fwd.Proofs.FPModel` ABI and duplicate `evalExpr` import path.
The old failing probes and original file hashes are retained in the evidence
packet. Pure Real source proofs are reused; no shared package is overwritten.
The public import closure is pinned together with available build receipts
and object hashes. `Misc/Audit.lean` checks the public results, source connection,
integer projection, frames, and independent targets with only the standard
Lean axioms `propext`, `Classical.choice` and `Quot.sound`.

`check_source.py` reruns the existing full-source/guard checks and adds eight
CPU interpreter cases covering every D, boundaries 0/1/127/128/129/257,
nonzero storage offsets, full input/output-complement backing checks, actual
payload/metadata aliases, and admitted NaN/Inf/signed-zero inputs. The finite
cases compare with independent causal Torch softmax. Exceptional inputs are
checked for admission and frame behavior, without a Real-result comparison.
These use the actual Triton CPU interpreter with explicit CUDA metadata
facades; they are not GPU validation.

`check_compile.py` offline-compiles all four widths with ordinary aligned and
four-byte-offset pointer specializations, including normally specialized unit
strides. It saves TTIR, TTGIR, LLVM IR and PTX and checks the fixed TF32 MMA,
FMA and exp2 instruction families. These controls do not certify the complete
physical reduction/routing model. No GPU execution or native instruction
numerical conformance experiment was performed in this revision.

## Public source IO

`Correct.lean` exposes `attentionIO : AlgorithmProgramIO` and the independent
`attentionScore` / `causalAttention` formula. Its sole specification, `correct`,
uses shared `AlgorithmProgramIO.ImplementsOn`. The five inputs are logical
Q/K/V values and signed start/length metadata. The output window uses the
original output strides and the active packed-sequence mask. The source is the
Real interpretation of the selected FP32 quotation. Execution remains one
**ordinary independent full grid**, with each CTA reading the original memory
and disjoint writes merged by the existing launcher relation.

`Host.Shape`, distinct output/read labels and complete grid coverage retain
the original Real applicability conditions. The metadata domain pins only the
original start/length observations. Logical input binding supplies the Q/K/V
values; finite input/output window bounds imply all actual source accesses are
safe. The target uses the original binary64 host scale, natural exponentials,
and exactly the keys at or before the query. The full exterior frame preserves
all input regions under the stated label conditions. The stronger historical
flat-allocation result remains unchanged in `Proofs/KernelRealResult.lean`.

`Misc/NativeCorrect.lean` exposes `nativeAttentionIO : ProgramIO` and shared
`ProgramIO.ObservesOn`. It executes every original `FPHost.jobs` invocation
with the selected native symbolic backend. `nativeAttentionExpected` observes
a pure, source-independent online recurrence and scalar last-writer scatter.
It preserves **all original symbolic inputs and all words**, arbitrary source
strides, Q/K/V/output aliases and even an incomplete `maxInputLen` grid. Only
the original output-vs-metadata separation is required; Real's output
injectivity, shape and complete-grid conditions are not added to this result.
Actual load/store safety follows from the same live logical IO bounds, with
metadata stability proved through every executed source step.

The pure output observer evaluates the symbolic result under the existing
SM80 plan. The nine scalar/fixed-K8 instruction laws remain the primitive
numerical boundary; their deferred validation status is unchanged. There is
no optimization rewrite, new numerical assumption, leaf-only input restriction,
or native-to-Real claim. The historical complete-memory equality and global
word-observation theorem remains unchanged in `Proofs/KernelNativeResult.lean`.

`TensorBridge` retains all former complete Tensor statements, physical
readback, input-backing frames and empty identities. Only its imports change
to the retained helper modules. It reuses `Host.host_complete` and
`FPReference.whole`, the same identified execution foundations used by the
public IO proof. Neither main imports the bridge, and helpers import no public
reader. The IO additions have targeted Lean build receipts; no new CPU/GPU
execution or numerical conformance experiment was run for this organization.
