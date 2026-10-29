# token_softmax_llama: complete kernel proof

Preserved raw endpoint readers are also in Misc: `Misc/KernelRealResults.lean`, `Misc/KernelNativeResults.lean`. Their full declarations are unchanged.

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`.

The canonical source is the unchanged `ledger.contract_revision.source`,
`token_softmax_llama_repaired.py`, SHA256
`3a7a843d8532cc306d9ea665674859dbf91e88933c413074287757d43a86eab4`.
The complete file, including its guards, launch, and wrapper, is byte-identical
to the accepted canonical token_softmax_bloom source. Llama's quoted source
was already the common source used by Bloom's proof. `Proofs/KernelSource.lean`
makes source, guard-domain, and complete candidate correspondence explicit,
so the new public Llama theorems reuse those checked results directly.

The API `token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out,
max_input_len)` mutates the supplied output and returns `None`. Logits and
output are FP32 CUDA strided tensors with equal shape `[heads,tokens]`;
starts and lengths are signed-int32 CUDA strided tensors with equal shape
`[batch]`. All tensors share one device and have no negative-view flag.
The maximum must have Python type `int` and be nonnegative.

The unchanged guards admit zero and nonunit input/metadata strides, offsets,
holes, and read allocation aliasing, including shared FP32/int32 storage.
Output strides must separate rows or columns. Transposes, slices, offsets,
and irrelevant singleton zero strides remain admitted; exotic injective
layouts that satisfy neither separation inequality remain rejected.
The output allocation is physically disjoint from all three read allocations.
Starts and lengths are nonnegative, lengths do not exceed the maximum,
`start+length <= tokens`, and nonempty segments are pairwise disjoint.
Segments may be unsorted, empty, or separated by untouched gaps.

Empty batch/head/token axes skip content and pointer reads and kernel launch.
A zero maximum with nonempty axes still validates metadata, forcing all
lengths to zero, then skips launch. Every admitted no-launch case preserves
all caller cells. Active launches require `batch < 2**31`, `heads < 2**16`,
and `BLOCK_SIZE <= 2**20`. All PIDs, lanes, loaded metadata, and six strides
use int64. Every formed index, including masked lanes, is below `2**63`;
all formed pointers fit uint64 and are four-byte aligned. Actual Tensor laws
and the segment bounds establish active storage bounds. There is no new
contiguity, total-numel, nonzero-stride, or finite-FP32 restriction.

The formula uses a single padded global maximum, FP32 subtraction,
exponentiation, the complete segment sum, and division in source order.
The independent Real target is `exp(x[n]) / sum_j exp(x[j])` on original
logical Tensor values. The proof accepts arbitrary Real inputs.
The independent native reference retains the ordered reduction graph,
thread ownership, original 4/8/16-warp choice at widths 2048/4096, and
the six scalar raw32 instruction meanings maxNum, subRN, mulRN,
exp2Approx, addRN, and divFull. It covers every raw FP32 word, including
exceptional values, with the selected compiler layout `V` explicit.

`token_softmax_llama_index_commuted.py` retains the complete canonical wrapper
and changes exactly one integer metadata-address product:
`cur_batch * stride_length` becomes `stride_length * cur_batch`.
The complete candidate kernel AST equals Bloom's accepted candidate AST.
`KernelSource.equivalent`, `execution`, and `grid_execution` transfer the
empty-assumption program proof, complete Structural execution equality,
and full-grid equality. Floating instructions and their order are unchanged.

The public `Correct.lean` headline uses the shared `ProgramIO.ImplementsOn`
judgment for the complete selected schedule and independent finite softmax
formula. `IO.lean` reuses Bloom's explicitly typed metadata/payload/output
channels, actual strides and masks; `Proofs/IOCorrect.lean` transfers the
already-proved actual source access safety, successful complete execution,
all active values and exact exterior Cell frame. Repeated identical launches
remain admitted: uniqueness identifies PID/lane events, not list positions.

`FPEquiv.lean` has one standard whole-source equivalence headline, commuting
the integer length-address product with an empty numerical assumption list.
The complete native `ProgramIO` observation remains the ordinary supporting
`PublicIO.native` theorem. The prior combined full-grid source/candidate
statement, including one common raw state, every native output and the frame,
is retained verbatim in `Misc/KernelNativeResults.lean`; the previous raw
Real statement is retained in `Misc/KernelRealResults.lean`.
The unchanged TensorBridge applications retain observed native states,
caller-cell frames and preservation of all cells of all read allocations.
The shared `KernelTensor`, `KernelAddresses`, and `KernelBinding` proofs bind
the actual Torch payload Tensor, physical indices, signed-int32 metadata
loads, int64 sign extension, masks, and uint64 pointer arithmetic. Addressed
raw words allow differently typed read aliases without conflicting cell tags.
Success or a whole-row result is never an input assumption.

The former public host roots are preserved byte-for-byte as
`Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`, retaining their original
namespaces and theorem names. The row proofs and earlier accepted Whole003
artifacts remain unchanged. These historical host modules are optional and
are outside the public kernel import closure.

Independent source review confirms that the original file ignored metadata
strides. For starts storage `[0,1,4]` and lengths storage `[2,1,2]`, both viewed
with stride two, the intended segments are `(0,2)` and `(4,2)`. The original
loads `(0,2)` and `(1,1)`, leaving required outputs 4 and 5 untouched.
With six zero logits and output initialized to -7, those cells remain -7
instead of the mathematical target 1/2. The canonical repair fixes this
defect. No remaining defect was found in the canonical source/domain.

`check_kernel_correspondence.py` checks exact canonical bytes, the one-product
AST rewrite, the analytic original counterexample, and frozen shared evidence
hashes without running new numerical experiments. Identical source reuse
retains Bloom's 14 admission cases, 32 rejection controls, 18 focused source
controls, and 46 complete-storage source/candidate comparisons. The evidence
directory is `.veritile-cache/kernel-evidence/token_softmax_llama/`.
Current build receipts, combined public imports, standard-base axiom audits,
and independent-reference checks validate the Llama adapters.

The model boundary consists of ordinary Torch Tensor/storage laws, audited
Python-to-DSL correspondence, selected native layout and common primitive
instruction semantics, and GPU runtime/hardware conformance. There is no
verified Python compiler or GPU execution claim. Hardware conformance and
FP32-to-Real approximation are separate from the proved kernel theorem;
no unresolved per-task numerical experiment remains in this kernel packet.

`Misc/TensorBridge.lean` exposes Python-check correspondence and the complete prior
Tensor applications as ordinary theorems. Neither primary file imports this
bridge. The current pass changes no Python source, Kernels or Contract,
introduces no numerical law or GPU experiment, and verifies the changed
entries and existing bridge with targeted current-source Lean builds.

Source authority maintenance (2026-10-22): `Kernels.lean` now owns the existing
complete original/candidate quotations, `Strided.Parameters`, `Strided.kernel`
and `KernelSource.candidateKernel`. These are exact moves from
`Proofs/Model.lean` and `Proofs/KernelSource.lean`; all declaration names and
types are retained through their existing imports. Model, source-sharing,
equivalence, execution and grid proofs stay in `Proofs/`. Complete function
equalities by `rfl` and fresh joint main/retained optional imports are checked
in the source-only cache. Both programs, source precision, domain, shared Bloom
correspondence and proof completion status remain unchanged.
