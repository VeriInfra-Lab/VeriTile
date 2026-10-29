# token_softmax_bloom: complete kernel proof

Canonical source: `token_softmax_bloom_repaired.py`, SHA256
`3a7a843d8532cc306d9ea665674859dbf91e88933c413074287757d43a86eab4`.
The API `token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out,
max_input_len)` mutates the supplied output and returns `None`.

Logits and output are FP32 CUDA strided tensors with equal shape
`[heads,tokens]`; starts and lengths are signed-int32 CUDA strided tensors
with equal shape `[batch]`. All four use one device and have no negative-view
bit. The maximum is a nonnegative Python integer. The wrapper passes the
original four tensors and all six strides. Input and metadata views admit
zero and nonunit strides, offsets, holes, and shared read allocations,
including differently typed views of the same storage.

Output strides separate rows or columns according to the unchanged canonical
checks. Transposes, slices, offsets, and irrelevant singleton zero strides
remain admitted. The output allocation is physically disjoint from all read
allocations. Metadata contents must be nonnegative, satisfy
`length <= maximum` and `start+length <= tokens`, and give pairwise disjoint
nonempty segments. Unsorted segments, empty segments, and untouched gaps are
admitted. Empty batch, head, or token axes skip metadata reads and launch.
A zero maximum with nonempty axes still validates metadata, forcing zero
lengths, and then skips launch. Every accepted no-launch call preserves all
caller cells.

Nonempty launches retain the grid bounds `batch < 2**31`, `heads < 2**16`,
and `BLOCK_SIZE <= 2**20`. All PIDs, lanes, metadata, and six stride operands
are widened to int64. The checks bound all formed indices, including masked
lanes, below `2**63`, and all formed four-byte aligned pointers within uint64.
Active coordinates also satisfy actual Tensor storage bounds. There is no
additional total-numel bound or contiguity assumption.

The row formula uses a single padded global maximum, then FP32 subtraction,
exponentiation, sum, and division in the source order. The existing full-row
source-success and native lowering proofs are reused from token_softmax_llama
and softmax_triton1. The selected warp count is 4, 8, or 16 at the original
block thresholds. The native reference retains the ordered reduction tree,
thread ownership, and primitive instruction meanings. It permits arbitrary
raw FP32 words, including exceptional values. Primitive hardware instruction
semantics and the selected native layout remain explicit parameters; no
whole-row result, execution-success premise, numerical rewrite law, or
unresolved numerical experiment is substituted for the kernel proof.

`Kernels.lean` exposes the shared readable DSL source and a complete candidate
DSL kernel. `token_softmax_bloom_index_commuted.py` changes exactly one integer
metadata-address product: `cur_batch * stride_length` becomes
`stride_length * cur_batch`. `KernelRewrite` proves program equivalence under
empty assumptions and equality of complete source/candidate executions,
including failure behavior. Floating-point operation order is unchanged.

`IO.lean` declares explicit metadata, FP32 payload and output ports for every
scheduled segment/head, with the original strides, active masks and empty
scratch. Logical metadata uses the existing natural-index projection; the
unchanged Tensor bridge proves its signed INT32 load/int64 widening meaning.
`Correct.lean` exposes one shared `ProgramIO.ImplementsOn` specification and a
one-call proof. Its independent finite softmax is the full segment formula
under the visible coverage condition. Logical uniqueness identifies PID/lane
events, so repeated identical launches remain admitted.

`FPEquiv.lean` exposes one standard complete source equivalence with no
numerical assumptions. `Proofs/IOCorrect.native` separately provides the full
native `ProgramIO` observation, including the exact reduction graph and lane
ownership. `Misc/KernelRealResults.lean` and `Misc/KernelNativeResults.lean`
retain the exact prior raw grid results, including one common raw state for
source and candidate. `Misc/TensorBridge.lean` retains all previous Tensor results,
logical values, native caller cells outside writes, and all read allocations.

`Proofs/KernelSafety.lean` proves both unconditional metadata loads and every
active source payload/store access. `Proofs/IOSafety.lean` propagates the
unchanged read regions through the actual successive launch states.
`IOBinding`, `IOSchedule`, `IOResults`, and `IOScalar` connect those facts to
logical channels and reuse the existing complete execution proofs.

`KernelTensor` binds actual shared Torch Tensor objects, physical indices,
data pointers, strides, and caller cells. Its address-indexed pointer views
interpret a single raw-word reader separately for payload and metadata loads,
so legal cross-dtype read aliasing is not excluded by incompatible cell tags.
`KernelAddresses` and `KernelBinding` connect finite-width metadata pointers,
actual signed-int32 loads, signed int64 extension, masks, payload address
arithmetic, and original logical Tensor coordinates. `KernelGrid` and
`KernelGeometry` derive unique active writes and physical frames from the
unchanged admission checks, including singleton zero-stride outputs. No host
allocation protocol or unified final-byteheap framework is imported by these
public roots. Historical host roots are preserved separately in
`Misc/Legacy/HostCorrect.lean` and `Misc/Legacy/HostFPEquiv.lean`.

Validation evidence lives in
`.veritile-cache/kernel-evidence/token_softmax_bloom/`. The unchanged canonical
checker passed 14 admission cases and 32 rejection controls. Independent
source review passed 18 exceptional-value, segment, stride, and no-launch
controls. The genuine candidate passed 46 complete-storage bit comparisons,
including boundary widths, offsets, transposes, caller canaries, zero-stride
empty metadata, and shared FP32/int32 read storage. Lean build receipts pin
the current source and object hashes; public audits check ordinary-base axioms
and independent references.

## Public kernel and supporting interfaces

`Correct.token_softmax_grid_real` states the shared IO judgment over the
independent logical softmax. The complete schedule, source geometry, region
separation and logical coverage/uniqueness remain visible. `FPEquiv` keeps the
single `token_softmax_source_equiv` headline. Neither entry exposes a raw
state, executor, existential final memory or Tensor precondition.

The ordinary `IOCorrect.native` theorem retains all FP32 input words with no
finite-value restriction and no coverage condition beyond the original native
scope. It proves all selected outputs and the exact exterior frame. The prior
raw Real/native headlines are retained in `KernelRealResults` and
`KernelNativeResults`; full source/candidate grid equality remains available.

`TensorBridge.token_softmax_kernel_correct` and
`TensorBridge.token_softmax_kernel_fp32` retain the complete Tensor statements.
They import supporting foundations, never the public reader entries. This
format pass changes no Python source, input contract, stride, precision or
numerical boundary, and runs only targeted Lean checks.

Optional proof layout (2026-10-21): `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
Historical wrapper readers are physically retained in `Misc/Legacy/HostCorrect.lean`, `Misc/Legacy/HostFPEquiv.lean`. They remain optional historical interfaces and are checked separately from the current primary/auxiliary import union.

`Misc/KernelRealResults.lean` and `Misc/KernelNativeResults.lean` retain the optional raw-grid reader theorems; the required source/IO helpers remain in `Proofs/`.

The two moved raw readers explicitly simplify the existing strict `KernelReal.model_profile` / `KernelNative.model_profile` certificates when applying `KernelCore.complete`. Their statements, domains and targets are unchanged; these are proof connections to the existing fixed models, with no new numerical rule.


## Raw function body correspondence

The selected repaired Python function is unchanged. Its eleven top-level
statements are pasted verbatim in the shared token_softmax_llama/Kernels.lean
quotation, including both program-ID casts, the arange cast, both loaded
metadata casts, padding, reduction order and store mask. Independent manifests
check the Llama and Bloom originals against that one shared source. Both existing Python candidate files are unchanged and checked against the
shared length-address multiplication candidate quotation.

Invocation metadata records signed INT32 start/length storage and the existing
natural-valued mathematical IO channels, signed INT64 runtime strides and the
block constexpr. These logical channels do not prove native nonnegativity or
width; the existing Tensor binding, content admission and native index proofs
retain those obligations. FloatDType remains a caller parameter covering all
previous floating channel cases; the checked Python wrapper still requires
FP32 payload and output. No input domain is narrowed for the migration.
