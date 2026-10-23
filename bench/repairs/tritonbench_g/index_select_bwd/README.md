# Index-selection backward accumulation

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

Current pair entry (2026-10-21): `FPEquiv.adaptive_tail_equiv` compares the
actual fixed-width launch with a new tail-aware launch. The original always
uses BLOCK=512; `index_select_bwd_adaptive.py` chooses the smallest supported
power of two covering the actual slice size. For example, a 513-column row
uses blocks 512 and 1 instead of 512 and 512, reducing inactive lane work.
The source kernel is genuinely shared: both wrappers invoke
`index_select_cat_bwd_repaired_kernel`, with distinct BLOCK specializations.
No benchmark speedup claim is made. The complete wrapper AST check verifies
that validation, index ordering and slice pointers differ only in BLOCK.

The main `≡[[]]` proof uses the shared complete `ProgramIO` implementation
judgment: both sources execute safely, all active output words agree, and both
exterior memory frames hold. It admits arbitrary FP32 words and uses no
numerical assumptions. `Program.adaptiveRun_eq` also proves equality across
all rows and slices, preserving the original duplicate-index addition order.
`Correct.index_select_bwd_complete_real`, retained as an ordinary theorem in
[`Misc/CompositionCorrect.lean`](Misc/CompositionCorrect.lean), connects the
original complete source composition to the independent full derivative formula; candidate equality
preserves that result. Optional native statements remain ordinary theorems in
`Misc/NativeCorrect.lean` and do not replace the pair theorem.

The independent backward target is

`G′[r,c] = G_entry[r,c] + Σ { H_entry[i,c] | index[i] = r }`.

The original benchmark performs a plain overwrite store. It does not add and
contains no atomics. For `G=[[5]]`, `index=[0]`, `H=[[2]]`, it writes 2 although
the backward target is 7. Repeated indices additionally race across its GPU
programs. Its prior candidate commutes only `pid1 * BLOCK_SIZE_COL` to
`BLOCK_SIZE_COL * pid1`; both variants have the same accumulation defect.
`Proofs/OriginalRewrite.lean` preserves that exact historical FP32/FP32/int64
pair and proves successful complete executions with equal memories. This
separate result certifies the address rewrite, not the accumulation repair.

The repair keeps the supplied destination and adds one incoming row at a time
in ascending input-index order. Each row uses consecutive slices of at most
512 columns. A launch loads the current destination and incoming FP32 words,
performs one FP32 addition with the current value as the left operand, and
stores back through the same destination pointer. Duplicate rows therefore
use an ordered left fold seeded by the original gradient. There is no
reassociation or atomic operation. The adaptive candidate changes only the lane block width.

`Kernels.lean` is the authoritative readable typed Triton source shared by
both public entries. `Contract.lean` defines the independent Real derivative,
ordered native graph, and single-slice update. `Correct.lean` proves successful
complete kernel execution, the independent update, complete typed-memory
equality, and exterior frame over arbitrary Real operands. The optional
`Misc/NativeCorrect.lean` theorem observes the same source against the independent native FP32 add graph,
with successful execution and complete memory equality. All FP32 bit patterns
are admitted. The native scalar add is a semantic parameter; no algebraic law
or native-to-software-IEEE conformance assumption is deferred to experiments.

`Misc/TensorBridge.lean` contains the checked Real and FP32 applications, Python-check
correspondence, and checked physical access bounds. `Correct.lean` contains the
single independent Real kernel specification; `FPEquiv.lean` exposes the actual
adaptive comparison. The optional full row/slice composition result is imported
separately from `Misc/CompositionCorrect.lean`.

The Real entry and optional native entry state the shared `ProgramIO` judgment over independent
logical arrays. `IO.lean` displays both FP32 input windows, the destination's
in-place output window, active mask, and the actual source invocation.
`Proofs/AccessSafety.lean` checks the source's signed lane conversion, clamped
addresses, and every active load/store against the declared capacities.
`Proofs/Interface.lean` composes that safety proof with the existing complete
execution proof. These IO statements expose neither raw states nor post-state
memory formulas; the underlying exact-cell output and exterior frame remain
part of the shared judgment and the retained Tensor application.

`Proofs/KernelBinding.lean` derives the source operands from the original
caller Tensor objects and one current byte-memory map, retaining storage
offsets and earlier duplicate-row updates. Shared Tensor laws connect direct
pointer loads to logical Tensor words. Existing checked extent, signed-index,
int64-lane and physical-span lemmas bound every active and clamped pointer.
Kernel indices name a host-selected row view; the repaired kernel itself has
no index pointer. The full destination backing interval must be separate from
both read roles; the incoming and index roles may share storage.

The checked domain remains native contiguous FP32 CUDA gradient matrices and
a contiguous int64 index vector on one CUDA device. Strided layout, resolved
negative/conjugate flags, natural pointer alignment, matching gradient stride
tuples and widths, destination row count at least incoming row count, one
index per incoming row, byte extents below `2**63`, and every used signed index
in range are checked before writes. Nonzero storage offsets and irrelevant
singleton strides are admitted. Empty row or column work returns `None`
without a kernel launch or index-value read. Gradient payloads remain in their
original storage; only index metadata is copied to the CPU.

The prior whole-call proofs are retained as optional `Proofs/HostReal.lean`
and `Proofs/HostFP.lean`. Their existing `Program`, `Logical`, `Native`, `Host`
and `Physical` chain proves ascending duplicate order, the full derivative,
byte frames, and wrapper behavior. The historical conditional software-IEEE
helper remains optional in `Numerics`; it is not a required FP equivalence,
a numerical experiment task, or a public acceptance premise. Kernel proof
acceptance does not require a new host lifecycle or launch-order proof.

`check_contract.py` executes the actual repaired Triton source in its CPU
interpreter with an explicitly labeled CUDA metadata/context facade. The
current run covers 18 wrapper cases (10 empty), 55 repair launches, a shared
read-storage case, and 16 rejection-before-launch cases. Four additional
interpreter launches exercise the unchanged original source and its exact
single-expression column-product candidate, reproducing the nonzero-seed
and duplicate overwrite defects. GPU race outcomes and GPU primitive
conformance are not claimed. Original benchmark files remain unchanged.
