# Multi-token paged K-cache copy

For token `s`, let `q = s // n`, `p = lengths[q] - n + s % n`,
`block = tables[q, p // block_size]`, and `slot = p % block_size`.
For every head `h` and component `d`, the independent target copies initial
`K[s,h,d]` to cache coordinate `(block,h,slot,d)` in the legacy layout, or
`(block,h,d//width,slot,d%width)` in the split layout. Other cache elements and
all other memory regions are unchanged. Each universally quantified program
covers one `(token,head,split)` tile; no restriction to `n=1` is introduced.

## Source and contract

`Kernels.lean` is the canonical readable typed Triton source used by both public
proofs. FP32 payloads are copied without arithmetic; metadata storage is INT32,
widened to signed INT64 before address arithmetic. The candidate changes only
`token*stride_kt + head*stride_kh` to `head*stride_kh + token*stride_kt`.
Both actual JITs are present in `kcache_copy_triton_repaired.py`; their public
wrappers share `_checked_copy` and exactly the same assertions. There is no
performance claim for this exact integer-address rewrite.

The contract retains contiguous strided FP32 K/cache and INT32 lengths/table,
one CUDA device, no unresolved negative views, aligned nonempty data pointers,
and disjoint data spans. Source rank 3/4 is admitted; rank 4 reshapes its first
two axes without copying. Lengths/table have rank 1/2 and one row per sequence.
The positive Python integer `n < 2^31` divides token count; the layout selector
is Boolean. Heads, dimensions and block size are positive, and split width is
a positive power of two. Legacy cache shape is `[blocks,heads,block_size,dims]`;
split cache shape is `[blocks,heads,splits,block_size,width]` with
`splits*width=dims`. The existing index, grid and tile resource bounds remain.
Sequence lengths lie between `n` and `block_size*table_width`; selected block
IDs are valid and every selected `(block,slot)` pair is unique. Empty source
counts are admitted and launch no programs. No payload normalization is added.

`Contract.lean` supplies the independent per-program formula, selected contiguous
strides and initial-memory binding. `Proofs/Contract.lean` retains exact Python
admission; `Proofs/KernelContract.lean` proves the signed position, layout and
physical-to-logical binding equations.
`logicalState` reads payload and INT32 metadata at the actual Tensor `dataPtr`
plus the four-byte element offset. `logicalState_binding` derives the kernel's
metadata premises from the checked physical readback; `logicalState_payload`
connects its source cells to arbitrary caller values. Neither assumes outputs.

## Public proofs and retained results

`Correct.lean` exposes the shared `ProgramIO` interface: two signed metadata
inputs, the FP32 payload input, the cache output, every read/write window and
mask, the actual source and all three program-ID axes. Its single specification
uses `ProgramIO.ImplementsOn` to prove actual source access safety, successful
execution, the independent identity-copy formula and the complete exterior cell
frame. The original destination injectivity condition is explicit on the
logical metadata; no new alias restriction is added. The mathematical backend
interprets the only reached floating primitive, FP32 payload load, as identity.
The original Algorithm projection result remains in `Misc/KernelRealResult.lean`.
`TensorBridge.tensor_physical_correct` retains the actual-pointer result.
`TensorBridge.tensor_input_correct` concludes the admitted K Tensor's logical values through
the input-only `InputReadLaw` in `Proofs/InputBinding.lean`. This generic law ties
logical inputs to the same physical payload heap at the actual metadata-derived
addresses; it assumes no kernel, output, or numerical identity.
`FPEquiv.lean` exposes both actual source schedules on the same visible IO and
compares `original ≡[[]] candidate` through the shared actual-schedule program
view. `Proofs/Pair.lean` reuses the existing `ProgramIO.Equivalent` theorem;
`IO.lean` retains both concrete source schedules and observations. It proves both traces safe, both executions
successful, equal output cells and both full exterior frames. This equivalence
also covers colliding destinations; it does not require the Real specification's
injectivity condition. The stronger unconditional complete typed-memory equality,
snapshot copy and exterior frame remain in `Proofs/KernelEquivalence.lean`,
together with the original empty floating-assumption audit. Payloads are unrestricted, including NaNs, infinities, signed
zero and subnormals. There are no deferred numerical transformation assumptions.
The public `TensorBridge.tensor_kernel_results` uses `logicalFPState` of the same admitted
metadata and physical payload heap, with the same `InputReadLaw` as Real. It
retains both successful executions, their snapshot/equality/frame result and
the preserved logical K source values on both sides.

The old finite/event interpretation is retained in `Proofs/Model.lean`.
`Native`, `Source` and `Execution` retain the existing signed-control widening,
nonwrapping pointer, whole-grid and independent copy results. `EventCorrect`
retains the previous whole-grid Real event result; `HostCorrect` retains the
optional historical wrapper/object/byte-frame theorem. Neither public entry
imports the optional host proof. These results do not replace DSL execution.

The original benchmark is unchanged. Its unwidened INT32 pointer product has
the concrete counterexample `token=65536, stride_kt=65536`: the product wraps
to zero instead of `2^32`. For example, a contiguous legacy input with shape
`[65537,1,65536]`, `n=1`, unit block size and distinct valid blocks admits that
program coordinate; different values at `K[0,0,0]` and `K[65536,0,0]` expose the
wrong source read. The repaired source widens before multiplication.
This original defect remains distinct from both the explicit contract and the
address-commuted candidate. Shared grid-product overflow/segmented-launch work
remains deferred; the historical guard is retained without new compatibility work.

## Validation and modeling boundaries

Run `python bench/repairs/tritonbench_g/kcache_copy_triton/check_contract.py`.
For each JIT variant the CPU AST replay passes 7 admissions and 65 rejections,
covering both layouts, multi-token block crossings, rank-4 reshape, storage
sentinels, raw FP32 classes, invalid metadata, pointer alignment/overlap and
the original pointer-product overflow. The checker pins the original repaired
AST and verifies the candidate differs only by the selected integer add swap.
CUDA context and launch are mocked; no GPU execution or independent GPU
numerical experiment is claimed.

Targeted Lean checks cover the public entries and retained optional helpers,
with standard-base axiom audits. Contiguous Tensor backing, same-format
load/store behavior and compiler/device execution are platform model boundaries,
not numerical experiment TODOs. Relevant API/model sources are
[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
[Triton store](https://triton-lang.org/main/python-api/generated/triton.language.store.html),
[PyTorch data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
[PyTorch is_contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html),
and [PTX memory access alignment](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).

`Misc/TensorBridge.lean` retains the optional physical/logical Tensor applications
and Python-check correspondence as ordinary theorems, with their original
statements. Both primary files use only raw source geometry/program coordinates
and shared IO judgments; execution states, metadata binding proofs and scatter
implementation details are in `Proofs/`. Neither primary imports the bridge.
`Proofs/SourceSafety.lean` checks the actual dependent metadata gathers and payload
load/store steps. `Proofs/PublicIO.lean` derives their address bounds from shared
input binding and the displayed windows, reusing the existing complete execution
and snapshot/frame proofs. No source, Python guard or numerical law was changed.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.


The source readability cleanup replaces `repairedKcacheCopy%` with two complete
ordinary `triton` quotations in `Kernels.lean`, keeping the same public source
functions and exact input/output buffer lists. The campaign preservation check
proves equality of the complete old and new `kernel` functions for both
variants by `rfl`. This changes no source program, candidate, contract, precision,
proof statement or acceptance requirement.
