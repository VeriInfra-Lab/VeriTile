# Kernel boundary review

The canonical selection is the `contract_revision.source` entry for
`bgmv_shrink_kernel` in `validation/task_progress.json`:
`bgmv_shrink_kernel_repaired.py`, SHA256
`7152c240f53ba0c35f44fc3967491bb31e02c60a0cbe235cafdcfacaa0c3018d`.
This review leaves that source, its guards, all existing Lean proofs, the
historical README and numerical/source pins unchanged.

The independent real target starts from the supplied output:

`Y_after[b,n] = Y_before[b,n] + scaling * sum_k X_before[b,k] * W_before[index_before[b],n,k]`.

Sentinel rows (`index=-1`) and inactive output coordinates retain their input
values. The native graph retains matching FP16 or BF16 products and reduction
joins, their extension into the FP32 accumulator, FP32 scaling, and 64 separate
FP32 atomic updates into the existing output. Each cell has its actual atomic
order; an atomic update is not modeled as an ordinary reorderable addition.
The checked source admits both int32 and int64 metadata, contiguous 3D weights
or contiguous 4D weights with a singleton second axis, aligned offset views,
K=0, sentinel rows and empty launches. It rejects lazy/strided inputs.

Two source boundary defects prevent asserting that target and successful
execution over every call admitted by the current checks.

## Storage identity does not exclude physical overlap

The wrapper rejects output aliases by comparing `_cdata`. A DLPack round trip
can create a new storage identity over the same bytes. The new self-contained
checker uses real PyTorch DLPack storage and pointers, then runs this task's
actual extracted wrapper and kernel AST. Only CUDA admission/context and Triton
operations use CPU adapters.

For `B=L=N=1, K=257`, output aliases the input or weight at byte offset 512.
All pointers meet the 16-byte alignment guard and storage identities differ.
The initial FP32 output word `0x3f803c00` is `1.0018310546875`; its low halfword
is the FP16 value 1 at reduction coordinate 256. Coordinate 0 contributes 1.
In split order 0 through 63, split 0 adds 1 to output before split 1 loads that
coordinate. The low halfword changes to FP16 `0.005859375`. The actual source
therefore returns `2.0076904296875`, whereas the independent input snapshot
target and the disjoint-storage control both return `3.0018310546875`.

Sixteen complete source executions cover both half formats, X and W overlap,
both metadata widths, and both weight ranks. All 64 atomic updates execute in
each case, and the read operand's physical backing frame is also violated.
BF16 has the same mechanism: initial output `1.0019378662109375`, snapshot target
`3.0019378662109375`, actual source output `2.0019378662109375`.

Four further cases alias output with int32/int64 metadata for both half input
formats. Initially the adapter index is 0 and all guards pass. Split 0 adds 1
to the FP32 output, changing that index's word to `1065353216`. Split 1 reloads
the changed index and actively reads weight element `273795776768` from a
257-element backing. The CPU pointer adapter reports this access before reading
it. A disjoint control completes with output 2 and unchanged metadata 0.

Same-`_cdata` aliases are rejected before launch as a positive control. Existing
Lean results remain conditional on their physical binding/separation premises;
this review does not claim a contradiction in those proofs. The canonical
identity check alone cannot establish those physical premises for these calls.

## Missing bound on the broadcast tile

The wrapper admits `B=L=K=1, N=C=4097`, with small actual input allocations and
valid metadata 0. It selects `BLOCK_N=8192`, `BLOCK_K=256`, `SPLIT_K=64`.
The mask at `b_ptr_mask` broadcasts to 2,097,152 elements, above Triton 3.7.1's
1,048,576-element tensor limit. The actual source frontend raises:

`ValueError('numel (2097152) exceeds triton maximum tensor numel (1048576)')`.

This happens both with an explicit manual `ASTSource` signature and with the
actual JIT-generated binder and `_pack_args` feeding `ASTSource.make_ir`.
The latter records the actual K=1 and stride=1 constexpr specializations.
N=3 and N=4096 pass the corresponding frontend checks. No large tile enters
optimization or PTX generation; the limit fails during tensor-type construction.
The N=4096 result establishes frontend acceptance only.

## Precision, ABI and validation limits

Four small full offline compilations cover FP16/BF16 and int32/int64 metadata.
Saved TTIR checks the half-format product and reduction joins; the FP32 output
instruction is `atom.global.gpu.acq_rel.add.f32`. Scaling is an FP32 runtime
parameter in these compilations. Separate generated-binder observations include
finite Python scales 1, 1e-50 and 1e300; all select `fp32`. Thus Python finiteness
does not itself state finiteness or nonzero status after FP32 conversion. These
observations do not assert a third source defect or universal scalar conformance.

Manual signatures and binder-derived cases are recorded separately. Binder
probes use the real installed binder and an explicit CUDA backend with CPU
Tensor dtype/pointer metadata; GPU device/stream/launch machinery is unexecuted.
The source replay chooses the concrete sequential split order 0 through 63;
it does not claim that every GPU schedule yields those numbers. Counterexample
reductions have one nonzero exact product per contributing split, and each has
a disjoint control that agrees with the independent snapshot target.

The new checker passes 16 output counterexamples, four metadata access
counterexamples, their 20 disjoint controls, 11 additional controls, six frontend
cases, 12 scalar ABI observations and four small PTX profiles. Additional controls
cover K=0's 64 atomic updates, sentinel rows, existing nonzero output, scaling,
empty launches and same-storage rejection. The unchanged historical checker
still passes its 18 complete cases, 37 rejections and partial-write frame check.
No GPU was executed. No new whole-domain correctness theorem is claimed.

Run `check_kernel_boundaries.py`; its default output is the workspace directory
`.veritile-cache/bgmv-shrink-kernel-evidence/replay`. Compiler caches and temporary
files also remain there. The delivery packet pins the canonical selection,
all pre-existing task sources, the checker, this report and the saved evidence.
