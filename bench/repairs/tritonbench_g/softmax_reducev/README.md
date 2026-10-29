# softmax_reducev: explicit contiguous Tensor contract

`token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len,
max_input_len, other_kv_index)` retains the original mutation API: it writes the
supplied `o` and returns `None`. It accepts the latest accepted repair's native
contiguous layout: FP32 logits `[heads,tokens]`, values `[pages,heads,D]`, and
output `[batch,heads,D]`, with signed-int32 page table `[batch,max_input_len]`
and rank-one starts/lengths `[batch]`. All six tensors are strided CUDA tensors,
contiguous in the default format, on one device, with no negative view bit.
Nonzero storage offsets are supported. No payload allocation, clone, normalization, or copyback is inserted.
Scalar metadata is read onto the host for validation before launch.

`D` is a positive power of two, and `max_input_len` is a positive Python integer.
The accepted repair's signed-int32 contract is retained: every tensor numel and
the maximum are at most `2**31-64`, leaving room for the final 63 masked lanes.
Empty batch/head calls return without pointer queries or launch. Nonempty calls
also check grid limits (`batch < 2**31`, `heads < 2**16`), the frontend tile limit
`64*D <= 2**20`, four-byte alignment for nonempty tensors, and physical separation
of the output's storage word span from each read allocation. Read allocations
may share storage. Canonical contiguous strides are passed to the unchanged
source dimensions, so irrelevant singleton strides do not narrow the contract.

For launched calls the wrapper reads signed-int32 starts and lengths and checks
nonnegative starts, positive lengths, lengths at most the maximum, and logits
windows within `tokens`. It then reads only active right-aligned page entries
and checks `0 <= page < pages`. These scalar observations synchronize with the
current stream; invalid metadata raises before mutation or kernel launch.
`InputContract` includes the content checks, `check_input_iff` proves admission
correspondence, and `checked_content` derives the historical `Content` premises
from scalar decoding and the checks. The native entry decodes signed int32 words
from the same caller memory. No-launch branches skip content reads. Unused page
cells are unconstrained. The Real theorem uses finite active logits and gathered
values; unused values may be NaN. The structural native theorem retains all
FP32 word patterns under the same checked metadata domain.

The kernel fix is the latest accepted repair: padded page indices become zero,
and the V load uses the active-key mask with `other=0.0`. Therefore padded lanes
neither dereference the sentinel nor multiply zero by an unused NaN V value.
`other_kv_index` remains in the signature and is unused. The original online
maximum/sum/accumulator order, 64-key blocks, one warp, three stages, default
fusion and terminal division are preserved.

`Kernels.lean` contains only the complete source and source parameters. The
historical loop, Real, native graph, profile and complete-grid proofs are reused
under `Proofs/`. `Contract.lean` uses shared Torch metadata and storage
observations. `check_iff` proves metadata admission correspondence;
`checked_admissible` derives the historical index/shape domain from checked
metadata plus the derived content facts. `checked_extents` and `checked_accesses`
connect every accessed logits/value/output/page/start/length offset to the
queried tensor sizes. Address helpers prove signed-int32 active and padded
indices, exact masks, finite uint64 active pointers, and storage bounds from
actual shared FP32 Tensor laws and ordinary int32 storage API laws.

The independent entries `Correct.tensor_correct` and `FPEquiv.tensor_fp32`
execute every source program in a complete `(batch,head)` schedule, retain the
actual supplied shared `VeriTile.Torch.FP32.Tensor`, return `None`, establish every
output channel, and preserve every other caller storage cell, including output
prefix/suffix canaries. Both cover empty calls. The public entries instantiate
local execution with the literal complete source and prove successful
completion; they have no abstract whole-kernel runtime or output-formula premise.
Pointer views and final write interpretation are mathematical storage mappings,
not Python snapshots or copyback. The preserved `memory_evaluate`,
`native_memory_evaluate` and `execution_bridge` identify these evaluators with
source execution on the reached caller allocation state.

The ordered native graph has seven scalar raw32 instruction ports: maxNum,
subRN, mulRN, addRN, divFull, fmaRN and exp2Approx. It preserves the D128 reversed
last denominator shuffle and the per-channel contender for D16/D32 shared
writers. `local_refinement` assumes only scalar instruction agreement. The
profile argument describes actual compiler layout and replica choices; it is
not a numerical reduction oracle. Historical observed SM80 profiles cover
D=1,2,4,8,16,32,64,128 and unaligned D16. Generic power-of-two source/math scope
is distinct from this saved compiler evidence. GPU scalar numerical experiments
remain deferred; no FP-to-Real equality is claimed.

The trusted boundary is ordinary Torch metadata, contiguity, physical allocation
and raw-word interpretation, the selected compiler instruction/layout meanings,
and successful race-free GPU execution. Targeted source correspondence is a
checked Python/Lean binding, not a verified Python compiler.

Validation: targeted Lean compilation and standard-base axiom checks; CPU source
and host checks covering 19 admitted online-loop/empty cases and 36 rejections.
The CPU harness executes the repaired JIT body with pointer views, verifies
identity and caller-storage frames, and reproduces the padded-NaN/invalid-sentinel
regressions. It does not validate GPU numerical instruction conformance.
