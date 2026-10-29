# matmul_kernel whole-call repair

The original caller advertises M/N/K and caller tensor strides, but the kernel silently uses 4096 and fixed contiguous strides. It also loads K tails and stores M/N tails without masks. The pinned original defect report demonstrates out-of-bounds loads, omitted features, wrong transposed/strided inputs, and output-hole corruption. The repair passes dimensions and all six actual strides, masks K loads with zero, masks output stores, and skips empty M/N launches. The three-argument tl.dot with its live FP32 accumulator, default fusion/TF32 behavior, explicit FP16 cast, caller-supplied block sizes, and None return are preserved.

## Selected ordinary caller contract

A, B and C are ordinary rank-two FP32 CUDA tensors on the same device, with exact shapes (M,K), (K,N), (M,N), arbitrary valid storage offsets, positive strides, no lazy negative/conjugate bits, and naturally four-byte-aligned nonempty data pointers. C must use storage independent from each nonempty input; read-only A/B aliases are allowed. Each nonempty view has separated rows (sr >= cols*sc) or separated columns (sc >= rows*sr). This admits transpose, stride-two and padded row/column views; it does not narrow the task to contiguous tensors or 4096. View addresses start at the actual data pointer, so ordinary storage offsets are represented by that origin and capacity is the remaining allocation span. PyTorch's ordinary valid-storage-view boundary supplies capacity for the last addressed element. Scalars are nonnegative exact Python integers. BM/BN/BK are caller-selected powers of two at least 16, with ordinary successful compilation and resource legality; no fixture-only block restriction is introduced. The wrapper selects A’s CUDA device for the actual launch and restores the caller’s prior device afterward.

For nonempty output, the validator bounds all strides, dimensions, padded M/N coordinates, padded K plus a final pointer increment, and all formed source element offsets below signed32. These are source-specific arithmetic bounds. The separate generic grid-product/runtime-limit project is deferred. Empty M/N returns None without launch after metadata checks. K=0 with nonempty C launches, writes zero through the FP32→FP16→FP32 path, and returns None.

## Independent target and numerical meaning

`mathematical` is the coordinate sum S[r,n] = sum(k<K) A_entry[r,k]*B_entry[k,n]. `value_mathematical` proves that source K blocks, 8-feature atoms, and masked padding cover that sum exactly in a Real idealization. This is a separate mathematical interpretation, not an FP-to-Real accuracy claim; the Real cast interpretation is the identity.

The native result is `Native.reference`, an independent ordered sequence of fixed K=8 TF32 ISA result-lane calls followed by FP32→FP16 narrowing and FP16→FP32 widening. It uses original logical A/B coordinates, zero for K padding, all K blocks and the live prior accumulator. It retains the padded native operations even when they can affect signed zero or nonfinite words. The domain allows every raw input word, including NaNs and infinities.

Actual-binder offline Triton 3.7.1/SM90 artifacts preserve default 4 warps, 3 stages and fusion. Small blocks lower to mma.sync.m16n8k8.row.col.f32.tf32.tf32.f32; larger blocks lower to wgmma.m64n{16,32,64,128,256}k8.f32.tf32.tf32. The emitted PTX passes input FP32 words directly to the TF32 instructions; the model does not invent a cvt.tf32 instruction. The selected finite opcode and its ordinary fragment routing/ABI are compiler metadata. Each numerical port handles one fixed instruction result lane (eight A and eight B operands plus a live accumulator), or one scalar conversion. No whole dot, arbitrary K reduction, execution, or output equality is an assumption. Instruction internal precision, rounding, nonfinite behavior and packed-conversion conformance remain explicitly deferred local obligations.

## Proven connection

`Kernels` contains current-memory modulo row/column loads, K masks, ordered nested K loops, final casts, full two-dimensional launch and active store masks. `Source` models program vector stores from pre-store memory and proves their equality to sequential cell execution, using read-only A/B. `Geometry`, `Scheduling` and `Bounds` prove all-output coverage, unique program/lane and K-feature ownership, separated-layout address injectivity, all active allocation bounds, and padded formed-pointer bounds including final increments. Executions under any permutation of the active store schedule agree.

`Physical` uses actual little-endian four-byte loads and stores and proves their connection to the word execution, all output bytes, and preservation of every byte outside C's logical footprint. This includes all A/B bytes, C holes, and other allocations. `Host` proves empty paths, K=0, supplied C identity, and Python None return. `Misc/Whole.lean` exposes `whole_real`, `whole_native`, and `whole_fp`; the historical host readers are retained in `Misc/Legacy/HostCorrect.lean` and `Misc/Legacy/HostFPEquiv.lean`. No numerical callback is present in the Real or fixed-native-interpretation theorem; whole_fp has exactly three local fields: mma, narrow and widen.

## Evidence and boundaries

The 15 accepted modules and new checked Contract/Correct/FPEquiv entries compile. `Checked.admitted_domain` derives block geometry, separated-view validity and the signed32 source envelope from the implemented validator plus ordinary allocation spans. `checked_real`, `checked_native` and `checked_floating` reuse the complete accepted proofs. The targeted `check_contract.py` passes 26 actual source Triton CPU interpreter calls covering valid storage offsets, input aliases, transposes, step2/padded layouts, all tails, multiple K blocks, empty/K0, output-hole frames, exact None and FP16 rounding (2051→2052), plus nine guard rejections including foreign unaligned storage. CPU CUDA residency is shimmed; no native GPU execution is claimed. The accepted fixed instruction and scalar conversion numerical ports remain deferred.

The Lean interpretation is source-specific and explicitly documented, with AST/statement binding and selected lowering artifacts. There is no shared typed-DSL translation certificate or proved Python/PyTorch/Triton compiler. Ordinary tensor storage, JIT argument binding, compiler instruction/fragment routing and legal runtime completion remain external realization boundaries. No new source acceptance is credited to the original defective program; root review/promotion is separate.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `596b295a94f32ce1ff9d6d77942dfec3a68b067715f1750688c41a42cb58e3af`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(a.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/TensorBridge.lean`, `Misc/Whole.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
Historical wrapper readers are physically retained in `Misc/Legacy/HostCorrect.lean`, `Misc/Legacy/HostFPEquiv.lean`. They remain optional historical interfaces and are checked separately from the current primary/auxiliary import union.
The unchanged `Public.Admitted` structure is shared through `Proofs/Admitted.lean`; `Misc/Whole.lean` retains the optional host claims and theorem applications.
