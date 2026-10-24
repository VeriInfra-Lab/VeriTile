# attention_kernel_aligned: complete kernel and caller Tensor results

The selected Python is unchanged: `attention_kernel_aligned_repaired.py`, SHA256 `365cda104f62655345ad002a7d2815fba0d3c812e20c5a569e9b0214c92c85ea`. The original benchmark remains unchanged, SHA256 `ca0ecc71aeb5d33c1e216ec1891a726281fae909e8ee765945a3bb12414453b9`. Before this revision, all 92 existing repair and benchmark files were copied and pinned under `kernel-evidence/preserved/`. Historical declarations remain available. New public endpoints prove direct execution of the existing readable kernel quotations.

## Contract and independent mathematics

Five ordinary strided CUDA tensors share one device and either FP16 or BF16 storage. Q/K/V/Out have shape `[B,H,N,D]`; contiguous bias has shape `[B,H,N,128]`. D, BM and BN independently belong to `{16,32,64,128}`, warps to `{4,8}`, stages to `1..4`, and `N <= 64*BN`. The existing Python resource guard remains unchanged: it rejects BN=D=128 with stages=4 and BM=BN=D=128 with stages=3. Exactly 1004 of the 1024 dtype/configuration pairs are admitted by that guard.

Every live Tensor has positive strides. Each numerical Tensor retains its own batch/head/row/column strides, including permutations and gaps. Full storage intervals are pairwise disjoint. The sorted output-stride/span scan proves nonoverlapping output coordinates. Grid Y is at most 65535. The existing signed-i32 and unsigned64 pointer checks include padding and final pointer advances, with two-byte alignment. Empty B/H/N returns before unused live stride, pointer, separation and address checks. No guard was changed.

The scalar is a finite ordinary Python int/float in FP32 range. The contract retains the observed FP32 launch word, including signed zero, and its decoded Real projection. Exact type/scalar classification and ordinary API observations are explicit boundaries.

For each batch/head, the independent exact-Real target is:

```
s[i,j] = (scale * 1.44269504) * sum_d Q[i,d]*K[j,d]
       + bias[i,j//BN] + bias[i,64+(j%BN)%64]
out[i,d] = sum_j 2**s[i,j]*V[j,d] / sum_j 2**s[i,j]
```

This noncausal target includes every key. The bias is not multiplied by log2(e). The Real interpretation erases rounding; it makes no native-to-Real equality or accuracy claim.

## Kernel-only results and Tensor binding

`Contract.lean` exposes the actual Tensor `Call`, `InputContract`, `TensorLaws`, acceptance equivalence and derived access bounds. `KernelOnly/Inputs.lean` derives the source parameters from shared Torch metadata and observed original strides, pointers and full allocation spans. It computes the reversed sorted-axis scan. Tensor laws state only ordinary coordinate, capacity, pointer and contiguity facts. Actual input/output addresses and finite TensorStorage bounds are derived. Positive unusual singleton bias strides remain allowed; only its nontrivial last dimension forces stride one.

`Correct.kernel_correct` and `BF16.Correct.kernel_correct` prove successful direct `Whole.runPrograms` execution for any permutation of the whole submitted grid and every caller logical output against the independent formula. `FPEquiv.kernel_native` and its BF16 counterpart prove all original FP16/BF16 output words against the corresponding independent native target. Neither theorem assumes successful execution, a supplied output, a dot/reduction result, or a complete lowering equation.

The two state-free `Specification` files contain independent direct-address typed recurrences: every operand, tail mask, cast, reduction, online update and final division is explicit. `Binding` proves their equality to the original source arithmetic. Target audits exclude source `answer`, `rounds`, `advance`, source execution and the grid runner. The two existing quotations in `Kernels.lean` correspond to the two actual Python JIT functions; no alternate Python implementation was introduced.

All views consult the caller's physical reader at original addresses. `RawFrame` covers every tagged region/offset outside logical output stores. `observe` is mathematical word readback from those exact stores, with an exterior word frame at the caller's output pointer. Every read Tensor element is preserved. `backing_exterior` also covers complete read allocations, including prefixes, gaps and tails. Native readback uses the selected physical owner for each output feature; all exterior frames hold for every owner. Empty calls have empty work, identity source execution and unchanged caller words. No host, allocator, copyback or unified final byteheap implementation is added.

## Precision and explicit model boundaries

FP16 QK has explicit FP16 dot output and FP16 bias operations. BF16 QK has FP32 dot accumulation, BF16 narrowing, then the source's explicit BF16 +0 and bias operations. Both branches scale Q in FP32 before narrowing, cast probability to storage before PV, retain FP32 PV/online state and divide before the final output cast. The native graphs retain accumulator seeds, selected reduction trees, routing and compiler fusion.

Each native endpoint uses one instruction model with identity raw word leaves, scalar/conversion operations and individual coordinates of fixed seeded m16n8k16 instructions. All input words, including NaN/Inf, remain in its structural domain. The compiler plan contains integer routing metadata only; resource/driver/hardware applicability and compiler/source/ISA correspondence remain explicit external boundaries. The historical local-transport refinements are retained but are not numerical premises of the new endpoints. There are no deferred numerical equalities or experiments. The previously scoped shared launcher grid-product issue remains outside this task.

## Validation and evidence

`kernel-evidence/delivery.json` pins the current task build receipts, public audit, exact repository import closure and effective objects. Existing core and package caches are reused; this is a targeted task build. Public axiom footprints stay within the standard base, and the independent targets pass non-circularity audits.

The unchanged checker passes 1042 admissions and 43 rejections, including all 1024 dtype/configuration pairs and FP32 scalar-word controls. Eight complete FP16 Triton CPU interpreter grids cover query/key tails, independent column/row/head pitches, every output and full output-gap byte frames. The historical original-source output-head defect is reproduced and the selected source passes. Host metadata is mocked; these tests do not claim GPU execution. BF16 has formal and compiler evidence, without a CPU numerical execution claim.

Eight actual JIT binder profiles compile for SM80 and pass source/precision checks; their 32 TTIR/TTGIR/LLIR/PTX artifacts are saved. Their shared-memory sizes are 1568, 57344, 122880 and 18496 bytes per dtype. Compilation and being within the SM80 shared-memory limit do not establish runtime success or universal compiler conformance. No GPU execution was performed. The unchanged historical checker's printed word “deferred” is superseded by the current explicit model-boundary classification in `numerical-obligations.json`.
