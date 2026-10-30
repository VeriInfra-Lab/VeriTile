# Complete BF16-returning attention kernel

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [KernelOnly/Misc/Audit.lean](KernelOnly/Misc/Audit.lean), [Whole001/Misc/ReviewEntry.lean](Whole001/Misc/ReviewEntry.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

`FPEquiv.grid_equiv` now compares the complete selected source against the
fixed-BLOCK_M index strength-reduction candidate with `≡[[]]`. The ordinary
readable candidate in `Kernels.lean` changes only `start_m * 128` to
`start_m << 7` in the row-base computation. Its Python kernel and checked wrapper
are in `attention_fwd_triton2_shifted.py`. All original floating operations,
helper stages, mixed precision, tails, full 3D grid and output windows remain.
`Proofs/Pair.lean` proves complete state and safety preservation, then joins the
existing successful-run/output/frame theorem. There is no numerical input
restriction or experimental rule. This is an address strength reduction; no
measured speedup is claimed. `check_pair_source.py` checks the exact source delta.

The historical reciprocal candidate has different source storage and repair
details and remains separate. Its old proof is not used for this new pair.
The former single-source results retain their original names, complete
statements and proofs as ordinary theorems in `Misc/NativeCorrect.lean`.

`Correct.kernel_raw_correct` exposes shared `ProgramIO.Implements` for the
complete grid and the independent normalized attention formula.
The optional `FPEquiv.kernel_raw_native` theorem in `Misc/NativeCorrect.lean`
exposes `ProgramIO.Observes` for the original
instruction model's final BF16 words. [IO.lean](IO.lean) displays the five finite
FP32 input storage channels, BF16 output shape/strides, permitted input aliases,
absence of private scratch, source identity and actual launch metadata.
[Misc/TensorBridge.lean](Misc/TensorBridge.lean) retains the complete Tensor applications
`kernel_real`, `kernel_native`, `empty_kernel`, checked Real/native transport,
and the former arbitrary-state raw results as `kernel_raw_correct` and
`kernel_raw_native`.

The selected source is `Whole001/attention_fwd_triton2_repaired.py`, SHA-256 `eb7192d0aaab5db92b98eba026b05eb728774263b5f3afecbd7270d9d84bc005`. This delivery leaves its Python guards, outer kernel, helper and complete readable `Whole001/Kernels.lean` quotation unchanged. The original 45 benchmark/repair files are retained under `kernel-evidence/preserved` with their manifest.

The independent Real specification is noncausal base-2 normalized attention. For one batch/head, `score[n,j] = (sum[d<96] Q[n,d]*K[j,d]) * QS[n//128] * KS[j//64]`, and `out[n,d] = sum[j<N] 2^score[n,j]*V[j,d] / sum[j<N] 2^score[n,j]`. There is no log2(e) multiplier. For a live row, the first block has positive mass and its rescaling factor is zero, eliminating the source's initial denominator value 1.

The caller accepts ordinary resolved strided CUDA FP32 Q/K/V of shape `[B,H,N,96]` and FP32 packed scales `[B,H,ceil(N/128)]` and `[B,H,ceil(N/64)]` on one device. Nonempty Q/K/V have exact strides `(H*N*96,N*96,96,1)`; scales have their exact packed strides. Read-only aliases and nonzero storage offsets are allowed. The fresh return tensor is BF16. Empty B/H/N still allocates the output and unused FP32 M, bypasses stride checks as the source does, and returns without launching. The original checked integer/address envelope and grid.y bound remain in force; the shared grid-product compatibility question stays outside this task.

`Contract.lean` exports `KernelContract.accepted_iff` over actual Tensor observations. The original checked admission proof is preserved byte-for-byte in `KernelOnly/Admission.lean`. `TensorLaws` supplies ordinary indexing, capacity, data-pointer and fresh-allocation facts, without execution or result assumptions. `actual_address` and `scale_read` connect all six original views, including three-dimensional packed scales, to the shared physical input reader.

The Real judgment and optional native judgment cover successful execution of every program in
any complete grid permutation, every output and the complete tagged exterior
frame. Source trace safety is proved for the actual prelude, reached key-block
loops and BF16 stores in [Proofs/SourceSafety.lean](Proofs/SourceSafety.lean) and
[Proofs/CompleteSafety.lean](Proofs/CompleteSafety.lean), then transported to
those same launches in [Proofs/IOSafety.lean](Proofs/IOSafety.lean).
[Proofs/FiniteReference.lean](Proofs/FiniteReference.lean) connects the independent
recurrence using only the finite actual input ports, leaving arbitrary other
memory and legal read aliases admitted. The original raw tagged outputs,
access/formed-address closure and physical Tensor consequences remain in the
bridge and reuse the same source proof. `backing_exterior` covers entire original input allocations, including prefixes and tails. `TensorBridge.empty_kernel` proves empty work, identity source execution and unchanged observations.

The native recurrence retains FP32 loads and default TF32 QK, ordered query/key-scale multiplication, FP16 P and V, FP16 PV accumulation/result, widening to FP32, FP32 online updates and final division, and BF16 narrowing. It retains the four physical owner replicas, ordered local reductions and xor-2/xor-1 exchanges, 16 seeded K8 TF32 steps and four seeded K16 half steps. The 13 instruction families are scalar operations, conversions and individual coordinates of fixed MMA instructions; raw input words are identity leaves. No whole-result correspondence is assumed, and no native-to-Real accuracy equality is claimed. Historical conditional transport theorems remain available but are not premises of the new endpoint. No numerical experiment or numerical equality is deferred.

Validation uses 38 current task-module build receipts, public/audit import-closure hashes, standard-base axiom and non-circularity audits, and the current shared Algebra cache probes. The unchanged host checker passes 41 controls. The actual wrapper/default JIT binder compiles five SM80 profiles: `(B,H,N)=(1,1,1),(1,1,64),(1,1,65),(1,1,129),(2,3,129)`; three empty profiles skip launch. Twenty TTIR/TTGIR/LLVM/PTX artifacts are retained. The whole selected source also passes four nonempty CPU-interpreter controls and three empty controls on real Tensor backing, including nonzero offsets, read aliases, exact BF16 output 2 and full input/unused-M frames.

These are finite source/compiler controls, not a GPU execution or universal compiler-conformance theorem. The saved profiles require 98304 or 131076 shared bytes per CTA; resource availability and runtime execution are external conditions. Tensor API truthfulness, stable caller memory and the selected compiler/source/ISA interpretation are model boundaries. The task does not implement the host allocator/copyback or a unified final byteheap. The read-only evidence verifier is `kernel-evidence/verify.py`; delivery is frozen in `kernel-evidence/delivery.json` for independent review.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `eedab60247bb8128eb6bfe357474f1f18d8a3ac9a55a92e6d8a6769c739af322`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(q.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.

The Real specification and optional native theorem contain source geometry, scalar pointer bases,
the original finite-address/stride domain and any full-grid permutation.
Logical inputs, successful execution, actual accesses, outputs and the exact
frame are connected by the shared IO judgments; no raw state or execution
predicate appears in either IO statement. `TensorBridge` imports supporting raw
compatibility proofs without depending on either public reader entry. Both
layers reuse `Whole.run_correct`, `Machine.storage` and the original independent
Real/native interpretation proofs. The formatting change adds no numerical
assumption and reruns no CPU/GPU experiment or independent replay. Its targeted
build covers all new IO/safety/value helpers, both public entries, the original
raw compatibility results, TensorBridge and the existing KernelOnly audit.
