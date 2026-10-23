# triton_linear_activation: selected kernel revision

The selected implementation is `triton_linear_activation_repaired.py`, SHA256
`271cfcf8614afc2747837dbf8902f3addc9a3a605a6c1a85063ae2b6c346f8a9`.
The original source, admission checks and 98 autotune configurations are retained.
The separate `triton_linear_activation_shifted.py` candidate changes only integer
group-width arithmetic: `GROUP_M * grid_n` becomes `grid_n << 3` when the
compile-time GROUP_M is 8, with the original formula as fallback otherwise.
This is address strength reduction; no measured performance gain is claimed.

`Kernels.kernel` exposes the complete readable typed DSL in
`SourceKernels.lean`. Its 26 non-docstring top-level statements retain
grouped program geometry; unwrapped store coordinates and wrapped loads;
zero-then-bias addition; the descending `range(K, 0, -BLOCK_K)` loop;
exact-multiple and masked K loads; the two-argument `tl.dot` followed by an
outer addition; both final pointer increments; the optional saved store;
source-order activation expressions; and the final masked output store.
The DSL spelling `-0` is the same stop value 0 and selects its signed descending
range expansion. Pointer store casts are made explicit. The frozen source
audit compares the entire normalized Python/DSL AST after helper inlining and
these documented notation changes.

## Public results

- `TensorBridge.correct`: successful ideal Real execution on actual prepared Tensor
  bytes, every active output and saved preactivation, and the complete untouched
  tagged-cell frame. The independent Tensor target is
  `bias[n] + sum(k<K, X[m,k] * W[n,k])`, followed by the exact named mathematical
  activation or identity fallback.
- `TensorBridge.fp_equiv`: successful execution of the source expression graph under
  an explicit compiler policy, every native output/saved word, actual Tensor
  destinations, and the same complete tagged-cell frame. Its independent Tensor
  expression reference preserves source order and constructs each dot from
  fixed K8 TF32 or K16 FP16 instruction lanes.
- `TensorBridge.all_outputs_covered`: every logical output belongs to an active lane
  of a valid program, for every one of the 98 original configurations.
- `Proofs/KernelBounds.formed_inputs` and the public `Formed` conclusion cover
  signed-32 formed offsets, including masked lanes and the unused final input
  pointer increment, under the existing source bounds.

These are kernel invocation theorems for any valid launched program. They
cover all of that program's stores; grouped coverage covers the whole logical
output domain. No new whole-grid interleaving, host engine, allocation,
autocast implementation, returned view, backward, or copyback theorem is claimed.
An empty M/N grid has no valid invocation. K=0 remains a proved zero-trip loop.

## Actual dtype and Tensor boundary

Admitted caller X, W, optional bias, and optional caller saved Tensor are
canonical FP32 CUDA tensors: contiguous, zero storage offset, exact canonical
strides even on singleton axes, no lazy negation/conjugation, naturally aligned
active pointers, and disjoint complete live backing intervals. Rank-3 X is
flattened to M=B*S for the kernel. Existing signed-32 guards are retained.

Without autocast, actual kernel pointers are FP32. The explicit
`custom_fwd(cast_inputs=torch.float16)` branch prepares FP16 X/W/bias even when
ambient CUDA autocast selects BF16. The actual saved pointer is also FP16
scratch in that branch. The existing later copyback to the caller's FP32 saved
Tensor is outside this kernel endpoint.

The FP32 branch uses shared `VeriTile.Torch.FP32.Tensor` and `TensorLaws`.
The FP16 branch uses shared `TensorMetadata` with an explicit task-local
2-byte storage/API binding, because the shared FP32 API cannot represent half
storage. Public input words are read from actual little-endian bytes: four
bytes for FP32, two bytes in the low 16 bits of the native carrier for FP16.
Inactive optional roles are mathematical placeholders: absent bias uses the
actual X pointer, and absent saved-buffer strides are exactly zero. No extra
Tensor allocation or copy is introduced. Shape, physicalIndex, capacity, actual pointers, device, and separation are
ordinary Tensor/preparation facts; none assumes an output or execution result.

## Precision and scope

The source graph and its native evaluator are separate. `sourceOrder` evaluates
two-argument dot then addition; `contracted` allows dot/add and multiply/add
contraction. The theorem holds under either explicit policy. It never asserts
that the two policies produce the same floating result. FP32 add/mul/FMA,
maximum, half pack/unpack, and literal words are concrete. Division, libdevice
tanh/erf, and one named fixed instruction lane are explicit machine-semantics
ports. There is no arbitrary-K whole-dot or completed-output equivalence premise.
See `KernelValueScope.md` and `kernel-numerical-obligations.json`.

The Real interpretation applies an input-only scalar projection to prepared
bytes; exceptional native values project to zero there. It is an ideal
mathematical interpretation, with no claim of numerical equality or error bound
against native words. No finite-input or numerical guard was added.

Frames concern all modeled tagged allocation cells, including suffix cells
and other regions. Post-store byte-memory alias coherence and GPU/compiler
implementation conformance are not claimed.

## Evidence and original defects

Existing checks passed: 319 actual CPU source/reference/frame cases, seven
additional rejected inputs, and eight ordinary failure prefixes;
73 preparation/AMP cases, ten rejected inputs, 588 grouped geometry cases,
and 588 independent dense cases. These include every configuration,
K=0 and ragged K, both compute modes, activation fallbacks, rank-2/rank-3 input,
empty grids, saved backing suffixes, noncanonical singleton strides, offsets,
and allocation overlap. These are CPU/FakeTensor controls, not GPU validation.

Original-source defects remain classified separately:

- M=3, N=5, BM=16, BN=32: wrapped lanes (0,0), (0,5), and (3,0) all store offset
  0 with the original wrapped mask. This is duplicate store ownership/race;
  it is not reported as a CPU arithmetic mismatch.
- K=1, BK=32: the original unmasked lane 31 addresses K=31 outside logical K.
  The address witness executes no unsafe load.
- Original autocast saved-buffer witness: caller FP32 starts at 7, the hidden
  half buffer receives 0, but caller remains 7. The existing repair copyback
  yields 0. This is a host output defect, outside the new kernel theorem.

The frozen packet is
`.veritile-cache/task-deliveries/triton_linear_activation/delivery.json`.
It records current source/object receipts, source AST evidence, test logs,
public audits, dependency pins, and explicit boundaries.

Historical host proofs are preserved as `Misc/HostCorrect.lean` and
`Misc/HostFPEquiv.lean`; `README.host.md`, `FPValueScope.md`, and
`numerical-obligations.json` describe that historical selected compiler/host
model. Its fused accumulator, reordered bias, and selected activation graph
are not silently reused as source-faithful floating execution.

`Misc/TensorBridge.lean` is the optional actual-pointer interface, exposing `TensorBridge.actual_pointer` and `TensorBridge.actual_address`. The complete Tensor applications are retained in this bridge; neither primary imports it. Supporting pointer proofs remain in `Proofs/KernelPointer.lean`.

`Correct.kernel_correct` now uses the shared `ProgramIO.Implements` judgment
with a separate logical dense formula. The optional `FPEquiv.kernel_fp_equiv`
theorem in `Misc/NativeCorrect.lean` uses the same
judgment for the existing source/compiler-model native observation, with the
compiler policy and fixed ISA/libdevice ports explicit. It does not assert a
new optimization pair or equality between contraction policies.

`IO.lean` exposes the actual source invocation, FP32/FP16 channel types, wrapped
operand windows, unwrapped output addresses, masks, optional bias/saved roles,
and empty private scratch. Inputs are logical `[blockRows,inner]` and
`[blockColumns,inner]` operand tiles plus a column bias vector; both outputs
are `[blockRows,blockColumns]`. Source-compatible aliases remain admitted;
canonical strides and saved/output separation are unchanged public premises.
The program ID is explicit. Zero rows/columns and arbitrary program IDs remain
in the raw kernel scope: backing includes wrapped source reads even when no
output is active. K=0 has no operand reads.

`Proofs/SourceSafety.lean` proves access safety of the actual 26-statement
source through the descending K loop, using the existing execution states.
`Proofs/IOBindings.lean` connects those accesses to the public windows.
`Proofs/RealInterface.lean` and `Proofs/NativeInterface.lean` reuse existing
successful source execution and complete output/frame proofs. The original raw
results remain in `KernelReal.outputs` and `KernelNative.outputs`; prepared
Tensor bytes and physical readback remain in the unchanged full
`TensorBridge.correct` / `TensorBridge.fp_equiv` applications, built on those
same execution foundations.


Public FP entry, 2026-10-21: `FPEquiv.kernel_equiv` proves the actual complete
`original ≡[[]] shifted` program pair. Both readable sources are outside
`Proofs/`; `Proofs/KernelSource.lean` retains only decomposition helpers under
the original full names. `Proofs/Pair.lean` proves exact changed-statement run
and safety equality for arbitrary algebra/state, then lifts it through the whole
unchanged source suffix. Both programs execute successfully, agree on every
active output and saved output, and retain the complete exterior frame.

The public scope remains one full invocation at arbitrary program ID, under
exactly the original positive blockInner, canonical strides and saved/output
separation premises. Both modes, bias/save flags, activation branches and the
complete descending K loop remain present. GROUP_M is not narrowed to 8; the
fallback preserves generic parameters. There is no finite-input guard,
experimental numerical rule or pending numerical experiment for this pair.
`check_pair_source.py` restores the one guarded shift and candidate names and
checks exact full-module AST equality, preserving all 98 configurations and the
host flow. The optional native and Tensor results retain their prior scopes.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/Reader.lean`, `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。
