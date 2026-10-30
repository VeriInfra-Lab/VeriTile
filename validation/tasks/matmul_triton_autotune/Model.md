# Model sources and boundaries

The canonical source and native binder probe are pinned in
`.veritile-cache/task-deliveries/matmul_triton_autotune/entry-source.json` and
`native-probe.json`. The latter records exact source/argument specialization,
configuration, compiler options and hashes of TTIR, TTGIR, LLVM and target code.
Installed versions are Triton 3.7.1 and Torch 2.13.0+cu130. The probe executes the
actual assertion/wrapper AST with mocked CUDA ownership/allocation/submission,
uses `native_specialize_impl` and `JITFunction._pack_args` on actual Python
scalar values, and compiles the unchanged kernel JIT body/signature offline.
Only the host autotune decorator is omitted to select each real configuration.
No GPU execution, full launcher replay or arbitrary-hardware coverage is claimed.

## Memory and Tensor meaning

Shared Tensor storage/stride/pointer definitions come from
`VeriTile/Torch/FP32.lean`. Generic Matrix/FP16 output bindings and their byte
writeback lemmas are reused read-only from the accepted
`matmul_leakyrelu/KernelOnly/{Tensor,Memory,FPEquiv}.lean`; its source, guards and
SM90 floating graph are not imported as this task's specification.

The model follows the shared memory rationale in
[`documents/GpuMemoryModel.md`](../../../documents/GpuMemoryModel.md).
[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
and [store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
define masked accesses and conversion to the pointer element type. Installed
`triton/language/core.py` defines load at line 2146 and store at line 2221;
`triton/language/semantic.py` implements pointer element typing/casts.
[NVIDIA PTX addresses](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses)
require naturally aligned addresses for the access size. FP32 input alignment
and complete byte intervals are retained, not replaced by element-only bounds.
The repair also forms valid inactive pointers by clamping coordinates; its
signed64 and byte-address proofs are reused, not assumed by a final output gate.

The ordinary library boundary supplies correct rank/stride classification,
successful allocations, storage indexing and pointer observations. Nat strides
represent the nonnegative Python branch; actual FP32 Tensor values are connected
to physical storage by `logical_fp32` after the checked negative bit is false.
The proof does not implement Torch, its allocator or the launcher. Both input
views use one heap, permitting arbitrary read-only input aliasing. FP16 output
writeback reads the executed state and preserves the entire exterior heap.

## Selected numerical interpretation

[Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
accepts an incoming accumulator and an input-precision profile. Installed
`triton/language/core.py:2027` and `semantic.py` retain that accumulator.
Pinned TTIR has `inputPrecision = tf32` for both sampled targets.

The CUDA SM80 profiles emit
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` with live accumulator input.
They contain no explicit `cvt.rna.tf32`; original FP32 register words feed the
instruction. See [PTX MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
for operand format, accumulation and implementation-dependent internal precision.
`Instructions.cudaMma8` is its per-output-lane, fixed K8 arithmetic boundary.

The sampled HIP gfx942 profile emits `v_mfma_f32_32x32x4_xf32` with live carry.
See the [AMD CDNA3 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf).
`Instructions.hipMfma4` is its per-output-lane, fixed K4 arithmetic boundary.
It is not labeled a full-precision IEEE FP32 dot. The graph folds these fixed
atoms across each source BK, including all zero-padded lanes. `block_exact` and
`instruction_lane_bound` prove the selected BK values cover complete atoms.

These two fixed instruction arithmetic parameters include their TF32/XF32
operand interpretation, internal reduction precision, rounding and special
values. They are not input guards or empirical numerical assumptions. Native
opcode/lane mapping, K grouping and hardware compliance with this interpretation
remain explicit model boundaries. The offline samples establish which graph
was observed, not a compiler correctness theorem for every specialization or
architecture. The public theorem proves the quoted source under the selected
model for all admitted shapes/configurations; it does not silently restrict the
Python device domain to the sampled architectures.

The scalar layer is concrete: ordered comparisons are `FP.lt || FP.eq`, slope
is FP32 `0x3c23d70a` and occurs on the left of `FP.mul .ieeeValue`, and the final
FP16 narrowing is `QuantizeKvTransform.MixedBits.narrow .ieeeValue`, using the
actual binary16 rounding/packing definition. It uses RNE, gradual underflow and
canonical NaN conversion. Native instruction payload/subnormal correspondence
to this named profile is part of the boundary above. Basic scalar arithmetic,
encoding and casts are definitions/proofs, not postponed experiments.

The independent Real theorem uses the ordinary sum and exact slope `1/100`,
with identity casts. No native-to-Real equality, cross-configuration equality,
reassociation, alternate optimized kernel or additional numerical assumption is
claimed. There is therefore no new numerical experiment TODO for this revision.
Historical numerical-port entries remain historical evidence only.
