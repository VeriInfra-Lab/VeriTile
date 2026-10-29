# rotary_transform: checked original preparation

The current kernel review found an admitted source-domain defect. The selected
source admits nonempty interleaved rotary width 262146 and split rotary width
524290, but its first matrix pointer has 2097152 elements in both cases.
Triton 3.7.1 rejects this shape before generating a kernel: its frontend limit
is 1048576 elements. Both cases reproduce for SM80 and SM90 through the actual
JIT binder; width-514 controls compile in both branches with default settings.

See [SOURCE_DEFECT.md](SOURCE_DEFECT.md), the replay in
[source-defect-replay.md](source-defect-replay.md), and the checked witness in
[Proofs/BlockSizeDefect.lean](Proofs/BlockSizeDefect.lean). The Python source,
guards and existing public theorems are unchanged. This finding prevents a
complete source-success claim over the current admitted domain. The remaining
description records the retained historical interpretation proofs; those proofs
do not establish successful compilation for these admitted shapes.

The separate repair removes the original accidental scalar-offset increment
and lets BLOCK_K cover every rotary column using next_power_of_2(max(32,R)).
The complete original JIT AST is unchanged. The original split BLOCK_M=8
through R=64, otherwise 4, and interleaved BLOCK_M=4 remain.
Positive-stride coefficient tables and tensor offsets keep their intentional
contiguous preparation. No new head-width limit of 256 is imposed. Scalar and
tensor offsets are nonnegative; exhausted or empty coefficient tables use the
original fallback. No input clone, packing, broad dtype/device conversion,
cache snapshot or CU materialization is added.

Both scalar/tensor offsets, fixed/packed sequences, split/interleaved pairs,
conjugation and inplace/out-of-place branches remain. Tensor metadata preserves
its original signed int32/int64 dtype. CU is a unit-stride vector; its original
object is passed to the kernel, with no added CU contiguous copy. CU begins at
zero, ends at the packed token count, is nondecreasing, and every segment fits
max_seqlen. Empty packed sequences are supported. The original coefficient and
tensor-offset contiguous calls, default empty_like allocation, nonrotary suffix
copy, CUDA context and returned Tensor selection remain.

The payload contract is FP32 strided CUDA tensors on one device, positive heads
and width, zero storage offsets, absent lazy negative/conjugate flags, and a
canonical contiguous input X. R = 2*cos.shape[1] is at most the head width.
For nonempty inputs the wrapper checks complete backing-range separation,
actual scalar alignment (four bytes for FP32/int32, eight for int64), physical
containment and representable allocation endpoints. Metadata validation reads
values to CPU and synchronizes before any kernel write. Signed32 source
intermediates and padded expressions are bounded. After the original
preparation, formed padded input/output/coefficient pointers are checked against
unsigned 64-bit overflow. Empty payloads and zero rotary width return after the
original suffix copy, with no launch. Coefficient fallback (cos=1, sin=0)
remains arithmetic; no raw-word identity is claimed for exceptional values.

`Contract.lean` records Tensor observations and finite guards over metadata
read from initial bytes. `Proofs/Guards.lean` derives Config.Valid and IndexABI
from those checks, including global monotonicity from adjacent CU checks.
`Kernels.lean` contains only source/configuration definitions. The complete
historical vector, physical memory, metadata-reload, in-place, coverage and
arbitrary complete work-order proofs are retained under `Proofs`. Helper public
closures were moved there without losing their statements.

`Proofs/Binding.lean` connects these source results to actual caller addresses,
the actual output Tensor and every output byte. `every_token` proves that the
batch/row theorem covers every returned packed token, including when CU has
empty intervals. The physical frame covers all bytes outside the output's
logical extent, including CU, offsets, coefficient backing/padding and input
storage for out-of-place calls. Inplace returns the exact original X object;
otherwise it returns the successful ordinary empty_like allocation, whose
shape/device/dtype and freshness are explicit ordinary Tensor API laws.
The memory state in this binding is the state after that ordinary allocation
and before the suffix copy/kernel; ordinary allocation preserves existing
caller storage. Its source interpretation includes the suffix copy and all
kernel stores, without assuming any completed output or launch theorem.

The historical `normalizeFor` CU region is only a runtime view of the same
caller bytes under checked stride one (`normalize_stride_one`); it does not
represent an added Python copy. For the existing coefficient/offset contiguous
calls, staging represents their ordinary input-only copy/view semantics.
Actual prepared coefficient pointers are observations in `Binding.Output`;
`pointerCheck` derives their formed-pointer bounds. No whole result,
output coverage, index, ownership or source-success obligation is deferred.

The remaining boundaries are ordinary Tensor allocation/copy/view semantics,
local byte memory/codec laws, normal race-free CUDA launch/device behavior,
and source-to-interpretation/compiler realization. The source-specific Lean
interpretation and actual AST replay are not a verified Python/Triton compiler.
Default fusion is retained. The selected scalar graph is
`fma(cos,a,neg(mul(signedSin,b)))` for low output;
`fma(signedSin,a,mul(cos,b))` for split high output; and
`fma(cos,b,mul(signedSin,a))` for interleaved high output. Conjugation uses
`sub.f32(+0,sin)`. Only raw-word scalar mul, neg, sub0 and fma numerical/compiler
correspondences remain conditional. All FP32 bit patterns are admitted;
GPU IEEE conformance and FP-to-Real accuracy are not claimed.

The short public entries are `rotary_transform_correct` and `rotary_transform_fp32` in namespace
`VeriTile.Bench.Repairs.RotaryTransform`.
The full closure and public entries pass targeted Lean 4.29 builds and
standard-base axiom audits, with no new axiom or sorry. Run
`python bench/repairs/tritonbench_g/rotary_transform/check_contract.py`: 78 complete
actual-JIT/host CPU storage cases cover all 32 flag combinations with both
integer precisions, tails, empty launches and fallback, plus 16 prewrite
rejections. The original scalar-shift defect is reproduced.
CUDA launch is mocked; these checks do not validate GPU arithmetic.

The additional public theorem `rotary_transform_correct_real` quantifies an independent arbitrary `Memory ℝ` for payloads and coefficients. Integer metadata is still decoded from the checked caller bytes. It covers the complete original wrapper result, all feature columns (including the copied suffix), every option branch, and the exterior frame. The earlier decoded-word theorem remains available under its existing name. Helpers are in `Proofs/RealWrapper.lean`.
