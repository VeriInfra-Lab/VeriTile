# INT8 quantization and GEMM: checked original tensors

**Current canonical revision: source defect pending repair.** Ordinary Triton
JIT specialization turns the wrappers' Python unit strides into constexpr
integers, but both JITs call `.to(tl.int64)` on them. Four admitted small inputs
produce eight confirmed compiler failures; both original-source controls compile.
See [SOURCE_DEFECT.md](SOURCE_DEFECT.md) for the concrete cases and replay command.
The canonical Python and input contract are unchanged. This revision is not
accepted as a completed kernel correctness proof, with or without numerical
deferral.

The description below records the historical wrapper-model work and its declared
boundaries. It does not establish successful execution of the current Python
revision. New `Proofs/Kernel*.lean` results preserve useful quotation-execution
and scalar-model work; their remaining end-to-end connections are listed in the
defect report. Basic arithmetic, conversion, saturation and encoding semantics
are proof/model obligations, not numerical-experiment TODOs; the historical
numerical-deferral classification below is not used by this delivery.

This package preserves all four production APIs: `quantize_int8_perrow(fpa)`,
`matmul_int8(a,a_scale,b,b_scale,out=None)`,
`matmul_quantize_int8(fpa,b,b_scale,out=None)`, and
`quantize_int8(weight,axis=0)`. Original benchmark files are unchanged. The two
complete repaired JIT kernels retain the historical source operation order,
all four producer configurations and all 34 GEMM configurations, including
split counts 1/2 and group sizes 8/16.

The producer and Torch helper accept FP32 input matrices. The producer returns
an INT8 matrix and explicitly FP16 row scales. Standalone GEMM takes INT8 A/B,
FP16 or FP32 row scales, and FP32 column scales. Fresh GEMM output is explicitly
FP16; supplied output may be FP16 or FP32 and is returned as the exact supplied
object. Each partial preserves INT32 accumulation, INT32-to-FP32 conversion,
FP32 multiplication by row then column scale, and mandatory FP16 narrowing.
FP32 destinations widen that narrowed partial for stores or FP32 atomic adds;
they do not bypass the FP16 transition. The Torch helper returns INT8 values,
FP32 scales, and literal `None`.

The wrappers check strided tensors, rank, dtype, CUDA ownership, shape, no
unresolved negative view, and nonempty pointer alignment before allocation or
launch. Inputs are contiguous row-major, except B may also be
transpose-contiguous column-major. Nonzero storage offsets and arbitrary inert
singleton/empty strides are supported. Canonical launch strides are proved to
address the original tensor views; inputs are passed directly. Same-dtype
read-only aliases are admitted. Mixed-dtype raw-storage aliases and any supplied
output sharing input storage are rejected, matching the selected one-width-per-
storage raw memory model. No finite-value or nonzero-scale guard is imposed.

For a nonempty producer grid, `0 < K <= 2**20`, `M <= 2**31-1`, and
`4*(input_offset + M*max(K,1) + 2*next_power_of_two(K)) < 2**63`.
M=0 bypasses these unused bounds. GEMM checks every actual autotune
configuration: signed64 dimensions, padded coordinates, grouped-grid arithmetic,
all formed input/output byte rectangles, and the final unused K pointer
advance. The first grid dimension is at most `2**31-1`; the second is at most
65535. Destination byte scaling uses its actual 2- or 4-byte element width.
Empty M/N bypasses unused launch bounds and returns the correct empty descriptor;
standalone K=0 is supported. `Contract.lean` records the exact checks, and
`Proofs/Contract.lean` proves successful dynamic admission iff that contract and
derives all native index/grid bounds from it.

The Torch helper accepts integer axes -2, -1, 0, 1, excluding bool, and rejects
an empty reduced dimension. Only literal axis 0 performs the source's intrinsic
`t().contiguous().t()` operation. Nondegenerate axis-0 results have column-major
strides `[1,rows]`; singleton/empty transposes take Torch's actual no-copy branch
and return the exact temporary object with its actual inert strides. This
intrinsic transport is part of the helper API, not input normalization.

The four `Specification.*_wrapper_correct` entries in `Correct.lean` cover the
complete checked wrappers, all output values/scales, returned objects and
metadata, physical readback through the returned views, current-device
restoration, and caller storage frame. Independent targets read the original
TensorView inputs. GEMM's mathematical target is the signed INT8 inner product
multiplied by its original row and column scales. Quantizers use maximum absolute
value divided by 127 and toward-zero INT8 conversion; the composed target uses
those produced INT8 values and scales. These are arbitrary-real interpretations
of the same typed structure, with total real division, not encodings of arbitrary
reals into FP words or claims of universal FP-to-Real equality.

Composition uses the producer's actual output state and exact allocated handles.
`produced_admission` derives the repeated GEMM admission from the original
composed checks and ordinary allocation metadata, alignment and freshness.
`producedRequest_original` proves those are the actual producer objects passed
to GEMM. The producer-to-GEMM storage/precision link is derived; no assumed
intermediate or final values are supplied. Autotune trials, output zero hooks,
selected final split schedules, and caller frame are included.

The separate four `FloatingSpecification.*_native_if_local` entries cover the
compiler-bound native execution and every returned output byte. They consume
original input view/storage laws and ordinary output allocation Tensor laws,
derive data-pointer identities for all returned tensors and both composed
producer intermediates, then use
independent ordered word targets on those original inputs. Targets retain the
selected INT32 dot plan and atomic order, every FP32/FP16 transition, and the
producer reduction tree. Torch helper scalar operations have their own profile;
Torch amax is not silently assigned Triton maximum's NaN behavior.
`Kernels.lean` contains only the complete production AST binding and its source
import. Correctness and floating readers import their own proof dependencies.

Ordinary Tensor metadata/layout/pointer/capacity semantics, successful fresh
allocation, local typed/raw load/store bindings, source/frontend/compiler
correspondence, and successful race-free execution are implementation
boundaries. Allocation layout fields contain physical index relations, never
assumed numerical results. Structural source/grid/address/readback obligations
are discharged for these boundaries. Only native primitive numerical conformance
remains deferred: selected INT32 MMA/WGMMA wrapping/saturation and conversion,
FP32 operations/FTZ/fusion, FP16 narrowing/widening/addition/atomics, reduction
maximum/NaN behavior, division by 127, division and toward-zero INT8 conversion.
Little-endian byte serialization is the structural `RawByteABI.decode` contract,
with concrete bit extraction in `cellByte`; it is not a deferred numerical law.
The numerical primitive premises do not state whole-output or whole-memory equality.

Historical repaired Python SHA256:
`bb3411459f3accd210a86a225c62d979375870933fbbcb88242e3ee44fa6f1c8`.
Review: `2026-10-18/int8-matmul-quantization-whole003-repair-integration/repair-review.json`,
SHA256 `b9e64b55b3b6d884ce631e97e62e19b2e6ca526452546ac1d9b30a6ad79bea73`.
These identify the historical accepted source; the direct wrapper changes do
not alter acceptance ledgers.

Validation: targeted Lean compilation, standard-base axiom audits, and
noncircular independent-target audits, with no `sorry` or new axiom.
`python bench/repairs/tritonbench_g/int8_matmul_quantization/check_contract.py`
passes 244 complete actual-AST API cases and 31 guard rejections. It checks all
34 GEMM configurations, B layouts, row scale types, fresh/supplied FP16/FP32
outputs, exact identities/metadata/strides/frame, offsets, permitted aliases,
empty/K=0/tail cases, trial/final reset behavior, composed admission and outputs,
all helper axes and copy/no-copy branches, nonfinite/overflow/zero producer
payloads, and native-bound failures. It also checks the complete production AST
and pinned JIT AST hashes. CUDA allocation and launch are mocked; these CPU
checks do not validate native ISA numerics.
