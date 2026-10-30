# matmul_dequant_int4 repair

This private candidate preserves both original Triton kernels, all 17 fused and 8 standalone configuration occurrences, and the original host/helper arithmetic. The accepted repair initializes the packed quantizer weight accumulator with explicit int32 zeros. The host contract rejects unsupported metadata and unsafe formed addresses. The source's `torch.ones` dequantization temporary, `torch.mm`, quantizer transpose/contiguous/view/device operations, and exact supplied-output identity remain intact. No normalization copy or snapshot was added.

## Checked contract

The production s1 API explicitly uses FP16 A, scales, temporary and output. This precision comes from the source. The inherited standalone dequantizer and quantizer also support their existing FP32 profiles; this repair adds no dtype profiles.

The packed weight is int32 `[K/8,N]`, the scale is `[K/group,N]`, and packed zeros are int32 `[K/group,N/8]`. K and N are positive multiples of eight; a positive integer group divides K. M is positive for s1. Quantization additionally checks finite original weights, finite strictly positive computed scales, and integer `tp_rank` equal to the input CUDA device.

The checked host tensors are exact rank-two, strided, contiguous CUDA tensors, with offset zero, no lazy flags or autograd, adequate storage, natural alignment, and valid 64-bit storage/pointer bounds. Inputs and supplied output occupy disjoint whole storage intervals on one device. For `[rows,columns]`, row stride is `columns`, or `1` for a singleton row; column stride is `1`, or `rows` for a singleton column. These are the inherited source singleton layouts. Formed masked addresses retain actual strides. The quantizer returns the source's transpose/contiguous layout, including `(1,rows)` when either dimension is one.

`Contract.lean` mirrors the primitive metadata checks and computed-content guards. Signed32 bounds include inactive standalone lanes. Output allocations retain ordinary successful allocator guarantees and the Python candidate checks their pointer domain. Supplied output is returned exactly; the absent-output path returns the original fresh allocation. Calls restore the caller CUDA device, including context-manager unwinding.

The unused `matmul4_kernel` has a separate raw ABI: canonical strides, FP16 inputs/output, FP32 accumulation, K and group multiples of 128, N a multiple of 256, and `NO_GROUPS` exactly when group equals K. `Direct.RawIndexGuard` bounds padded M, the final K pointer increment, every formed index, and the grid. `checkRaw` is a Lean declarative admission predicate for this raw ABI. No new Python raw-kernel wrapper was invented.

## Public proofs

The public endpoints are:

- `Correct.lean`: `Direct.matmul_dequant_int4_real`.
- `FPEquiv.lean`: `Direct.matmul_dequant_int4_native` and `Direct.matmul_dequant_int4_if_local`.
- `Reader.lean`: whole-package axiom audits and structural witnesses.

Neither public proof file imports the other, directly or transitively. The historical results remain available in `Proofs/WholeReal.lean` and `Proofs/WholeNative.lean`.

The Real theorem quantifies arbitrary `State ℝ` payloads independently of any byte decoder or representability restriction. The source-specific recurrence is related to independent base-16 digit extraction, complete dense GEMM, and quantization by extrema including zero, division by 15, nearest-even rounding/clamping and base-16 packing. Returned tensor descriptors, layouts, devices and caller frames are included. Standalone and s1 retain the original ones initialization and evolving autotune trials. Raw fused trials may use different configurations and compiler-selected fusion/ISA metadata.

The native theorem starts from one original absolute-address byte heap. Views bind original allocation bases without moving or normalizing inputs. Proved little-endian codecs load FP16/FP32/int32 values; source-specific recurrences run on evolving memory; only the actual output spans are serialized back. The result describes every output byte, exact returned tensor identities, all original input-storage bytes, and all caller bytes outside returned/scratch spans. Quantization covers all three returned allocations. Schedule coverage, ownership, active-stride equivalence, frames and machine bounds are derived.

`Proofs/Binding.lean` consumes explicit installed-library and compiler selections. The s1 plan selector receives the actual A, scratch and selected output strides and device. Raw compiler selection receives shape, `NO_GROUPS` and configuration and returns an ISA family and lane fusion map; it receives no payload or completed output. Finite Torch matrix plans contain scalar product/FMA/add nodes, fixed K16 instructions and optional half-truncation nodes, with complete logical K coverage. Reduction-tree metadata similarly describes finite scalar max/min trees covering each group element.

`matmul_dequant_int4_if_local` requires exactly the inherited 27 numerical family containers: 26 scalar families and one fixed K16 ISA component family. Scalar `/15` and Tensor/Tensor division remain distinct tags for each width. There is no variable-K dot, completed-output, source-execution, frame, schedule-coverage or helper-result equality assumption.

## Scope of validation

All 39 Lean modules compile, including the restored historical closure and new direct API readers. The Python harness passes 66 dequant/s1 admissions, 20 quantizer cases, 34 fused configuration/branch cases and 65 rejections, plus library-failure context restoration. It checks unchanged full kernel ASTs, all configurations, singleton/tail layouts, output identities, helper operations, complete outputs and byte frames.

The harness executes retained arithmetic ASTs and helper code on CPU backing storage; CUDA allocation, context and launch are mocked. It does not establish GPU numerical conformance. No GPU execution, compiler artifact generation, physical-lane-map extraction, or installed `torch.mm` algorithm observation was performed in this repair. The installed library/compiler-to-plan mappings, ordinary Torch allocation/copy/view/slice semantics, Tensor ABI and completed stream ordering remain explicit external implementation boundaries. Scalar/fixed-ISA conformance is deferred to the listed local numerical obligations. Real correctness is an interpretation theorem, not FP-to-Real equality or an error bound.

`SOURCE_BINDING.md` records the source/model correspondence and AST pins. No original task source or acceptance row, shared core, Lake registration, ledger or Git history was changed.
