# Source and model binding

The canonical Python is `revision001/int_scaled_matmul_repaired.py`, SHA256
`17bd051b849369e11f3c2b57440052a8c304722b32985499300ab572c553a33b`.
The original benchmark and all `revision001` source/proof files are retained.
The assertions and source arithmetic are unchanged in the current revision.

## Source, precision and complete outputs

`revision001/Kernels.lean` owns the readable quotations for the two kernels.
The current `Kernels.lean` shares those definitions. The retained width
elaboration inserts signed8 interpretation after integer loads and signed32
projection at carry updates. It models the source's explicit PTX `add.u32`,
not a hypothetical identity between saturating carry-fed dot and modular add.
`SaturatingTile.source_dot_eq` and `IntegerInstructions.program_eq` prove the
zero-seeded instruction refinement under the exact checked tile bounds.

A/B storage is signed INT8. Each bounded tile dot is INT32, and `add.u32` updates
the INT32 carry modulo 2^32. The optional suffix promotes signed32 to FP32,
performs one FP32 multiplication, and stores through a signed32 toward-zero
conversion. C remains INT32. The mathematical reference sums all K original
signed8 products before the signed32 projection, independently of any tiling.
No FP32/BF16 matrix-product substitution or precision variant is introduced.

`KernelExecution.source` calls the complete shared `BlockWord.grid` or
`ScaledWord.grid`, whose program step is `FP.Structural.exec` on that quotation.
`source_correct` proves success and the independent result. The finite native
executor is connected through proved signed32 grouping, coordinates, masks,
terminal advances, zero-seeded MMA and modular additions. Source and native
success are conclusions. Complete output values plus the exterior frame prove
the two memories equal; no source execution or final-result equality is a
hypothesis.

`KernelNative.instructions` explicitly follows the observed PTX multiply order,
row scale then converted accumulator. Its all-word scalar proof reaches both
the entire instruction grid and physical byte grid retained in `TensorBridge`.
The public `FPEquiv` uses the shared `ProgramIO.Equivalent` relation with the
same fixed K32 primitive and scalar profile displayed by `NativeIO`.
`NativePrimitive` proves the instruction identities, and `NativeComplete`
connects these source schedules to the original finite signed-index executors
with exact complete-memory equality. No whole-dot numerical law is assumed.
The six retained current lowering profiles contain 72 such scaled multiplication
sites across their four scaled variants. They have no FTZ multiplication.

## Tensor and physical memory

The exact assertions use shared `VeriTile.Torch.TensorMetadata`. The retained
`TensorBinding` adapter adds INT8/INT32 byte widths and ordinary storage/view
facts to this shared metadata. Its laws contain no matrix-product result,
source-envelope bound or assumed output memory. `Inputs` binds both original
INT8 inputs and optional FP32 scales to one physical byte reader, so read-only
aliases are preserved. Contiguous coordinates agree with data pointer plus
logical byte offset, including nonzero storage offsets.

`ByteExecution.program` runs the same finite instruction loop and active masks,
and each `step` performs four little-endian byte stores alongside the logical
INT32 write. `render_write` and `sequence_render` prove this simulation through
all actual stores and grid programs. C's initial byte binding is explicit.
`ByteHost.post`, reused as a result predicate only, gives physical word, logical
Tensor and four-byte readback plus every exterior byte. The exported address
lemmas derive four-byte alignment and no UINT64 address wrap from Python's
existing guards. No allocator or wrapper implementation is proved here.

The contiguous Python contract permits irrelevant singleton strides. The
wrapper supplies canonical dense strides to the source, whose accessed logical
indices therefore have the proved addresses. This is not a claim that Torch
contiguity implies literal singleton strides.

## Primitive and compiler boundaries

The concrete source primitive profile is the shared `FP.Config.ieeeValue` and
`FP.mul`/`FP.round` definitions. It has nearest-even rounding, separate disabled
input/result flushing and canonical arithmetic NaNs. Defined signed conversion
uses exact rational decoding and the independently specified toward-zero
interval. Outside that defined conversion domain, the model retains an
explicit local 32-bit result. All caller FP32 words remain admitted.

`KernelModel.correspondence` proves the former generic scalar-law interface
from these operation definitions. It is not a measurement and imposes no
numerical-transformation premise. `KernelNative.multiply_comm` handles the
observed operand commutation within this exact profile. This does not establish
native NaN-payload behavior or replace the explicit undefined conversion port
with a guessed value.

The compiler evidence is scoped to the saved Triton 3.7.1 SM90 profiles,
with BK=32, four warps and two stages. The unscaled/scaled source has zero-seeded
tile dots and separate `add.u32`; MMA16 zero initialization and WGMMA64 scale-d=0
are inspected. The mathematical proof covers all ConfigContract tile choices;
ordinary successful compilation/resource/API binding is a boundary of applying
that kernel theorem to hardware. No universal compiler correctness or new
host/autotune/launch compatibility theorem is claimed.

Relevant API/ISA specifications:

- [Triton `dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html):
  input/accumulator types and the accumulator operand.
- [Triton `load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
  and [store](https://triton-lang.org/main/python-api/generated/triton.language.store.html):
  masks, block-pointer boundary checks, padding and pointee conversion.
- [Triton inline assembly](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html):
  explicit elementwise instruction and operand/result types.
- [PTX integer add](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-add):
  32-bit addition and saturation modifiers.
- [PTX MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
  and [WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma):
  signed INT8 products, INT32 accumulation, saturation and scale-d.
- [PTX floating multiply](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)
  and [conversion](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt):
  rounding/FTZ modifiers and float/integer conversion semantics.
- [PTX addresses](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands):
  byte addressing and natural access-size alignment.
- [Torch `data_ptr`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
  [storage offset](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html)
  and [contiguity](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html):
  the original Tensor view and its first logical element.

These are modeling references. The checked source/IR/PTX artifacts record the
specific application of the boundaries; no new GPU experiment or hardware
conformance result is inferred from a citation.
