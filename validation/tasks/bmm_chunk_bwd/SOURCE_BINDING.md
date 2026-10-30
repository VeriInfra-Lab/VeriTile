# Source and model boundaries

The authoritative Python source is unchanged (SHA256
`91e5b8d4d3d423bb1a00fadb347273c5d90d2a3afd5d2b05b51f98eadd90c737`).
Its complete decorated arithmetic AST has SHA256
`67520e1af930adfb65824574210b578bc3f1e4e123ab013f5a6c9dc64b09d794`.
`check_contract.py` pins this AST; `check_kernel.py` independently pins the full
file and permits exactly the documented integer-product candidate edit.

`Kernels.fp32Source_eq` connects the readable FP32 DSL specialization to the
retained complete source proof. Runtime dimensions, queried strides and program
IDs have a mathematical Nat interpretation in the DSL. `NativeScalar`,
`NativeMask`, `NativeAddress` and `NativeExecution` prove the actual signed-INT64
controls, loop limits, masks and four-byte pointer correspondence from admission
and already allocated buffer capacities. `KernelRewrite.native_product` proves
the candidate product is also exact in finite 64-bit word arithmetic.

`KernelReal.complete` specializes the existing complete Real result to one
kernel grid, with arbitrary Real inputs and initial output. The Real flattened
memory is logical word memory, not a byte heap. `KernelFP.success` proves actual
source and candidate execution success and connects the complete source stores
to the finite native schedule. `KernelFP.outputs` links each valid logical output
to its source writer and independent `FPReference.expected` recurrence.

The retained generic FP interpreter uses its `.real` compatibility channel for
some source registers, always with `some .fp32` compute precision here; physical
cells and final stores have `.fp32` tags. This channel contains native expression
words, not mathematical Real arithmetic or an implicit precision promotion.

Input bytes decode little-endian into raw FP32 leaves. `KernelTensor.Inputs`
connects metadata to the shared `VeriTile.Torch.FP32.TensorLaws` storage model;
`input_load` proves the checked continuous, nonnegative-view input reads are the
logical Tensor values, including nonzero storage offsets. `Output` records the
already bound output buffer. Its physical coordinate and the four serialized
output bytes are proved in `output_address` and `output_bytes`. No allocator or
returned-Tensor identity theorem is needed. For a fresh nonempty output,
the ordinary `empty_like(a)` API binding supplies the canonical queried strides
used by `call p`, for the admitted canonical-contiguous A. Contiguity alone is
not claimed to imply literal singleton-dimension strides. A supplied output's
queried strides are instead tied to the existing explicit Python guard.

The shared API and memory assumptions have these references:

- Torch [data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
  [storage_offset](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html),
  [is_contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html),
  and [resolve_neg](https://docs.pytorch.org/docs/stable/generated/torch.resolve_neg.html).
- Triton [load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
  and [store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
  use pointer element types and masked access. Active FP32 accesses are modeled
  as four-byte aligned typed loads/stores; masked-off pointers are not dereferenced.
- NVIDIA PTX [memory instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions)
  specify natural address alignment and typed memory operations. The public
  contract checks the actual nonempty data pointer rather than assuming every
  possible Torch storage is aligned.
- The CUDA Programming Guide's [size and alignment requirements](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#size-and-alignment-requirement)
  describe the alignment of ordinary device allocations. The already allocated
  fresh output binding exposes sufficient alignment as an explicit API fact;
  supplied output instead obtains it from the existing Python input guard.
- Triton [dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
  exposes input precision and accumulation behavior; the selected default FP32
  profile uses TF32 inputs with FP32 accumulation.
- NVIDIA PTX [matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions)
  and [asynchronous warpgroup matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions)
  define the fixed MMA/WGMMA families and live accumulator operand.

`NativeGraph.consume` is structural compiler staging of the pending source dot
and its immediate accumulator addition. `DirectFP.interpret` expands the staged
node into the retained proved fixed-K8 plan. The primitive receives only two
8-word vectors, the actual reached carry, fixed instruction/output coordinate
and live=true. No variable-sized dot equality or full-program result is assumed.
The 36 offline Triton 3.7.1 SM90 profiles check both candidate versions and both
residual branches, including live carry predicates. This evidence is bounded
to that compiler/target profile, not a universal compiler proof.

Literal/cast/binary/MMA primitive implementations and compiler correspondence
are explicit external model boundaries. They are not deferred numerical
transformation experiments. The candidate has no floating transformation,
finite-word restriction, numerical rule premise or pending numerical experiment.
