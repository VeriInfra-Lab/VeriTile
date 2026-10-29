# matmul_tma

Start with `Kernels.lean`, `Correct.lean`, and `FPEquiv.lean`. The source Python
is unchanged. `MatmulTma.lean` and `FPCorrect.lean` are compatibility imports;
all seven historical specifications and their original scopes are retained in
`Proofs/LegacyReal.lean` and `Proofs/LegacyFP.lean`.

The wrapper creates **FP16 A and B internally**, optionally transposes each,
allocates FP32 C, replaces C with a new FP16 allocation when requested, launches
exactly one program with `BLOCK_M/N/K=M/N/K`, and returns that selected tensor.
The public contracts quantify arbitrary dimensions and all eight flag
combinations. They do not introduce FP32 inputs, additional tiles, or a K loop.
The wrapper accepts no external tensor, so external lazy-negative/conjugate
views are not input cases of this API.

`Correct.lean` states the independent real matrix product
`sum_k A[i,k] * B[k,j]`, proves the shared typed source's real projection, and
composes the existing complete execution proof. A whole-wrapper theorem also
instantiates the same allocation/launch/return path with identity casts and an
independent finite-sum Real dot, deriving mathematical returned tensor cells and
the frame relative to pre-allocation memory. The flat return theorem allows
the full allocation list, including the unused FP32 output in the FP16 branch.
Transpose address bounds and contiguous output injectivity are derived from the
wrapper metadata; input/output windows and every off-output flat cell are
covered.

`FPEquiv.lean` composes the typed execution through the actual randn/empty call
order, transpose strides, source request, selected allocation, readback, and
return. The discarded first output is framed. The allocator's ordinary API
frame additionally gives a frame relative to memory before the first randn.
Native request composition includes the three extracted pointers, block-pointer
shapes/strides/orders, grid `(1,1)`, `NUM_CTAS`, `NUM_WARPS`, FP16 operands and
FP32 dot accumulation. Byte-range bounds follow from ordinary pointer and
allocation metadata. Allocation/setup failures propagate without running the
source.

The numerical layers remain distinct:

- The complete structural result has an explicit typed dot and cast backend.
- The whole-wrapper numerical registration entry exposes a local row/column
  dot and two scalar conversion ports through `Proofs/PrimitivePorts.lean`.
  Their explicit laws are composed into `matmul_tma_primitive_wrapper`,
  including native bounds, discarded-output and pre-allocation frames. The
  ports cannot inspect memory, allocator responses or final output buffers.
- The retained cast result supplies direct binary16 output conversion.
- The closed software profile decodes half inputs, separately rounds FP32
  products and additions in ascending K order from positive zero, and performs
  the selected output conversion. It is not asserted equal to WGMMA.

Native admission explicitly requires power-of-two block dimensions and a
successful response for the exact packed configuration. Power-of-two sizes
alone do not assert tensor-core, architecture, resource, NUM_CTAS or NUM_WARPS
support; rejected requests remain setup failures.

Successful Torch allocation responses, allocation freshness/capacity/frame,
extracted pointer metadata and exact native setup admission are ordinary API
boundary premises; none supplies final output values or successful kernel
execution. No compiler, Torch, allocator or runtime implementation theorem is
introduced. Native WGMMA accumulation order/precision, fusion, special-value
policies and native output conversion conformance remain numerical obligations.
No admitted FP rewrite atom or new axiom is used by these execution contracts.

Targeted validation on 2026-10-17 rebuilt all owned Lean modules and preserved
the original typed software witness. The unchanged Python wrapper AST passed
all eight flags at both `16×32×64` and `128×128×128`, checking allocation order,
view strides, output replacement and return identity using CPU tensors and a
captured launch. Offline Triton SM90 compilation passed both output signatures
with runtime shape/stride arguments, producing FP16-input/FP32-accumulator
WGMMA and the FP16 conversion only in the half branch. This is compilation and
host-metadata evidence; **no GPU execution or hardware equivalence is claimed**.
Historical review and evidence are recorded in the earlier checkout at
`bench/review-checkpoints/2026-10-17/matmul-tma-whole-task/` and
`bench/task-reviews/matmul_tma.json`; see the provenance in the current
[task checklist](../task_progress.md). Those artifacts have not been copied
into this checkout. The historical task is accepted with explicitly
deferred numerical validation. All seven historical public statements remain
available. The live numerical queue records three primitive-rule families,
24 exact bindings and eight flag configurations; its actual consumer passed.
The independent Real theorem and complete wrapper proof supply correctness
within the recorded source/API scope; no native numerical validation is implied.
