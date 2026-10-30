# isfinite: checked kernel contract

The public entries prove the direct strided kernels in
`isfinite_kernel_repaired.py`. The original benchmark remains unchanged in
`bench/tritonbench_g/isfinite_kernel/isfinite_kernel.py`. Its empty wrapper
has a concrete division-by-zero counterexample, replayed by `test_revision.py`.
The separate repair handles that case before launching. Existing chunking is
retained; no additional host orchestration or grid compatibility work is
required by these kernel proofs.

## Target and precision

The primary `Correct.lean` specification interprets inputs as mathematical
Reals. Every Real is finite, so each active output is `true`. This ideal
interpretation does not assign NaNs or infinities to the Real domain.

The retained all-word theorem in `Misc/WordCorrect.lean` instead uses the
independent IEEE exponent-field formula. At each active index its output is:

```
FP32: ((input_word >> 23) & 255) != 255
FP64: ((input_word >> 52) & 2047) != 2047
```

Both source precision branches are retained. The default is FP32; FP64 is an
explicit branch of the original helper. Every word is admitted: signed zeros,
subnormals, normals, infinities and all quiet/signaling NaN payloads. The
mathematical bit formula does not reinterpret these words as arbitrary reals.
It retains its original theorem name and statement as an auxiliary result.
There is no finiteness guard.

The original direct source uses `pid * BLOCK_SIZE`; its displayed candidate
uses `BLOCK_SIZE * pid`. All floating operations, same-precision casts, loads,
masks and Bool stores are identical. This exact integer transformation needs
no numerical assumption or experiment and makes no performance claim.

## Input contract

Both operands are rank-one Tensor or original raw `StridedBuffer` objects
backed by strided Tensor storage, with equal lengths and the same CUDA device.
Input dtype is FP32 or FP64 and output dtype is Bool. Ordinary Tensor operands
reject an unresolved negative view bit. Raw `StridedBuffer` preserves its
original documented raw pointer/dtype reinterpretation semantics.

Shape and stride components are Python integers; length is nonnegative and
strides fit signed int64. For nonempty views, actual pointers have natural
element alignment, accessed bytes remain inside the backing storage, and the
storage interval fits the 64-bit pointer address space. Negative and zero
input strides remain supported; negative output strides remain supported.
Nonempty output stride is nonzero. Complete accessed byte footprints of input
and output must be disjoint, including when they share a backing allocation.
Empty views need no alignment, storage-bounds or nonzero-stride check.

`Contract.lean` contains these conditions and their executable metadata check.
`TensorBridge.python_checks_iff_preconditions` proves that the check admits exactly the
public contract. `contract_memory_meaning` derives physical byte bounds and
active output-address injectivity. `active_pointer` relates signed native
indices and the complete 64-bit pointer sum to these addresses. Pointer
arithmetic is converted to a natural address only after forming the complete
signed sum; no padded allocation is assumed for masked lanes.

## Files and public proofs

- `Kernels.lean`: the complete original `isfinite_func` helper and four raw
  Python writer quotations retain both precision profiles and the existing
  index-commuted candidate. Only the function bodies are pasted inside
  `triton { ... }`; pointer types, checked CUDA import aliases, runtime call
  bindings and `BLOCK_SIZE` are outside. Every original cast and mask remains.
  Region indices are absolute byte addresses: `View.byteStride`, signed
  pointer offsets and the proved native element-width scaling connect this
  convention to Python typed pointers, including negative and zero strides.
- `jit-correspondence.json`: five checked helper/body mappings, with distinct
  FP32/FP64 input profiles. Source imports are checked against the original
  Python bindings, and helper definitions must themselves have a checked pair.
- `Proofs/SourcePreservation.lean`: all executable statements are definitionally
  equal to the earlier proof view. The old collector omitted input port `x`;
  the current signature records it correctly. Whole-state execution, failures
  and actual access safety are unchanged for every primitive algebra. The
  public IO, mathematical/FP specifications and host input contract are
  unchanged.
- `IO.lean`: the shared public descriptor fixes the selected source and primitive
  classifier, FP32/FP64 storage interpretation, Boolean output, signed byte
  windows, masks and program ID. Both readers use this one layout.
- `Correct.lean`: `isfinite_real_correctness` proves the shared `ProgramIO.Implements`
  contract for the independent mathematical Real classifier `RealStorage.reference`.
  The existing active-output injectivity condition remains explicit.
- `Misc/WordCorrect.lean`: `Correct.isfinite_kernel_real` preserves the previous
  all-word exponent-field contract, including every FP32/FP64 special value.
- `FPEquiv.lean`: `isfinite_index_commuted_equiv` proves the shared two-source
  `originalProgram ≡[[]] commutedProgram` judgment for every scalar classifier.
  The shared program view is justified by the existing complete
  `ProgramIO.Equivalent` proof. Both actual
  access traces are safe; both runs succeed; all observed output cells and
  exterior frames agree. No numerical or finite-input premise is used, and
  equivalence does not require output injectivity.
- `Misc/RealEndpoints.lean` and `Misc/NativeEndpoints.lean`: the former raw
  state-level results, including exact entire-memory equality and the optional
  classified native result, retain their full statements as auxiliary theorems.
- `Proofs/Kernel.lean`: active-only readback and memory-frame arguments;
  injectivity is required only for active output addresses.
- `Proofs/Source.lean`: complete execution of each canonical DSL source.
- `Misc/LegacyCorrect.lean` and `Misc/LegacyFPEquiv.lean`: retained optional
  host-era results. Neither public entry imports host execution proofs.

The wrapper supplies assertions. Successful wrapper execution, returned
Tensor identity, allocation and device restoration are not acceptance
obligations for this task.

## Primitive model boundaries

There are **no deferred numerical transformation experiments**. The meaning
of CUDA `finitef` / `isfinited` is the all-word classification above. It is a
primitive-model boundary, as are same-precision casts, typed loads/stores,
metadata observation, storage lifetime and absence of concurrent mutation.
The old `Bytes.LocalClassifier` interface remains only for optional host
results; it is not a numerical experiment obligation.

References for the scalar operations and memory convention:

- NVIDIA libdevice [`__nv_finitef`](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_finitef.html)
  and [`__nv_isfinited`](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isfinited.html).
- Triton [`load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
  and [`store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html):
  typed pointers, masks and unspecified masked-load values.
- NVIDIA [PTX memory instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld):
  byte addresses and natural access-size alignment.

## Validation

`test_revision.py` passes 84 actual Triton CPU interpreter cases across both
sources and both precisions, plus 17 guard rejections and 38 admissions. Cases
include special values, positive/negative/zero input strides, negative output
strides, tails, raw reinterpretation, disjoint gaps in the same allocation,
arbitrary output-byte sentinels and empty input. Existing metadata-only large
extent controls remain historical wrapper checks, not a native GPU claim.

The CPU interpreter does not implement CUDA libdevice externs, so only those
two callbacks use `numpy.isfinite`. Indexing, masks, casts, loads and stores
execute the actual Triton sources. This checks structure, not native scalar
implementation conformance.

`check_cuda_lowering.py` compiles 16 SM90 profiles offline: two sources × two
precisions × four stride ABIs. The receipt `cuda-lowering.json` records 64-bit
pointers, signed-32-bit bounded masks and the FP64 path without a downcast.
No GPU run or independent numerical replay is claimed. The targeted Lean build
checks both public entries and retained optional host helpers, with only
standard-base axioms and no numerical assumptions in the equivalence theorem.

`Misc/TensorBridge.lean` holds the optional check equivalence, storage/pointer bindings,
and concrete-byte application; the primary kernel specifications stay in
`Correct.lean` and `FPEquiv.lean`.

The public headlines expose logical IO, precision, strided views and launch
geometry. Actual source access safety is connected through
`Proofs/AccessSafety.lean`; `Proofs/IO.lean` and `Proofs/EquivIO.lean` compose
the retained execution proofs with the shared judgments. The unchanged
checked-Tensor application is `TensorBridge.isfinite_kernel_correctness`,
using the retained source foundation and deriving injectivity from the guards.
The concrete-byte and pointer applications remain in `Misc/TensorBridge.lean`.

During this formatting pass, the unchanged six-module source dependency chain
was rebuilt against the current shared Structural algebra before the IO proofs
were checked. No source/contract change, numerical assumption, GPU run or
independent numerical replay was introduced.

Source readability cleanup (2026-10-22): `Kernels.lean` displays all four complete ordinary quotations for FP32/FP64 original/candidate sources. `Proofs/LegacySurface.lean` archives the previous macro; `Proofs/SourcePreservation.lean` proves both complete dispatched functions equal by precision cases and `rfl`, over every View, extent and block size. Dtype-dispatched classification, signed byte-pointer expressions, precision, input domains and existing completion status are unchanged.
