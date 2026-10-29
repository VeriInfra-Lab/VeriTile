# masked_select: kernel-only stable selection

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

The independent target is the stable compacted list

```
[input[i] for i in range(n) if mask[i]]
```

Its length is the number of true mask entries. A selected input index `i`
writes to the number of selected entries strictly before `i`. The actual
kernel obtains that position by subtracting one from the inclusive prefix.

## Source and precision

`Kernels.lean` reuses the existing readable typed Triton DSL in the original
task's `Kernels.lean`; both public entries reference that exact shared source.
FP32 payloads are copied without arithmetic. The mask is Bool; Python's Bool
`cumsum` produces Int64 prefixes; output storage remains FP32. Every FP32 bit
pattern is preserved, including NaNs and signed zeros. No BF16 variant or new
Python optimization was added.

The original twenty autotune configurations are the actual implementation
variants: block widths 256, 512, 1024, 2048 and 4096, each with four warp counts.
`FPEquiv.lean` proves any two configurations complete and produce the same
active outputs and exterior frame through shared `ProgramIO.EquivalentOnCells`.
The retained underlying helper also proves equality of the entire memory.
It compares complete executions of these real source instances;
local registers and program counts need not agree. Existing kernel-grid
composition is reused. It does not prove the host autotuner or launch API.

## Input contract and memory meaning

The checked Python wrapper is unchanged by this revision. Before preparation,
both arguments must be strided tensors, FP32 input and Bool mask, on the same
CUDA device, with broadcastable shapes and `0 < broadcast.numel < 2**31`.
The Bool mask must have no negative view bit. Arbitrary ordinary input strides
remain supported by the original broadcast/contiguous producer operations.
The existing repair resolves a lazy-negative input before its original
contiguous operation. No new copy or compatibility path is introduced here.

After preparation, Python checks the actual FP32 input pointer for four-byte
alignment. `checkInputs_iff` and `checkPrepared_iff` prove exact agreement with
`InputContract` and `PreparedContract`. Empty broadcast inputs retain their
explicit rejection; a nonempty all-false mask correctly produces empty output.

`KernelInputs` states the actual three input-memory contents: prepared payload
values, Bool mask values, and inclusive prefix counts. It contains no expected
output values or kernel-success premise. Torch's broadcast/contiguous/cumsum
APIs supply those input buffers under their ordinary primitive contracts; their
internal implementation and wrapper execution are outside this task's scope.

`KernelBuffers` supplies the actual contiguous storage descriptors, with
4/1/8/4-byte widths, bounds, alignment and a disjoint fresh output. Read buffers
may alias. `TensorBridge.prepared_memory_meaning` binds the checked pointer to the FP32
input descriptor and proves exact byte addresses of active accesses.

The source interpreter represents the nonnegative Int64 prefix by a Nat
projection. This is not a claim that Nat subtraction equals arbitrary signed
subtraction. `KernelResults.prefix_int64_meaning` proves exact native prefix values and
selected destinations under the stated extent bound. A zero prefix gives
native `-1`, but proves that lane unselected, so it cannot store. The theorem
`TensorBridge.checked_native_indices` establishes signed-int32 offsets and masks for every
padded lane of every original configuration. These are program/address proofs,
not numerical experiment assumptions.

## Public proof entries

- `IO.lean`: named shared `ProgramIO` with all three input channels, actual
  selected source/configuration schedule, dense reads, active compact output,
  no scratch, and independent stable-list formula. `IO.Prefixes` requires the
  input prefix to equal inclusive mask counts. Exact typed-cell input binding
  preserves the original input domain; read buffers may still alias.
- `Correct.masked_select_stable_correctness`: actual full-grid access safety,
  successful execution, independent stable list and every exterior cell.
- `FPEquiv.masked_select_configuration_equiv`: `original ≡[[]] configured` for any two
  original configurations, with both source traces safe and complete. Opaque
  carriers include every FP32 word. There is no numerical transformation law.

The source-bearing `ProgramIO.Implementation` view fixes the exact-cell input
binding, inclusive-prefix domain and complete invocation schedules in its
type. It reuses the original `EquivalentOnCells` proof without weakening the
domain or output/frame conclusions. The old IO statement remains the ordinary
`masked_select_configuration_io_equiv` theorem in `Misc/NativeCorrect.lean`.

`Proofs/PublicIO.lean` reuses the existing kernel-grid proof and follows its
actual predecessor states to prove source safety. `Proofs/KernelResults.lean`
retains exact original raw program/full-grid and prefix statements;
`Proofs/ConfigurationEquivalence.lean` retains the original whole-memory scope.
`Misc/TensorBridge.lean` preserves the checked native-index and physical-buffer
statements. Helpers do not import reader entries. Historical host results remain
in `Proofs/LegacyCorrect.lean` and `Proofs/LegacyFPEquiv.lean`.

## Original defects and model boundaries

The original Python benchmark remains unchanged. Its contiguous negative-view
counterexample is retained: logical `[-1,-2,-3,-4,-5]` shares positive storage,
and the unmodified source returns `[1,2,3,4,5]`. The repair already resolves that
metadata. The original empty-input `prefix[-1]` failure is also retained in the
historical proof; the checked wrapper rejects empty broadcast extents.

There are no deferred numerical transformations. Typed raw loads/stores,
metadata observation, successful library buffer preparation/allocation,
compiler/runtime correspondence and absence of concurrent external writes are
primitive model boundaries. No experiment is requested to establish basic
copy semantics. Relevant API references are PyTorch
[`cumsum`](https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html),
[`contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html),
[`resolve_neg`](https://docs.pytorch.org/docs/stable/generated/torch.resolve_neg.html),
and Triton [`load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
/ [`store`](https://triton-lang.org/main/python-api/generated/triton.language.store.html).

## Validation

The targeted Lean build checks both public entries and the retained optional
helpers and TensorBridge, with standard-base axioms only. The shared structural
equivalence uses an empty numerical-rule set. The independent stable formula is checked not to reference
kernel execution.

`Proofs/check_contract.py` retains seven raw-word output/input-frame cases,
ten rejections and the original negative-view counterexample. Every one of the
five actual block widths is checked independently after resetting output
sentinels; earlier trials cannot hide missing writes. It also preserves input,
mask and prefix buffers. Kernel, broadcasting helper and configuration ASTs
still match the original source exactly. CUDA/allocation dispatch is mocked,
and CPU AST replay explicitly converts vector Python `and` to elementwise
conjunction. These checks are not a GPU run.

`check_cuda_lowering.py` compiles all twenty actual configurations offline for
SM90 with FP32/Bool/Int64/FP32 buffer ABIs. `cuda-lowering.json` records their
PTX hashes; checks confirm signed-int32 masks, 64-bit prefix loads and FP32 raw
loads/stores. GPU execution and independent numerical replay are not claimed.

`Misc/TensorBridge.lean` exposes optional check correspondences and prepared-buffer
bindings as ordinary theorems. The primary files do not import this bridge;
`Correct.prefix_int64_meaning` remains supporting kernel arithmetic.
