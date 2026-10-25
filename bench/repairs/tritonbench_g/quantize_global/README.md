# Global FP32 quantization: kernel proof

The selected existing repair reads FP32 input and an FP32 absolute-maximum
buffer and writes signed INT8 output. The original benchmark remains separate.
The Python input contract and its existing segmentation are retained.

## Mathematical target and kernel scope

The maximum `m` is a declared kernel input. The public mathematical condition
is `m >= 0` and `abs(x[i]) <= m` for every logical input element. The independent
output formula is

```
q[i] = 0                         if m = 0
q[i] = nearest_even(127*x[i]/m)   otherwise
```

The theorem proves the result lies in `[-127,127]`, so the final signed INT8
interpretation matches that mathematical integer. It includes zero inputs,
zero maximum, positive and negative half ties, and masked tails. Supplying the
maximum produced by `abs().amax()` meets the scale condition through the Torch
API boundary. The kernel proof does not implement or prove that host reduction.
The existing independent maximum-tree mathematics remains available for reuse.

The actual repair uses `tl.div_rn`, then a separate FP32 multiplication by 127,
explicit lower/upper clamps, signed64 `llrint`, and an INT8 store. Real is the
mathematical specification, not a claim that native arithmetic is exact.

The complete selected Python function body is pasted unchanged in
Kernels.lean. All thirteen statements, signed INT64 casts, masks, clamps,
llrint and the implicit INT8 store conversion remain. Pointer storage, the two
annotated runtime INT64 arguments and the block constexpr are supplied outside
the body. The source checker compares the complete body and actual call
bindings, including the precise division spelling.

The frontend emits a distinct Op.divRN node and requires an independent
FP32 primitive callback; it never falls back to ordinary division or a
reciprocal graph. FP64 and unknown floating-family operands must first be
converted explicitly to FP32. The mathematical interpretation uses division;
the floating interpretation retains the precise primitive and operand-local
precision/conversion scopes. Proofs/Model.lean binds the existing Scalar.div
port only to this precise callback. Ordinary division keeps its own backend.
The existing native index bridge, INT64 llrint executor, modular INT8 storage,
numerical ports and public contracts remain.

Proofs/Source.lean identifies the actual precise assignment and proves the
complete signed-source execution against the independent per-element scalar
graph. Strict Nat/Int identities connect the mathematical coordinates.
The derived signed INT8 conversion and integer-stage byte storage are proved
idempotent using integer modular arithmetic; no floating identity is assumed.
Proofs/KernelSafety.lean checks actual loads and stores, including the
unconditional maximum read. Current frozen-source receipts record the full
proof replay separately from the historical CPU checks below.

## Admission and memory binding

Python requires a strided, contiguous CUDA FP32 Tensor, no unresolved negative
view flag, fewer than `2**63` elements, and a four-byte-aligned nonempty data
pointer. Nonempty inputs must have finite values. Scalar shapes and empty
shapes are admitted; empty input skips the finite scan and launches no kernel.
The finite rejection remains after the first output allocation, matching the
existing Python source. Both returned tensors stay on the input device.

`Contract.lean` and `TensorBridge.python_checks_iff_preconditions` state the same guards.
`Proofs/InputBinding.lean` reuses core Torch FP32 storage, stride, storage-offset
and pointer laws. `bindInput` constructs the regional input snapshot from that
actual view; `original_logical_input` identifies each read with the caller's
logical element. It adds no runtime copy. `FiniteObservation` binds the
nonempty `isfinite(...).all()` observation to those same FP32 words, and
`checked_finite` derives their finiteness from the checked contract. Empty input
requires no finite observation.

Output and maximum buffers are supplied at the kernel API boundary with their
ordinary bounds, dtype, alignment and freshness. Allocator internals and the
returned Tensor objects are outside the public kernel specifications.

## Public proof entries

- `IO.lean` declares the payload, scalar maximum and signed integer output ports,
  exact active windows, source/integer-storage bindings, all repeated autotune
  trials, independent formula, and logical scale domain. Source storage remains
  FP32/FP32/INT8; logical `.real` channels carry the selected floating interpretation.
- `Correct.lean` has one complete-grid `IntegerProgramIO.ImplementsOn` specification
  and a one-call proof. `FPEquiv.lean` compares two actual nonempty autotune
  schedules as `original ≡[[]] candidate`, including the retained 1024- and
  2048-lane configurations. The shared integer program view retains each
  actual invocation, integer-storage binding and observer. `Proofs/Pair.lean`
  reuses the existing complete `IntegerProgramIO.Equivalent` proof, including
  both safe successful runs, all typed output cells and exterior frames.
- `Proofs/KernelSafety.lean` proves every actual source access, including the
  unconditional maximum load and masked integer store. `Proofs/IOCorrect.lean`
  connects the unchanged executor/schedule, input binding, all output cells and
  exact exterior frame to the shared IO judgments using existing source results.
- `Misc/KernelRealResults.lean` and `Misc/KernelNativeResults.lean` retain the
  previous raw-state statements, including the per-program Real result and
  full-memory native equality. They do not import public reader entries.
- `Misc/TensorBridge.lean` retains the original admitted Tensor specialization,
  input binding helpers, finite observation, and Python-check correspondence.
- Earlier wrapper and software-reference results remain in
  `Misc/LegacyHostCorrect.lean` and `Misc/LegacyHostFPEquiv.lean`, with their
  original theorem namespaces and assumptions. The new public entries do not
  import the host executor or primitive-conformance helpers.

**Numerical transformation assumptions: none.** Changing the lane partition
and repeating autotune trials changes no scalar floating operation. Basic
FP32 division/multiplication, comparison, `llrint`, narrowing and Torch abs/max
semantics are modeling boundaries, not deferred per-task experiments. The
optional prior software-reference conformance premises are retained with that
classification; they do not determine current acceptance.

## Source defects, references and validation

The original reciprocal graph has concrete exceptional cases: for zero maximum,
`0 * (1/0)` is NaN; for FP32 `x=m=2**-149`, `1/m` overflows and the original
scaled value becomes infinity before `llrint`. The repaired division graph
returns mathematical zero in the first case and INT8 127 in the second under
the selected FP32 profile. Equivalence to the defective original graph is not
claimed. Existing layout/device and native-index repair scope is preserved.

Primitive/storage references are
[libdevice `__nv_llrintf`](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrintf.html),
[Triton `div_rn`](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html),
[PTX address alignment](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands),
and the core Torch binding references for
[`data_ptr`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html)
and [`storage_offset`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html).
FP32 accesses use four-byte alignment; INT8 stores use one byte.

```sh
lake build bench.repairs.tritonbench_g.quantize_global.Correct \
  bench.repairs.tritonbench_g.quantize_global.FPEquiv \
  bench.repairs.tritonbench_g.quantize_global.Misc.TensorBridge \
  bench.repairs.tritonbench_g.quantize_global.Misc.LegacyHostCorrect \
  bench.repairs.tritonbench_g.quantize_global.Misc.LegacyHostFPEquiv
python bench/repairs/tritonbench_g/quantize_global/check_contract.py
```

The CPU checker executes the actual Python JIT/wrapper AST: 32 complete
output-pair/frame cases and 13 rejections, covering both block sizes, repeated
trials, empty/scalar/zero/subnormal/max-finite inputs, ties, tails, storage
offsets and devices. It also checks Python/Lean guards, the complete source
projection and the original zero/tiny reciprocal counterexamples. Existing
segmented endpoint checks are retained; no new runtime compatibility scheme is
introduced. CUDA is mocked. No GPU execution or independent numerical replay is
claimed; the shared runtime grid-product work remains deferred.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
