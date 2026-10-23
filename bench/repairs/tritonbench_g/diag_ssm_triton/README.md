# diag_ssm_triton: four kernel proofs

This candidate keeps all four floating kernel bodies: real FP32 forward and
backward, and complex64 forward and backward with interleaved FP32 components.
The original benchmark remains unchanged. Python adds explicit input assertions
and widens shape/index/loop arithmetic to signed int64 before products. It adds
no forward-input copies. The original backward `grad_y.contiguous()` remains.

## Mathematical target and kernel scope

For each batch `b` and dimension `d`:

```
h[0,b,d]   = s[b,d]
h[t+1,b,d] = h[t,b,d] * Lambda[d] + x[t,b,d]
y[t,b,d]   = h[t+1,b,d]
```

Backward starts with `carry = 0`, visits `t = L-1, ..., 0`, and computes:

```
gx[t,b,d] = gy[t,b,d] + carry
per_batch_gl[b,d] += gx[t,b,d] * conj(previous[t,b,d])
carry = gx[t,b,d] * conj(Lambda[d])
gs[b,d] = final carry
```

`previous[0]` is `s`; for `t > 0`, `previous[t]` is the backward kernel's
explicit input `y[t-1]`. Conjugation is identity for real values. When that
input is the mathematical forward output, the independent recurrence/adjoint
proofs identify the gradients of the forward recurrence. Native backward reads
the rounded saved output; a Real specification does not identify those words
with exact real values.

The backward **kernel** writes `per_batch_gl[B,D]`. The subsequent Torch
`sum(dim=0)`, host dispatch, allocations, saved-object identity and wrapper
execution are outside the public kernel proof. Their earlier helpers remain
optional in `Proofs/`.

## Input assertions and memory meaning

Forward requires strided CUDA tensors `s[B,D]`, `x[L,B,D]`, `Lambda[D]`, matching
dtype and device, contiguous storage, no unresolved negative/conjugate view
flags, and 4-byte-aligned actual data pointers for nonempty tensors. The selected
real dtype is FP32; the source's explicit complex64 branch is retained.

Incoming backward `grad_y` has the saved output's shape/device/dtype, strided
CUDA layout, resolved view flags and the same nonempty pointer alignment.
Its incoming strides may be noncontiguous: the source already calls
`contiguous()` before passing the dense tensor to the backward kernel. This is
an existing API boundary, not a newly added compatibility layer.

There is no finite-value guard or new shape cutoff. Empty time, empty batch,
empty dimension, partial tiles, nonzero storage offsets and read-only input
aliases remain included. Output buffers are separate allocations supplied by
the kernel API boundary. No modulo is evaluated by an empty launch.

| Python assertion | Formal contract |
| --- | --- |
| Tensor object | `Argument.tensor`, `AdmittedArguments` |
| Strided layout, FP32/complex64, CUDA | `TensorContract.layout`, `.dtype`, `.cuda` |
| Forward contiguity | `TensorContract.dense` |
| No negative/conjugate view flags | `TensorContract.resolved` |
| Nonempty `data_ptr() % 4 == 0` | `TensorContract.alignment` |
| Matching dtypes/devices and exact ranks/shapes | `InputContract` |
| Incoming cotangent matches saved output | `GradientContract` |

`Proofs/Contract.lean` proves exact equivalence of each check function and its
named contract. `Proofs/InputBinding.lean` connects metadata to actual original
tensors: FP32 component coordinates, sizes, pointers, dense layout and resolved
logical flags. Its signed-index bounds follow from ordinary nonwrapping backed
tensor storage and signed tensor sizes. They are not experimental assumptions.

`InputBinding.bindReadViews` constructs the regional input state from all input
views of a single byte heap. `bindReadViews_logical` proves its component reads
are the original dense, resolved tensors' logical values. This construction
permits identical and partially overlapping read-only views: no pairwise input
disjointness is assumed. Allocation of fresh outputs is an API boundary, not a
new Python input restriction or a proved allocator implementation.

The alignment condition follows the natural alignment of scalar `.b32`
accesses in [PTX ISA 8.7, addresses as operands](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#addresses-as-operands).
The existing offline lowering check confirms these predicated scalar accesses.

## Sources and public proof entries

- `Kernels.lean` selects FP32 from the single readable typed DSL in the original
  task's `Kernels.lean`. No second handwritten kernel AST is maintained.
- `IO.lean` gives the actual source selections and finite typed channels. The
  forward inputs are initial state, transition and all time rows; backward
  inputs are initial state, transition, saved rows through `L-2`, and all
  incoming-gradient rows. Complex channels expose real/imaginary FP32
  components separately. Outputs cover every time row and both terminal
  gradients. Input and output masks use the actual program id and tail lanes.
- `Correct.lean` contains four shared `UKernelIO.RegionalImplements`
  specifications. Each invocation executes the entire time loop at an arbitrary
  program id; these are per-program theorems, not an added grid-scheduling
  theorem. The independent formulas consume logical finite arrays and do not
  inspect a machine state. The shared judgment proves actual source access
  safety from the declared windows, successful execution, every active output,
  and the exact exterior raw-cell frame. Existing output-separation premises
  remain explicit; no `FlatAlloc.Disjoint` premise is added to these regional
  endpoints. Read-only aliases remain admitted; complex forward also retains
  its input/output region-alias case.
- `Proofs/IOBounds.lean` connects declared finite windows to the existing source
  bounds. `Proofs/IOSafety.lean` and the original complex safety proofs establish
  the evaluated source trace, including conditional previous-state loads.
  `Proofs/IOValues.lean` and `Proofs/IOComplexValues.lean` connect the existing
  recurrences to logical inputs; the result helpers reuse the existing
  successful source executions and exact frames. Zero time still includes
  forward initial/transition loads and backward transition loads plus the two
  zero terminal stores. No unused saved-output or initial-state storage is
  required by the zero-time backward path.
- `Misc/Auxiliary.lean` preserves the exact four prior regional and four stronger
  flat statements as ordinary theorems, along with the closed-form and adjoint
  aliases. The flat results retain their original `FlatAlloc.Disjoint` premise;
  the regional endpoints do not claim automatic flattened byte execution for
  aliased views. Mathematical derivations remain in `Proofs/Mathematics.lean`
  and `Proofs/Adjoint.lean`.
- `FPEquiv.lean` compares each actual typed source with its existing candidate:
  `pid * block` becomes `block * pid`. Floating operations, complex signs,
  branches, loops, masks and stores are unchanged. `Proofs/FPExecution.lean`
  proves successful typed execution for arbitrary sequence lengths, including
  zero, by register-definedness invariants. Existing checked index-rewrite
  proofs then give successful execution of both sources and equality of every
  typed memory cell. `Proofs/FPInterface.lean` holds those derivations; the four
  public entries are short compositions with successful standard program
  equivalence and an empty numerical-assumption list.
- `Contract.lean` contains metadata input contracts; input binding, execution,
  mathematical and rewrite helpers stay in `Proofs/`. Historical byte/host
  scalar graphs are explicitly optional in `Proofs/ScalarGraphs.lean`.

## Numerical assumptions and retained defects

**No numerical transformation assumption is used by these four equivalences.**
The candidate changes only an integer multiplication order. No floating
arithmetic, reduction ordering or fused/separate graph is rewritten, and no
finite-operand condition is needed on active or inactive lanes.

Native primitive meanings, dtype, rounding, FTZ and compiler fusion are modeling
boundaries. They are not per-task numerical experiments. Earlier optional
software/native conformance and residual-error helpers remain available, but
their premises are not deferred obligations for this kernel acceptance. Torch's
batch-sum implementation is likewise outside this kernel task. Independent
numerical replay is inapplicable to this discrete rewrite; GPU execution is
reported separately.

The original unchecked benchmark is not retroactively certified. Retained
concrete counterexamples include a contiguous lazy-negative input (`+1` stored
where its logical value requires `-1`) and an int32 shape-product wrap
(`2**30 * 2`). The candidate's public contracts reject unresolved views and its
int64 widening repairs the latter. The separately deferred shared Triton
grid-product/segmented-launch issue stays outside this work.

## Targeted validation

Build the public entries with the installed Lean toolchain:

```sh
lake build bench.repairs.tritonbench_g.diag_ssm_triton.Correct \
  bench.repairs.tritonbench_g.diag_ssm_triton.FPEquiv
python bench/repairs/tritonbench_g/diag_ssm_triton/check_source.py
python -m py_compile bench/repairs/tritonbench_g/diag_ssm_triton/diag_ssm_triton_repaired.py
```

`check_source.py` compares all four JIT ASTs to the original, erasing only the
integer widenings. Its CPU execution checks cover both branches, empty axes,
tails, nonzero imaginary gradients against autograd, offset/aliased input,
noncontiguous incoming gradients, assertion rejections and original defects.
CUDA admission and launch are mocked; this is not GPU evidence.

The public Correct/FPEquiv entries, retained Auxiliary results and unchanged
TensorBridge compile in the 2026-10-20 targeted IO-format build. All four
FP-assumption reports print `none`. Public statement-surface and axiom gates
check the eight headline entries. The earlier Python source checks and
`py_compile` evidence remain applicable because Python and both canonical
Kernels files are unchanged by this format update.

Python JIT bodies are unchanged by this kernel-only organization update. The
existing sixteen Triton 3.7.1 SM80 offline lowering checks remain applicable
(four kernels and integer ABI specializations). CUDA was unavailable in that
validation; no GPU run or native numerical conformance claim is made.

`Misc/TensorBridge.lean` retains its exact supplementary actual-input layouts,
logical read views, native bounds and Python-check correspondence. Independent
replay and GPU execution were not run for this format update.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/Auxiliary.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
