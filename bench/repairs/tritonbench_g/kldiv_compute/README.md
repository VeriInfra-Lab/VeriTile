# kldiv_compute kernel proof

The selected Python entry is unchanged: `kldiv_compute_repaired.py`, SHA256
`1565e81f560319d9f8ab63338920c17270468f248fed28ebd2b7edca3dc7eded`.
The public proof describes its complete original JIT kernel. The kernel writes
the pointwise term `x * log(x / y)` to the supplied fresh output Tensor; it does
not reduce the terms, clamp inputs, or extend the formula at zero targets.
Python allocation, device context management and returning that Tensor are
outside the kernel theorem. Their historical proofs remain available through
`Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`, with the old public files
preserved byte for byte and their documentation in `README.host.md`.

## Source and exact admission

`Kernels.lean` gives the full readable typed DSL: program ID, 1024-lane block
offset, mask, two masked FP32 loads with omitted padding, FP32 division,
`tl.log`, multiplication and the final masked store. `KernelDSL.source_eq`
proves literal typed AST equality with the previously verified source. The
readable `commutedKernel` restores the existing candidate from the original
task: `block_start = B * pid` instead of `pid * B`. Division, logarithm,
multiplication, masks and stores retain their original order and FP32 precision.
Inactive load values are arbitrary
at every dtype, region and address, and cannot affect an active output.

`Contract.lean` reuses exactly the selected Python input checks: strided CUDA
FP32 Tensors, equal shape/device, contiguous layout, no unresolved negative
view, existing `numel < 2^31` guard, and four-byte active pointer alignment.
Read-only input aliases, nonzero storage offsets, arbitrary rank, scalar and
empty shapes, and unused degenerate contiguous strides remain admitted. There
are no new value, size or stride restrictions. The supplied output has the
ordinary properties of the wrapper's fresh `empty_like` allocation. No input
packing, output copying or dtype conversion is introduced.

## Public guarantees

The public namespace is `VeriTile.Bench.Repairs.KldivCompute.Kernel`.
`Correct.correctness` proves the independent formula
`mathematical x y = x * Real.log (x / y)` for every logical output, using Lean's
total Real conventions. It constructs successful execution of every complete
typed source program over evolving logical typed memory. Real values have arbitrary
observations and are not forced through a finite byte codec.

`NativeCorrect.native_observation` retains the independent ordered native word graph
`multiply x (logarithm (divide x y))`, using the same ordinary FP32 primitive
interpretations as the typed source. The interpretations are computational
backend parameters, not premises equating the kernel or a whole output to a
reference. Every FP32 word, including exceptional values and signed zeros, is
admitted. There is no accuracy claim, correctly-rounded logarithm assumption,
cross-primitive numerical identity or deferred numerical experiment.

The optional `TensorBridge` results operate on actual
`VeriTile.Torch.FP32.Tensor` bindings and one shared physical memory. Logical inputs are read from each Tensor's physical
coordinate map in the initial memory. The proof establishes:

- Successful execution for any permutation of the complete requested grid.
- Every output value, and all four native output bytes for the native Tensor bridge.
- Every raw read and byte observation outside active output storage unchanged,
  including input backing storage and inactive output tails.
- Exact supplied data-pointer arithmetic, sign extension, four-byte alignment,
  backing and 64-bit address bounds for all active coordinates.
- Exact signed32 offsets and masks, including the inactive final tail, from the
  selected wrapper's existing size guard; complete grid coverage also holds
  for the empty case.

Ordinary Tensor laws, successful fresh allocation, physical read/write/frame
laws and the native little-endian FP32 byte codec remain explicit environment
boundaries. `Program.complete_correct` proves the source execution; output,
execution and frame correctness are not external assumptions.

## Preserved source defect and validation

The unchanged original flattens physical input addresses without checking the
logical layout. For `x=[1,1]` and `y=[1,1,2][::2]`, it reads `[1,1]` for `y`
and produces `[0,0]`, while the logical result is `[0,-log(2)]`. The selected
wrapper explicitly rejects that unsupported strided input. The conditional
packing/copy repair and `revision002/` remain separate historical variants.

`python bench/repairs/tritonbench_g/kldiv_compute/check_contract.py` verifies
source pins and original JIT equality, 15 rejection cases, 19 source/reference/
frame cases and the original stride counterexample. It covers empty/scalar
shapes, block tails, offsets, input aliases, degenerate contiguous strides,
exceptional FP32 values, one allocation and exact return identity. CUDA
metadata/context are adapted for CPU; these checks do not measure GPU numeric
behavior. Public and preserved host Lean endpoints are checked together with
standard-base axiom audits. `numerical-obligations.json` records zero deferred
experiments for the new kernel endpoints; historical local conformance ports
remain in `numerical-obligations.host.json`.

The retained grid results are `Kernel.Correct.correctness` and
`Kernel.NativeCorrect.native_observation`, over raw region parameters and
block geometry with the shared logical IO judgment. Their exact final-memory equations include every active
output and the full exterior frame. `TensorBridge.Real.correctness` and
`TensorBridge.Native.equivalence` retain the complete Tensor domain, arbitrary
admitted schedules, physical pointers/control and native byte observations.
Both routes reuse `Program.complete_correct`; the physical correspondence
remains in `KernelBinding` and `KernelExecution`. No primitive precision or
numerical boundary changes.

## Public proof entries (2026-10-21)

`FPEquiv.lean` contains one implementation-equivalence specification:

```lean
specification kldiv_equiv (p : Parameters) :
    original p ≡[assumptions] commuted p
```

`assumptions` is empty. `IO.lean` gives both actual typed sources the same
`MaskedKernelIO₂` input/output windows and tail mask. The theorem quantifies
over every program ID, every floating backend and every initial state. Both
executions succeed, agree on every active output, and preserve the masked
exterior memory. It imposes no finite/positive-value restriction; empty inputs
are covered, and the selected Python still uses block size 1024. This is integer address commutation,
not a change to the KL arithmetic or a performance claim.

`Proofs/Equivalence.lean` reuses the existing successful source execution and
frame proofs. `Correct.lean` retains its complete-grid independent Real target
and source access safety. `Misc/NativeCorrect.lean` retains the former full-grid
`ProgramIO.Implements` result as an ordinary auxiliary theorem, with its
original positive-block condition, primitive interpretations and all FP32
words. `Misc/TensorBridge.lean` retains the complete physical output/byte/frame
applications. These auxiliary results are separate from the two-program FP
headline; no prior grid or Tensor result has been discarded.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/NativeCorrect.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
