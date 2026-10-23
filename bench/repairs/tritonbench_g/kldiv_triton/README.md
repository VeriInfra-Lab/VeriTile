# kldiv_triton kernel proofs

The selected Python entry remains `kldiv_triton_repaired.py`, SHA256
`ea1be2fd98aadf2f294964417898b348da5cd9d99d18632a96d7fe966625340d`.
`Kernels.lean` contains the complete readable selected typed DSL. Its executable
source is unchanged. The public namespace is
`VeriTile.Bench.Repairs.KldivTriton.Kernel`.

## Exact kernel boundary

Forward writes each element loss for `none`. For `sum`, `mean` and `batchmean`,
it writes one FP32 reduction per chunk into the supplied `(rows, chunks)`
allocation. The source loop advances the output pointer after each chunk store.
There is no sequential row accumulator or final scalar reduction in this JIT.
Python's later Torch sum and normalization are outside the new kernel theorems.
Backward writes the unscaled `-target` or `-exp(target)` derivative directly to
the supplied destination. Python cotangent selection/multiplication is outside
the kernel result.

`Contract.lean` gives independent Real formulas. Raw target zero contributes
exactly zero; otherwise its loss is `target * (Real.log(target) - prediction)`.
The log-target loss is `Real.exp(target) * (target - prediction)`. Reduced
outputs are independent finite sums over their own chunks, with tail lanes
zeroed. Real log and arithmetic use Lean's total Real conventions and make no
claim about native Real accuracy or exceptional-value agreement.

## Admission and actual physical bindings

The exact selected wrapper checks are retained. Inputs are matching rank-two,
contiguous, strided FP32 CUDA Tensors on the same device, with resolved negative
views and four-byte active pointer alignment. Read-only forward inputs may
alias, and nonzero storage offsets, empty axes and unused degenerate strides
remain supported. No finiteness/positivity guard or new size cap is introduced.
Backward preserves the existing scalar-or-matrix cotangent check and its
whole-storage separation check for nonempty destinations. Empty destinations
skip that alias check and perform no writes.

`MatrixBinding` and `ViewBinding` bind the actual Tensor metadata, observed
pointer, shape and whole backing to physical keys in one shared memory.
The forward output has the ordinary properties of the existing fresh zero
allocation. The backward destination is the original supplied object; no new
buffer, packing or copy is modeled.

`Proofs/KernelBinding.lean` proves actual PID sign extension, signed row-stride
products, supplied pointer arithmetic, active alignment and all four backed
bytes. Signed loop offsets, tail masks and the terminal increment follow from
backing bounds. Ordinary signed64 stride metadata and representable successful
dispatch IDs remain explicit environment boundaries, without narrowing Python
admission.

## Public results

`Correct.forward` and `Correct.backward` use the shared `ProgramIO.Implements`
judgment over independent logical arrays. `IO.lean` declares the actual FP32
sources, typed channels, row strides, masks, empty scratch and complete launch
schedule. The judgment proves actual source access safety, successful complete
grid execution, all mathematical outputs and the exact regional frame. The retained
`TensorBridge.RealApplication.forward` and `backward` construct successful complete
DSL execution over evolving physical memory and prove every logical output
against the independent Real targets. Each row is run at the memory resulting
from preceding rows; any permutation of the complete row grid is supported.
`ForwardPost`/`BackwardPost` also include the interpreter's own complete row
footprint, full physical raw-read and observation complements, Tensor pointers,
alignment/backing bounds and finite control facts. The source frame justifies
that the physical realization includes the complete source write footprint.
Arbitrary Real cells require no finite-byte encoding assumption.

The ordinary `FPEquiv.forward` and `FPEquiv.backward` theorems retained in
`Misc/NativeCorrect.lean` use the same source-connected IO judgment
with the existing native primitive model. They prove every output word, actual
source access safety, successful grid execution and the complete typed frame. `TensorBridge.NativeApplication.forward` and `backward`
also prove all four little-endian output bytes against the independent source-ordered graphs in
`Proofs/SourceNative.lean`. Those graphs retain loss multiplication operand
order, raw-zero comparison/selection, logarithm and exponential operations,
raw multiply-by-minus-one and log-target unary negation. A `ReductionPlan`
records the actual ordered binary addition tree for every block width; no
reassociation, commutativity or whole-sum numerical law is assumed. Primitive
meanings are ordinary computational interpretation parameters shared by source
and reference, not deferred equivalence assumptions. All FP32 words remain
admitted, including exceptional values and signed zeros.

The public endpoints have zero deferred numerical experiments. They do not
claim a correctly-rounded logarithm/exponential or a cross-primitive identity.
The shared historical `Proofs/Native.lean` remains unchanged, including its
separate pinned compiler profiles and operand-selection rules.

The retained Triton 3.7.1 CUDA arch90 (`sm_90a` PTX) profiles explicitly enable
floating fusion. A read-only audit of all 90 reduced profiles (15 block/warp
configurations, both flags, modes 1/2/3) found 564 explicit loss multiplications,
972 separate reduction additions and no FMA crossing the loss/reduction boundary.
The 3,384 FMA instructions are internal to the raw-target logarithm primitive.
The selected and compiled conditional JIT function ASTs, including decorators,
are identical. This is bounded evidence for those retained profiles, not a claim
about every compiler or device. The producer and per-profile instruction lines
are in `.veritile-cache/kldiv-triton-no-cross-reduction-fma.py` and the adjacent
JSON receipt; no compiler or GPU execution was performed for this audit.

## Preserved host proofs and validation

Previous public files are preserved byte for byte as `Proofs/HostContract.lean`,
`Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`. Their original declaration
names and complete host results remain available. `README.host.md` and
`numerical-obligations.host.json` preserve their scope and historical local
conformance ports. The conditional generalized repair and original benchmark
are also unchanged.

The existing CPU checker passed 33 rejection cases, 5 ordinary failure prefixes
and 134 actual source/reference/frame cases, including empty axes, width 32769,
all modes/flags, tails, offsets, aliases, degenerate contiguous strides,
exceptional FP32 values and direct argument/return behavior. It reproduced four
original-source defects: padded NaNs, raw-zero NaNs, the wrong mean denominator
and an out-of-view second-chunk store. These remain separate historical defects.
The CPU checks adapt CUDA metadata/context routing; they are not GPU numerical
measurements. Targeted Lean source builds and joint public/host imports check
the kernel results with standard-base axiom audits.

`Correct.lean` and `Misc/NativeCorrect.lean` contain the mathematical and native
formulas, IO selections and short composition proofs. The former raw-state declarations remain
ordinary theorems in `Proofs/RawCorrect.lean` and `Proofs/RawNative.lean`.
`Proofs/KernelInputs.lean` connects active logical inputs to the existing whole
grid results. `Proofs/KernelSafety.lean` follows the actual predecessor states,
signed loops and both store branches: reduced forward output is one scalar per
source chunk. `Proofs/PublicIO.lean` connects those results to the shared fixed
`Structural.exec` schedule, logical outputs and exact cell frame. It does not
accept an arbitrary executor or assume source safety.

Every complete row permutation, stride, output-address injectivity condition and
empty case remains supported. Input capacity conditions come from the declared
channel windows. The original raw results and all Tensor applications remain
available, and no Python assertion, source precision, source quote, numerical
boundary or experiment status changed in this format pass.


The 2026-10-21 implementation-pair review records **proof connection pending** for the main
`FPEquiv.lean` entry. The original task has output-address-commuted candidates. They are not connected to these selected repaired sources, which retain the zero-target/tail guards and widened index casts.
Single-source native declarations now live in `Misc/NativeCorrect.lean` as ordinary
theorems, retaining their historical names and complete contracts. Their
`Implements`/`Observes` conclusions remain useful auxiliary results and are not
reported as a completed implementation pair. The independent Real `Correct`
target and the actual source, precision, domains, grids and observations are
unchanged. This entry correction introduces no numerical experiment.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.
