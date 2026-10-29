# KLDiv Triton

Historical contracts, the new full-source public contracts, and the typed
numerical execution regressions compile against the current shared library.
All eleven public declarations have reviewed source/support fingerprints;
independent replay and corpus integration remain pending, so this port is not
registered as complete.

## Actual source

Forward retains signed loop counters, three row strides, both target branches,
and both reduction paths. Reduction zero stores the pointwise KL term. Every
other mode stores **one sum per executed chunk**, incrementing the destination
pointer after each store. An empty row performs no store. This differs from
`kldiv_ops`, which accumulates a row sum and performs a final scalar store.

The wrapper allocates one scalar per row for nonzero reduction. That allocation
is not generally sufficient for the source's multiple-chunk writes. Per-program
proofs must require bounds for the actual chunk footprint; they cannot establish
global race freedom or wrapper safety by replacing it with a single row sum.
For one row with 16385 columns, the wrapper caps the block width at 16384 and
allocates one reduced-output cell, while the source writes two. The second
write is out of bounds; `wrapper_single_row_allocation_insufficient` proves
this footprint conflict using the actual wrapper dimensions.

Backward preserves the output-then-target pointer initialization order and the
pre-loop offsets and mask, then executes every column block. Host multiplication
by an external output gradient is outside this kernel.

## Floating source scope

Forward independently supports six prediction and output carriers (fp32, fp64,
fp16, bf16, f8e4, f8e5), with fp32/fp64 target storage. Default backward
keeps six independent target/output carriers as an abstract AST family, but its
source-supported targets are fp32/fp64/fp16/bf16: the pinned frontend lacks
nonzero fp8 scalar constructors for `target * -1`. Log-target backward has
fp32/fp64 targets and six outputs. The shared negative-literal translation repair and this port's exact full-profile
precision checks pass. The wrapper's forward output is fp32; the low-level source model
also exposes other destination storage. Explicit final casts represent the
implicit destination conversion. The exact AST regression checks pass against
the current shared-library changes.

Pinned [Triton 3.5.1 math.py](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/math.py)
accepts only fp32/fp64 at the direct logarithm and exponential calls. These
frontend families do not establish native-device support for every storage
combination. Discrete conversions, fixed-width index overflow and signed
negative strides are not silently covered by the floating/Nat-stride model.

## Proof architecture

`Kernels.lean` contains preserved historical source and the new typed full
sources. The candidate changes only the order of the exact signed output-row
address multiplication. The entire loop and numerical expression remain shared.

`Correct.lean` retains five historical contracts. New independent formulas,
actual execution, full raw memory frame, trace safety, and flat-memory transport
are proved under `Proofs/`. Full-loop snapshot formulas require output
separation from inputs that future chunks will read; no injectivity assumption
is added to source execution or floating-point equivalence.

`FPEquiv.lean` compares complete state/failure results for every backend, including
aliases and inactive-lane arithmetic. All three source-facing FP declarations
compile and print no numerical assumptions. This is distinct from an
independent acceptance replay, which remains
pending even after local Lean gates pass.

Real specifications use totalized `Real.log`. Their zero-padding identity does
not assert that IEEE `0 * log(0)` vanishes. Numerical witnesses use an explicitly
identified exact-real backend, not a fabricated IEEE implementation.

`Proofs/Execution.lean` retains the typed mixed-storage, empty-row and two-loop
execution regressions. `Proofs/Witness.lean` additionally instantiates the real
contracts with finite disjoint allocations and a nonzero program id, including
an empty row with zero-sized regions; these finite-allocation witnesses compile.
The typed execution regression passes with both loop blocks composed explicitly
before body simplification. It checks separate reduced values -4 and -2,
pointwise and backward results, empty rows, half-precision weak literals and
untouched tagged sentinels. Independent replay remains pending for all new contracts.
