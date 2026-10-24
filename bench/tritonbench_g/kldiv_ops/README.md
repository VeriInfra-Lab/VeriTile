# kldiv_ops

`Kernels.lean` models both complete JIT kernels from `kldiv_ops.py`: both target
flags, the elementwise and reduced forward modes, and all column blocks. The
existing candidate commutes the exact signed output-row address product. It
leaves floating expressions, precision, loads, branches and stores unchanged;
this port does not claim a numerical optimization or a measured speedup.

The floating source family contains 216 default-forward, 72 log-forward,
36 default-backward and 12 log-backward storage profiles. `Proofs/SourceDomain.lean`
separates formal AST profiles from the pinned Triton frontend domain, including
raw-exp restrictions, the fp32 loop carry, and nonzero fp8 scalar restrictions.
See [source_scope.md](source_scope.md) for their source rationale and boundaries.

The public entries are:

- `Correct.lean`: six complete ideal-real flat-memory contracts, each quantified
  over source and candidate, independent row strides and every covered flat
  allocation. They prove actual source trace safety, termination, the
  mathematical row loss or gradient, and preservation of every other raw memory
  cell. The named IO contracts live in `Proofs/Contracts.lean`; dtype erasure
  is explicit.
- `FPEquiv.lean`: four complete typed state/failure equivalences, with an empty
  numerical-assumption list because the existing transformation changes only
  exact signed address arithmetic.
- `Proofs/NumericExecution.lean`: actual nonempty typed executions under a concrete
  exact-real model for the operations used here. Both target flags, both forward
  reduction branches and both variants use three columns with block size two,
  program id one and unequal strides. The checked output cells contain -2 in
  elementwise forward mode, -6 in reduced forward mode, and -1 in backward
  mode; the preceding and masked-tail sentinels remain 7. These are model
  witnesses, not IEEE or native GPU certificates.
- `Proofs/RealWitness.lean`: closed finite-allocation executions for both flags,
  both variants and every forward reduction code. Padded tail loads lie outside
  the allocation, while all active addresses are proved in bounds.
- `Proofs/Precision.lean` and `Proofs/ForwardPrecision.lean`: independent exact
  AST checks for all 216 default-forward and 72 log-forward formal profiles,
  including both masked loads, local clamp/log/exp precision, subtraction and
  multiplication, elementwise output conversion, sum/carry and post-loop store.
  Admission implies fp32 accumulation for every supported nonzero reduction.
  Control, the original fp32 zero initializer and backward weak scalars are
  checked separately.

Supporting full-loop invariants, safety proofs, real projection and flattening
bridges live under `Proofs/`. The former `KldivOps.lean` helpers are preserved in
`Proofs/LegacyReal.lean`; its eight one-block specifications are declared in
`Correct.lean` so inventory discovery continues to include them. Their original
statements, documentation and proofs are unchanged. The new whole-row contracts
are separate specifications.

Local verification on 2026-10-13: the dependency-aware `Correct` and `FPEquiv`
build passed against the refreshed shared library (3479 jobs), including the
named IO contracts, complete precision gates, actual numeric output and sentinel
checks, and finite-allocation witnesses. Both public entries together reach all
42 local Lean modules. The final 33-declaration axiom audit passed: all 18 public
headlines (including the eight preserved legacy declarations), four finite
execution/safety witnesses, four typed execution definitions, two numeric
readback theorems and five precision declarations use only standard Lean axioms.
The four FP assumption printers report `none`; all three independent mathematical
formulas pass the non-circularity check. All 45 historical declaration headers
and docstrings are preserved across the two files; `git diff --check` passes
for this directory. The public-specification relocation also passes a combined
3482-job build and the same 33-declaration audit. All eighteen coverage records
are migrated and validated against the complete 42-file source/support closure.
The two regenerated specification sheets have no self-reference or missing summary.

The latest receipts are `/tmp/veritile-public-spec-relocation-build.log` and
`/tmp/kldiv_ops_relocated_final_audit.log`; the exhaustive forward precision build is also
recorded in `/tmp/kldiv_ops_forward_precision_build.log` (3414 jobs).
Independent comparator replay and corpus integration remain pending; local
compilation does not complete those gates. Host reductions/normalization, host
`grad_output` multiplication, native integer overflow and device/grid scheduling
remain the explicit boundaries of these JIT-kernel contracts.
