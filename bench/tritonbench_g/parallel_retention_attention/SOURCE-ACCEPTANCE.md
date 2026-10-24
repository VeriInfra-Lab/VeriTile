# DQ and backward helper composition

`SourceCorrect.lean` adds two backend-parametric source contracts:

- `query_fp32` covers the complete original `_parallel_retention_bwd_dq`
  helper at fp32 storage: both loops, the two different scale sites, original
  casts and dot options, boundary loads, final store and complete memory frame.
- `backward_helpers_fp32` covers DQ, the barrier marker, then DK/DV after
  caller index specialization. Its independent reference lets DK/DV read
  memory after DQ stores, retaining arbitrary source-model aliases.

The reference uses initial-memory reads, ordered primitive calls and store
folds. It does not call the source interpreter or assume final output values.
All source-model dimensions, strides, initial memory and undefined values
remain parameterized. The source explicitly declares its input/output regions.
Unspecified block-pointer padding is not replaced with zero.

The existing forward O and standalone DK/DV contracts in `FPCorrect.lean`
remain unchanged. The new contracts do not identify the backend parameters
with a closed software bit model or native instructions. In particular, they
do not assume distributivity, reassociation, exact real exp/log, cast identity
or equality between dot-from-zero plus add and a native dot with live carry.

[Numerical assumptions](NUMERICAL_ASSUMPTIONS.json) records seven deferred
obligations with exact relations, corner cases and required captures. GPU
execution is **NOT_RUN**. Native integer widths, physical byte memory,
allocations, barriers and cross-program schedules remain separate. Host
NK/NV sums and autograd dispatch are outside these helper contracts.

The [proof package](../../review-checkpoints/2026-10-15/retention-query-source-proof/README.md)
records compilation, named axiom checks, public declarations and metadata
validation. The [shared experiment queue](../deferred_numerical_experiments.md)
also retains the five pending exact rule bindings and historical counterexamples.
