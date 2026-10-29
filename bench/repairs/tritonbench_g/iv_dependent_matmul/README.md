# Checked induction-variable matmul candidate

`iv_dependent_matmul_i64.py` is a separate candidate. The original benchmark is
unchanged: SHA-256 `dfd772c19ff1b007b3e6c474fa984a460d0e135593cc3d0f05d279c8746997d7`.
The public entries are `Correct.matmul_checked_real` and
`FPEquiv.matmul_checked_if_local`.

## Checked scope and preserved behavior

The wrapper checks integer, nonnegative dimensions; positive power-of-two block
widths at most `2^20`; the five documented schedule selectors; the original
FP32 default allocation dtype; and a CUDA device. It checks these explicit source
envelopes before allocation:

* `max(M+BM, N+BN, K+4*BK+3) < 2^63`, including the longest lookahead;
* `4*max(M*K, K*N, M*N) < 2^63` for active element-to-byte offsets;
* `ceil(M/BM)*ceil(N/BN) < 2^31` for signed program IDs.

Successful native compilation remains subject to ordinary Triton block/device
support. The wrapper preserves caller-supplied block sizes and all five pointer
schedules: `pre_load`, `post_load`, `post_pre_mixed`, `post_load_two_iters`, and
`post_load_three_iters`. Stages remain 4 for the last schedule and 3 otherwise;
there is no autotuner. No dtype, layout or device expansion is introduced.

The candidate widens metadata, PIDs, lanes and the induction variable before
integer arithmetic. Kernel AST equality after erasing exactly these casts and
renaming is checked against the original. FP32 operands, default TF32 dot
precision, accumulator update, explicit FP16 narrowing, and implicit widening
into the default-FP32 output are preserved. The wrapper keeps random A, random B,
then empty C allocation and returns that exact C object. Empty M/N returns after
those allocations with no launch. K=0 remains a launched zero-accumulator case.

## Source and proof chain

Original `SourceDefinitions.lean`, `ScheduleDefinitions.lean` and
`HostDefinitions.lean` hold the extracted source declarations. Both original
and repaired `Kernels.lean` import only source definitions. The repaired source
includes scalar ABI, signed64 masks, modular byte pointers, all pointer-register
schedules, the finite-ceiling loop, output stores and the ordered whole grid.
Historical declaration names and mathematical helpers are retained.

`Contract` and `Proofs/Contract` prove checked metadata/precondition equivalence
and derive the launch/control and active byte-offset facts. The source envelopes
are explicit guards. Ordinary allocation laws contain identities, metadata,
initialized memory, aligned/addressable storage, freshness and frames; they do
not contain a source envelope, kernel result or successful kernel execution.

`Native` proves active pointer and mask correspondence, including three-iteration
lookahead. `Execution` projects complete finite pointer execution to the typed
source. `Ideal` proves the independent full-K real sum for arbitrary Real
allocation snapshots. `CheckedHost` threads the actual allocations, empty branch,
launch, returned object, full grid and caller frame. The public Real theorem has
no finite-encoding or floating-point exactness restriction.

The observed Triton 3.7.1 SM80 lowering fuses `accumulator += tl.dot(a,b)` into a
live-seed TF32 MMA recurrence. `Instructions` models that actual compiled
recurrence using fixed K8 groups, local TF32 operand conversions, FP16 narrowing,
and FP32 widening. Its independent reference is never identified with sequential
FMA or with a separately rounded source Dot+Add graph. The public native theorem
assumes exactly four local correspondences: TF32 conversion, fixed MMA8, half
narrowing, and widening. It assumes no variable-shape dot equality, associativity,
split/fused equality, or whole-kernel result.

`OutputBytes`, `ByteExecution`, and `ByteHost` execute the actual finite C pointer
and four-byte stores, and compose their representation through the entire grid.
The public result identifies the allocated C and proves every output word/byte,
address bounds, physical A/B allocation snapshot readback and input preservation,
plus all exterior bytes relative to caller memory. Fresh output storage has the
contiguous allocation view; logical output indices read back from its actual base.

The former broad public entries remain under `Proofs/HistoricalCorrect.lean`
and `Proofs/HistoricalFP.lean`. Their helper theorems remain in their original
scope; current acceptance uses the checked entries above. `Proofs/Numerical.lean`
is an auxiliary conditional exact-arithmetic domain, not a native numerical
premise or current public import.

## Validation

* `check_contract.py`: 17 preallocation rejection controls and 30 actual Triton
  CPU-interpreted calls (20 launches), all five schedules, tails, K=0 and empty
  output, original arithmetic/staging AST, exact returned identity, input values
  and exterior output storage sentinels. Exactly representable fixtures isolate
  addressing and casts; these tests do not validate TF32 GPU numerics.
* `Proofs/replay_candidate.py`: actual source AST with signed CPU lanes and
  sparse pointers: 120 small whole-host cases and 20 cases at old i32
  output/input/staged-pointer overflow boundaries without large physical allocations.
* `Proofs/replay_api.py`: 140 wrapper cases, 40 returns and 100 injected API
  failures, checking allocation order/identity, empty launch bypass, ordinary
  exception propagation and post-submission partial states.
* `Proofs/compile_candidate.py`: 15 offline SM80 compilations across all five
  schedules and three boundary shapes. Audits observe i64 pointer offsets,
  live-carry `m16n8k8` TF32 MMA, and both half narrowing and FP32 widening.
* The final dependency rebuild passed all 46 original/candidate modules with
  standard-base axiom checks and no errors. Independent native targets include
  noncircularity checks.

CUDA execution is unavailable. The four local numerical correspondences remain
explicit and unvalidated against a GPU here. Shared launcher compatibility and
historical package cleanup are separate work.
