# ksoftmax_triton

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

The backward JIT has a complete typed quotation, independent mathematical IO,
whole-state FP equivalence and successful execution witnesses. Complete forward
real IO and four-profile typed equivalence pass the
[complete root rebuild](../../review-checkpoints/2026-10-13/softmax-complete-root/receipt.json).
The forward declarations are now directly discoverable in the public entries.
The [4,899-job aggregate rebuild](../../review-checkpoints/2026-10-14/public-barrier-aggregate-root/receipt.json)
passes that migration and the integrated debug-barrier semantics. The whole Python task is not complete.

The isolated [forward execution checkpoint](../../review-checkpoints/2026-10-13/forward-pointer-full-witness/receipt.json)
now checks complete fp16 LOG/causal qk and bk runs with distinct mask values,
their finite outputs, written future lanes and untouched tails. The separate
[structural equivalence checkpoint](../../review-checkpoints/2026-10-13/forward-structural-equivalence-stage/receipt.json)
proves original/candidate state and failure equality for four declared storage
profiles and all flags/mask options, with empty assumptions. The
[complete forward contract checkpoint](../../review-checkpoints/2026-10-13/forward-complete-contracts-stage/receipt.json)
also proves original and candidate finite flat IO after real erasure for the
fp32/fp32/fp32 profile, positive depth, all flags and absent/qk/bk masks, with
independent formulas and the full unwritten-memory frame. Its 195 captured
inputs are preserved.
The pointer-provenance repair is integrated; its
[17 root regression files](../../review-checkpoints/2026-10-13/combined-frontend-compatibility-root/receipt.json)
pass. These targeted results are included in the completed aggregate replay.

The [typed-total checkpoint](../../review-checkpoints/2026-10-13/softmax-forward-typed-total/receipt.json)
proves a common successful final state for both complete forward variants,
all four profiles, every flag and absent/qk/bk masks, from arbitrary initial
state and algebra with `D > 0`. The public ordinary success theorem exposes
this result, and root replay with the updated shared semantics passes.

The forward real formula projects `WithBot` values with `unbotD 0`. In the
LOG/causal examples, written future lanes (`row < i < K`, with `i : Fin D`)
have typed bottom values and real observation zero; this does not assert a
native IEEE zero result. Unwritten tails retain their initial values. The real contract requires
`D > 0` and covers lanes `i : Fin D` with `i < K`, for arbitrary natural `K`.
The host relation between `K` and `D` is outside that contract. Arbitrary mask
options, including unknown strings, are covered by typed state/failure equality;
successful execution for unknown strings is not established.

The installed [forward source checker](scripts/forward_source_check.py) passes
all 216 storage triples and rejects 1,512 mutations against the original
Python forward body. The [canonical-source receipt](../../review-checkpoints/2026-10-13/forward-canonical-source/receipt.json)
binds the migrated source, checker and shared audit helper. This source matrix
is wider than the four compiled typed-equivalence profiles.

Public entries:

- [Kernels.lean](Kernels.lean) imports both complete JIT sources; its selected
  generic audit view remains `_softmax_backward`.
- [Correct.lean](Correct.lean) declares separate original and candidate backward
  real contracts and the complete forward contracts. These replace the
  historical plain-forward public entry and its unused proof chain.
- [FPEquiv.lean](FPEquiv.lean) declares backward FP equivalence and a common
  successful typed final state, plus forward state/failure equality for four
  checked storage profiles.
- [Proofs/BackwardMathSource.lean](Proofs/BackwardMathSource.lean) retains the
  unchanged mathematical backward quotation used by the dtype-erasure proofs.

The backward contracts retain independent gradient, saved-output and destination
storage, all six row strides, both program IDs and every `LOG`, `CAUSAL` and
`IS_FP16` branch. They use the original causal read mask and the distinct
`k < K` write mask. In the logarithmic causal branch, a future lane below `K`
receives the negative active-gradient sum; it is not forced to zero.

Real correctness explicitly erases dtype and specifies successful flat
execution, the independent finite-sum formula and every unwritten memory cell.
The candidate commutes only multiplication in the gradient input address;
its real contract uses a separately checked flat-execution equality.
FP equivalence preserves the complete typed state and failure result for
arbitrary floating algebras, without numerical or domain assumptions.

The backward formal family contains 216 independently chosen storage triples across
fp32, fp64, fp16, bf16, fp8e4nv and fp8e5. The selected source audit view is
fp32. The [source-profile receipt](../../review-checkpoints/2026-10-13/backward-source-profiles/receipt.json)
records all 216 complete original AST matches and 1,728 rejected mutations.
No branch or source statement is removed for that comparison.

The [frozen proof-chain receipt](../../review-checkpoints/2026-10-13/backward-full-real-and-typed/receipt.json)
records the proofs before the public file-layout migration. The
[public-layout rebuild](../../review-checkpoints/2026-10-13/backward-public-classifier-revalidation/receipt.json)
passes all 37 modules after the finite-classifier extension, with 209 unchanged
recursive local inputs. The exact-real
typed backward backend has 96 successful full runs: six homogeneous storage profiles,
eight flag combinations and two variants. Concrete fp16 checks include LOG
outputs −2 and −4, ordinary outputs 1/4 and 0, and an unwritten tail retaining
7. These are interpreter results from empty registers, nonzero program IDs,
distinct stride pairs and a partial tile.

Backward native acceptance is narrower. The [offline native receipt](../../review-checkpoints/2026-10-13/backward-softmax-native/receipt.json)
records 628 compiled and 500 rejected profiles out of 1,128. In its homogeneous
matrix, all tested fp16/fp32/fp64 configurations compile; bf16 LOG and both
fp8 formats are rejected. This is offline compilation evidence, not GPU
execution or a numerical-error guarantee.

Indices are modeled as naturals. Host launch/heuristics, fixed-width address
overflow, target instructions and inter-program races remain outside the
per-program contracts. Independent comparator replay is still pending.
