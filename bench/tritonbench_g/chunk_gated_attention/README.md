# chunk_gated_attention

Source: `chunk_gated_attention.py`, TritonBench-G v1. The current complete
real entrypoint passed the [branch integration build](../../review-checkpoints/2026-10-14/acceptance-branch-real-root/README.md).
Current aggregate results are recorded in the [checkpoint index](../../review-checkpoints/2026-10-14/README.md).
Historical independent replay does not certify the current source; official
replay is blocked on this host by unavailable systemd/Landlock isolation.
Host grid composition and fixed-width address overflow remain outside the statements.

The source retains its actual branch-local carry update. The real proof
establishes that factoring this continuation preserves execution and memory
safety. The FP domain now locates the finite carry/dot guard inside the
selected gate arm, after that arm's actual prefix. This corrects the old
guard placement; it is not a claim that the old and new domains are identical.
Every gate/storage/seed/final selection remains in the public statement.
The FP proof uses `add_commute`; no unconditional IEEE equality is claimed.

- [Kernels.lean](Kernels.lean): complete original/candidate Triton sources.
- [Correct.lean](Correct.lean): both versions' IO, independent formulas and
  necessary allocation, separation and unspecified-value conditions.
- [FPEquiv.lean](FPEquiv.lean): complete conditional FP equivalences and the
  actual atomic assumptions.
- [Proofs/LegacyReal.lean](Proofs/LegacyReal.lean): historical specialized and
  prepared-input statements, excluded from the current public inventory.

| Kernel | Candidate | Current proof |
|---|---|---|
| `chunk_gated_abc_fwd_kernel_cum` | Reverse the exact natural-number head/stride multiplication in both pointer bases | Both full sources compute the independent chunk-local prefix sum, with successful region/flat execution, active-address bounds and full tagged-cell frame. Input/output aliasing is permitted. Valid outputs are independent of unspecified padding. |
| `chunk_gated_abc_fwd_kernel_h` | Reverse only the final fp32 carry/dot addition in each iteration | Both full sources compute the independent real closed form under an explicit zero-unspecified-value premise, with every iteration, both gate branches, seed/final switches, pre-update history and tagged-cell frame. Both finite-allocation flat execution and complete FP equivalence are proved. |

[Proofs/FP.lean](Proofs/FP.lean) proves both complete program equivalences. The
cumulative candidate uses **no FP atom**. The recurrent candidate uses only
**`add_commute`**, with finite operands checked after the actual gate branch at
every reached update. Both preserve execution failure as well as successful
state. Intrinsic `tl.exp` remains unchanged; no exponential rewrite is used.

The source retains K/V storage, gate/history storage and independent seed/final
storage, fp32 accumulation, pre-dot storage casts, `allow_tf32=False`, the
optional seed **addition**, pre-update history stores and the optional final
store. Endpoint loads use the original full chunk-end index in a flattened
`T*K` or `T*V` parent. They are not clipped to the last valid time. Omitted
padding stays unspecified.

The cumulative proof is in `Proofs/RealCum{Source,Program,Execution,Safety,Flat,IO}.lean`.
`bench/tests/GatedCumulativeReal.lean` exercises both complete versions with
nonzero program IDs, time/feature tails, input/output aliasing, arbitrary real
padding and a neighboring non-real tagged cell. The public cumulative IO uses
the shared adapter's explicit zero-oracle condition, although the lower-level
region and flat theorems do not need it for valid prefixes.

`Proofs/TypedCumExecution.lean`, `Proofs/TypedExecutionSteps.lean` and
`Proofs/TypedExecution.lean` supply
successful mixed-storage executions of both candidates. The recurrent witness
covers two chunks, partial time/K/V tiles and all eight gate/seed/final flag
combinations. These explicitly named abstract backends establish executable,
nonempty contracts and typed memory flow; their dot/exp values are not GPU
measurements or the independent real formulas.

[Proofs/RealHExecution.lean](Proofs/RealHExecution.lean) connects the complete H
sources to the independent closed form in [Proofs/RealHMath.lean](Proofs/RealHMath.lean).
The proof preserves arbitrary shapes, strides and program IDs. Its history
stores must be disjoint from K/V/gate inputs and have noncolliding active
addresses; the enabled final store must preserve history. Seed storage may
alias outputs because it is read before any output store. Unspecified lane
values are explicitly zero for this formula. Pointer/load/gate/step lemmas
retain arbitrary unspecified values before that premise is applied.

[Proofs/UndefinedTail.lean](Proofs/UndefinedTail.lean) proves that removing the
zero-tail premise changes the claim: with `T=1` and 16-wide tiles, zero valid
inputs and unspecified K/V tails equal to one, both full H sources store 15
instead of the formula's zero. This is a real-model counterexample, not a GPU
measurement. It applies to both gate branches.

`bench/tests/GatedAttentionReal.lean` checks both full sources and all switches
with nonzero program IDs, two chunks, partial time/K/V tiles, nonzero gates,
pre-update history, disabled final stores and an untouched natural-valued
neighbor. The finite-allocation regression instantiates both public real
specifications in six disjoint regions and checks the same outputs and frame.
The final chunk uses its checked-off full endpoint, rather than the
last valid input time.

[Proofs/RealHSafety.lean](Proofs/RealHSafety.lean) proves allocation safety for
the full source, using only enabled and checked-on accesses. The endpoint vector
keeps its one-dimensional parent bounds. [Proofs/RealHFlat.lean](Proofs/RealHFlat.lean)
and [Proofs/RealHIO.lean](Proofs/RealHIO.lean) transport execution, formulas and
the full tagged-cell frame into finite disjoint allocations. No successful
execution or expected output is supplied as a premise.

See [the review record](../verification_round.md) for validation receipts and
the current corpus count.
