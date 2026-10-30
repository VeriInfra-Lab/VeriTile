# KLDiv Triton Full-Source Proof Plan

> This port is integrated under the active corpus-completion plan. Root owns
> its implementation and registry updates; parallel agents own other ports.
> Keep dependency builds coordinated when shared artifacts are being refreshed.

**Goal:** Prove independent real IO/safety/frame and whole-kernel FP equivalence
for the actual complete forward and backward kernels, preserving historical APIs.

**Architecture:** Three public Lean entries share faithful typed source. Local
proof modules separate independent formulas, source projection, signed loop
invariants, actual stores/frames, precision regressions and numerical execution.
The reduction contract describes one scalar per executed chunk, not a row sum.

**Tech Stack:** Lean 4, mathlib, VeriTile Triton DSL and real/structural interpreters.

## Approved source scope

Forward floating source profiles independently select six prediction and output
carriers, and fp32/fp64 target storage for direct log/exp. Both log flags and every
Nat reduction mode retain the complete source loop. Backward retains its two
pre-loop assignments, six target/output carriers in the default multiplication
branch (fp8 targets are an abstract extension, excluded from source acceptance
because nonzero weak scalar construction is unavailable), and fp32/fp64 target
with six outputs in the exp branch. Runtime offsets
use signed Int registers; natural row strides and positive B delimit the initial
mathematical address domain. Signed negative strides and native fixed-width
arithmetic are separate obligations, never silently reinterpreted as proved.

Zero-row forward reduction performs no store. For N>0 the nonzero reduction
branch stores ceil(N/B) scalars at row*loss_stride + chunk. The Python wrapper's
one-scalar-per-row allocation does not satisfy general multi-block bounds.

## Steps

- [x] Preserve a legacy snapshot and audit every declaration header/docstring (51, no mismatches).
- [x] Split historical source, helpers and specifications without changing them.
- [x] Add complete typed forward and backward sources plus exact address-commuting candidates.
- [x] Check exact load, mixed-arithmetic, reduction, signed-loop and store ASTs.
- [x] Prove independent pointwise, padded-block and backward mathematical formulas.
- [x] Prove actual initialization and body execution, then arbitrary-length signed-loop invariants.
- [x] Prove actual active bounds, trace safety, flat execution/readback and full frames for the canonical mathematical source. Typed projection and candidate execution/safety transport gates also pass.
- [x] Prove full FP state/failure equivalence without numerical rewrite assumptions. All three source-facing declarations compile and print no numerical assumptions.
- [x] Check nonempty multi-block execution witnesses, reduction chunk footprints and empty-row behavior. Full typed numerical executions and active/sentinel readbacks pass at the default native stack using explicit two-block composition.
- [x] Gate every leaf, public entry and axiom audit; preserve all historical headers/docs. The current public and support chain passes, including all typed numerical regressions.
- [x] Remove the redundant historical top-level module after Kernels, LegacyReal and Correct compile and all 51 historical declaration headers/docstrings match.
- [x] Complete root source/contract review and migrate all eleven public coverage records with validated source/support hashes; report unsupported source profiles and the wrapper allocation counterexample.
- [ ] Complete corpus manifest integration and independent comparator replay when its required environment is available.

No proof may assume successful execution or desired output values. Real log is
totalized; real padding simplifications do not claim IEEE behavior. FP proofs
retain arithmetic on inactive lanes and arbitrary backend behavior.
