# Local completion audit

The local proof migration is complete for the 15 homogeneous storage profiles
and the explicit address/placement scope described in `source_scope.md`.
Repository integration and independent comparator replay are separate gates.

- Actual source: all three two-iteration loops, cached loads, pointer updates,
  and 24 stores remain in `Kernels.lean`. Exact load/store AST checks cover every
  profile and both versions; lane enumeration is pinned to the source model.
- Independent mathematics: `CopyMath.expected` folds the exact ordered write
  schedule. `PayloadReal.CopyInvariant` connects every actual intermediate state
  to its prefix; it does not assume kernel output or successful execution.
- New real contracts: four original/candidate region/flat headlines have complete
  successful execution, ordered oracle, raw memory frame, native lane injectivity,
  and actual load/store access safety. Floating `storedCell` normalization,
  nonnegative strides, input/output separation, allocation bounds, and the flat
  bridge's zero-undef requirement are explicit.
- Typed FP: full optional state equality holds for all 15 profiles, arbitrary
  backend states and aliases, with an empty extra-assumption list. The candidate
  commutes only an initial signed address multiplication.
- Nonvacuity: all 15 profiles have a successful typed execution theorem. Actual
  int32 and fp64 executions in a nonempty WithBot Real backend return cells
  8=2, 47=27, 55=35, and retain cell 56=-1, for both original and candidate.
  `FiniteWitness.source_contract` separately instantiates every complete real
  flat contract in disjoint 64-cell source/destination allocations, with all
  bounds, tile injectivity and access safety proved. Its int32 and erased-fp64
  executions read back the same four cells and retain the complete raw frame.
- Legacy: all 52 historical declaration headers/bodies and 37 declaration
  docstrings are unchanged across `Proofs/Legacy.lean` and `Correct.lean`.
  Their three public headlines retain their original partial/copy-slice scope.
- Trust: the current-head dependency-aware rebuild passes all 59 existing
  modules; the final public build also passes the new finite witness. Both
  public entries reach all 60 modules. The final 25-declaration audit includes
  all eight public heads, all-profile source checks, typed/real executions,
  numeric readbacks and finite witnesses. All axiom footprints stay within
  propext, Classical.choice and Quot.sound; `source_equivalence` prints no FP
  assumptions. The independent oracle passes its non-circularity check.

Current local evidence, refreshed on 2026-10-13:

- `/tmp/nested_current_head_build.log`: all 59 existing modules, including
  `RealBoth`, both public entries and every payload profile; exit 0, 3491 jobs.
- `/tmp/nested_finite_public_build.log`: final public closure with the finite
  witnesses, all 60 modules reachable; exit 0, 3492 jobs.
- `/tmp/nested_current_public_audit.log`: all 25 declaration checks, empty FP
  assumptions and independent-oracle check; exit 0.
- `/tmp/nested_source_oracle_review.log`: independent execution of the isolated
  Python `nested3` body with a host load/store stub agrees with the ordered
  oracle for strides (8,1), (1,8), (7,2) and (9,3), including all 14 loads,
  24 stores, exact access maxima and untouched cells. This is not Triton
  compilation, GPU execution or comparator replay.

The source review confirms all three loops, cached outer/middle loads and
pointer increments. Correctness retains the complete destination oracle and
all outside-footprint raw cells; no injectivity between separate stores or
successful execution is assumed. The only change to `Correct.lean` is its new
finite-witness import; its existing declaration text and `Proofs/Legacy.lean`
are unchanged from the current branch checkpoint. `git diff --check` passes.

Remaining coverage boundaries are explicit, not discharged by these checks:
native lowering and fixed-width overflow, mixed/discrete conversion families,
negative pointer increments, host grid coverage, and cross-program race freedom.
The wrapper's unsafe 4×4 and empty-launch 2×2 cases are not certified witnesses.
