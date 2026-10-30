# Completion audit

The integrated port contains 96 Lean modules: the earlier 76-module helper
checkpoint, three public/source entries, ten native-integer helpers, five hint
helpers and two additional axiom-audit modules. The pinned upstream Python and
public source/correctness/FP statements are unchanged by the native/hint addition.

Checked: complete typed source and empty-assumption structural FP equivalence;
complete original/candidate real-erasure execution and trace safety; symbolic
positive-K finite-flat IO with independent GEMM, original inputs and full raw
memory frame; standard `ComputeCorrect.RealizesFrame_without_Rounding` with
exact `MemCell` outputs, all assigned active stores and their complete raw
frame; actual host-grid output coverage and cross-PID output-window
nonaliasing; K=0 standard empty-write-map frame and successful finite-flat
complete memory preservation; a nonempty finite witness
with tails, stride gaps, multiple persistent tiles and 18/27/36 readbacks.

Critical helper receipts contain 116 explicit symbolic standard-axiom checks,
plus the earlier 12 baseline checks. Public theorem compilation and three
entry-point axiom checks are recorded separately in the integration receipt.
The two regenerated public spec sheets report no self-reference or missing
summary. The full positive-K `real_correct` is the final public correctness
declaration, and its original successful finite-flat contract is retained.
Standard Lean axioms only; no sorry, new axiom or native_decide proof is used.

The native-integer checkpoint passes 76 axiom checks, including three axiom-free
results. It proves a handwritten source-mapped signed-i32 control/coordinate/
address projection under explicit arithmetic bounds. Nonempty witnesses cover
both recorded configurations, multiple persistent tiles and K tails.

The hint checkpoint passes 43 axiom checks, including two axiom-free results.
Independent fresh local Lean replay reproduces all five module artifacts and
the complete axiom log with no diagnostics. For positive K, the full reached
cached-coordinate promises hold iff BM divides M and BN divides N; any nonzero
M/N remainder yields a reached contiguity counterexample. K=0 reaches no hint.
The installed binary/DWARF evidence and release-tag C++ comparison do not prove
source-to-binary identity, compiler soundness or a specific wrong GPU result.

The preceding 175-port snapshot passes the complete source-only audit and 206
tool regressions. The new helpers require an updated integrated audit and
aggregate build; their local receipts are not full-corpus acceptance.

Remaining gates:

- Full typed-source/TTIR refinement connecting the handwritten fixed-width
  projection to native execution and memory effects.
- Legal native Triton tile/dtype/launch configuration admission for symbolic
  parameters; the two recorded concrete offline compiles have narrower scope.
- Updated full integrated source audit after the native/hint helper addition;
  both recorded pointer-dtype profiles already pass their scoped comparisons.
- Official independent comparator replay and final aggregate acceptance.

No full original-Triton-port acceptance, native IEEE/GPU numerical correctness,
or concurrent launch interleaving claim follows from these receipts.
