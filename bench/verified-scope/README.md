# Selected main release scope

This scope follows the maintainer's 2026-10-30 decision: publish the already
verified set and leave the remainder explicitly pending.

- 6,169 root modules, with 7,621 local Lean source dependencies.
- All public entries of 124 selected tasks are in that dependency closure.
- 59 other task entry sets remain pending for this release.
- Python/Lean body correspondence is checked for 124/183 tasks and 19/19 examples.
- The separate historical original-program ledger still has 172 pending reviews.

`roots.txt` contains the previous 6,166-root verified set plus three public files
that are now explicitly rebuilt: DequantizeRowwise Contract and SwiGLU
Correct/FPEquiv. `manifest.json.gz` pins the complete local source closure,
build configuration, root list, task dispositions and Python test inventory.
A task cannot disappear or change disposition without updating that reviewed
scope. The normal body/formula/IO/reader guards remain separate required checks.

```sh
python3 bench/verified_scope.py               # source and disposition checks
python3 bench/verified_scope.py --json        # website/readable inventory
python3 bench/verified_scope.py --regressions # selected tests; pending test named
python3 bench/verified_scope.py --build       # build, joint import, axiom audit
```

The proof audit enumerates actual theorem objects from every selected module,
including private helpers, and traverses their dependencies in Lean's checked
environment. Only `propext`, `Classical.choice` and `Quot.sound` are permitted.
The proof's own domain and numerical assumptions remain part of its statement.
This does not certify Triton compilation, native GPU execution or all historical
contracts. Independent official comparator checks remain in their named shared
interface and integration workflows; a green interface check is not reported
as comparator replay of the entire selected corpus.

All benchmark Python test files have an explicit disposition. The historical
`test_exact_numerical_bindings.py` is pending because its receipt pins an older
Structural.lean. The test and its assertions remain unchanged in the manually
run historical Bench audit. All other current benchmark test files are in the
selected regression suite, along with archive-recovery safety tests.

The wider Artifact and Bench audit workflows remain manually runnable. Their
failures and pending classifications are retained in
[the integration record](../main-integration/README.md), with complete historical
evidence in a fixed archive. They are not silently relabeled as passing release
checks.
