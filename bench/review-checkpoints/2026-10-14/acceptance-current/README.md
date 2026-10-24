# Current full-corpus review bundle — not accepted

All 184 tasks and 3,931 current input files are captured. No input changed
during collection. Membership, source audit, inventory freshness and contract
metadata checks passed. All 67 legacy modules are registered.

The complete eight-library build failed at embedding_triton_kernel.Kernels:
int32 loads reject the untyped dynamic `other` argument. After preserving that
module's complete diagnostics, the runner terminated remaining jobs; the build
exit is -15. This is an incomplete failed build, not proof that every other
module passes or fails. Earlier known failures have separate snapshots.

Both actual comparator isolation probes failed: the user-systemd bus is absent
and this kernel exposes Landlock ABI 0 while the installed runner requires V5.
Library and corpus replay were not run. Six tasks still lack source pairs.
The complete gate receipt therefore has mechanical_gates_passed=false.

The separate check verified every recorded artifact and input hash, exact
archive membership, and equality with the current working-tree input set:
integrity_errors=[]. Its failure exit is expected because the acceptance gates
failed. See ../acceptance-current-check.json and ../ACCEPTANCE.zh-CN.md.

- receipt.json: commands, exit statuses, source/artifact hashes and missing tasks.
- inputs.zip: exact full input tree; dependencies remain pinned by lake-manifest.json.
- tasks.json and task-scopes.md: all 184 source/contract records.
- source.log, inventory.log, contracts.log, membership.log: passed checks.
- build.log and comparator_preflight.log: actual failed diagnostics.

Recheck with:

    python3 bench/acceptance_tritonbench_g.py --check bench/review-checkpoints/2026-10-14/acceptance-current

Collect a new bundle in a new directory with --output after fixing the remaining
code/scope gaps and using a host that passes both official isolation probes.
