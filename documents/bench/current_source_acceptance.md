# Final selected corpus

`validation/task_progress.json` selects the final source and public entries for all 183 tasks. `validation/contracts.json` records their independent mathematical targets, complete FP program pairs, required conditions and numerical dependencies. `validation/source.json` binds the entire retained Lean tree and its project import closure.

Run the final checks from the repository root:

```sh
python3 -B -m unittest discover -s tests/python -p 'test_*.py'
python3 -B scripts/bench/final_inventory.py --check
python3 -B -m scripts.bench.check_body_corpus --check
python3 -B scripts/bench/check_imported_kernel_review.py
lake build VeriTile VeriTileMath VeriTileFull TritonBenchGFinal FinalRegressions TritonBenchSpecExamples BugDetection PretrainKernels
python3 scripts/check_comparator.py --library
scripts/bench/audit_trust.sh --final-inventory
```

The proof comparator requires the pinned tools and their systemd/Landlock isolation; `.github/actions/setup-comparator` installs them in CI. Every retained corpus module must pass the axiom/specification audit and independent proof replay. The complete gate fails if any worker is missing, skipped, cancelled or fails.

PR and main-push CI run the source/regression and library checks. Complete
corpus builds, all 334 draft interfaces and per-module replay run daily at
04:00 Asia/Singapore (20:00 UTC), or on manual dispatch with `full_audit=true`.
Run the complete applicable proof checks locally before merging proof changes.

Source correspondence is tracked independently in `validation/source-correspondence.json`. Pending Python-to-Lean correspondence remains pending. The 65 bug-detection and 269 pretraining benchmark tasks are explicit unsolved challenges; their successful compilation does not complete their proof or backend obligations. Native execution and compiler conformance are not established by these checks.

The repository contains the final source tree and current validation records. Superseded source versions, review checkpoints and archived proof runs are not required by this workflow.
