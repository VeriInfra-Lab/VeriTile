# Exact installation plan

Root first completes the current four-task commit and independently reviews this package's delta. Then root supplies the full resulting commit SHA as `MIGRATION_BASE_COMMIT` and this package's frozen receipt SHA as `MIGRATION_SCRIPTS_SHA`. These values are intentionally not inferred or filled in here. `MIGRATION_RUN` must name a fresh directory outside the repository. The checkpoint below must also be absent.

```sh
python3 -B /tmp/veritile-standardization-20261015/queue-gzip-integration-scripts-wip/integration.py plan \
  --repo /mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile \
  --expected-base-commit "$MIGRATION_BASE_COMMIT" \
  --scripts-receipt-sha256 "$MIGRATION_SCRIPTS_SHA" \
  --catalog bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/current-source-catalog.json.gz \
  --checkpoint bench/review-checkpoints/2026-10-16/numerical-queue-gzip-migration \
  --output "$MIGRATION_RUN"
```

The read-only plan stage verifies a clean committed base, all candidate preimages and consumer pins, exact catalog/ledger/queue correspondence, current queue `--check`, and the baseline original-scope report. It rejects any previously accepted legacy row that no longer validates. It writes a concrete `plan.json` with rendered document hashes and prints its hash. Root reviews that exact plan and supplies `MIGRATION_PLAN_SHA` before running the following commands sequentially:

```sh
python3 -B /tmp/veritile-standardization-20261015/queue-gzip-integration-scripts-wip/integration.py install --plan "$MIGRATION_RUN/plan.json" --plan-sha256 "$MIGRATION_PLAN_SHA" --expected-base-commit "$MIGRATION_BASE_COMMIT"
python3 -B /tmp/veritile-standardization-20261015/queue-gzip-integration-scripts-wip/integration.py generate --plan "$MIGRATION_RUN/plan.json" --plan-sha256 "$MIGRATION_PLAN_SHA" --expected-base-commit "$MIGRATION_BASE_COMMIT"
python3 -B /tmp/veritile-standardization-20261015/queue-gzip-integration-scripts-wip/integration.py rebind --plan "$MIGRATION_RUN/plan.json" --plan-sha256 "$MIGRATION_PLAN_SHA" --expected-base-commit "$MIGRATION_BASE_COMMIT"
python3 -B /tmp/veritile-standardization-20261015/queue-gzip-integration-scripts-wip/integration.py snapshot --plan "$MIGRATION_RUN/plan.json" --plan-sha256 "$MIGRATION_PLAN_SHA" --expected-base-commit "$MIGRATION_BASE_COMMIT"
python3 -B /tmp/veritile-standardization-20261015/queue-gzip-integration-scripts-wip/integration.py check --plan "$MIGRATION_RUN/plan.json" --plan-sha256 "$MIGRATION_PLAN_SHA" --expected-base-commit "$MIGRATION_BASE_COMMIT"
```

Stages reject unrelated worktree changes. Each subprocess writes a distinct command record and log. A failed stage stops; inspect and preserve that attempt before preparing a repair. The scripts do not overwrite an existing archive or an existing command log, do not stage/commit, and do not automatically roll back over new work. The original queue survives in the real base Git blob and in a verified compressed archive. An interrupted installation can be repaired from those exact preimages, without discarding failure evidence.

Expected Git changes are the two Python consumers, two storage tests, two active progress documents, removal of the active plain queue, addition of its gzip replacement, regenerated queue Markdown, two queue-reference fields in the existing compressed legacy ledger, and a fresh migration checkpoint. No catalog, policy, source proof, source variant, task row, task review, numerical authority, primary checker or primary completion field changes.

Run the existing CI/global commands against the same current catalog and ledger after the storage checks. In particular retain `bench/deferred_numerical_experiments.py --check`, `bench/check_original_program_scope.py`, catalog reproduction, the full `bench/tests` suite, and the current global runner. Its baseline scope completeness result remains pending/failing for 175 legacy tasks. Build/comparator outcomes must be reported as actually run or not run; no prior result is rebound by this migration. Freeze the actual logs and inspect the Git diff before a separate root commit decision.

The final migration checkpoint contains:

- `before/active-queue.plain-json.gz`, with the original uncompressed SHA and size in the stage receipt.
- `before/scope-ledger.json.gz` and both old active documents under `before/bench/`.
- `plan.json`, the integration-scripts receipt, and exact subprocess records under `validation/`.
- `snapshot/{catalog,legacy,queue}.json.gz` and a compressed manifest containing real base/overlay origins, exact helper/source/authority pins, unchanged legacy row hash, policy and eight-task map references.

The snapshot is future-sidecar input only. The fixed-baseline prototype needs a separate source/API delta review before consuming it. Existing frozen prototype and review receipts must remain unchanged. No version completion number is synthesized from legacy counts or current proof inventory.
