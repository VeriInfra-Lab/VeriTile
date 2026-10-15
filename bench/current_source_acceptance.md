# Current source inventory and acceptance

Current installed scope: **184 tasks, 893 public declarations, 3,840 registered modules**; **58 software bit-value profiles, 125 backend-dependent profiles and one repaired candidate**. The original 41 remains **35 software / 6 backend-dependent**. See the [DQ/helper source extension](review-checkpoints/2026-10-15/retention-query-source-proof/README.md), the [preceding bit-value review](review-checkpoints/2026-10-15/raw-copy-software-metadata/README.md), and its [184-task table](review-checkpoints/2026-10-15/raw-copy-software-metadata/tasks-184.tsv) and [remaining six](review-checkpoints/2026-10-15/raw-copy-software-metadata/remaining-original41.tsv).

The Retention extension adds the complete fp32 DQ helper and sequential DQ/barrier/DK/DV contracts after caller index specialization, with independent input-memory references and complete memory frames. Native numerical bindings remain pending in the [unified experiment queue](tritonbench_g/deferred_numerical_experiments.md). The affected Retention contracts compiled successfully; all 20 new proof declarations passed named standard-axiom checks.

The preceding [destindex_copy](review-checkpoints/2026-10-15/destindex-copy-raw-bit-closure/README.md) and [matrix_transpose](review-checkpoints/2026-10-15/matrix-transpose-raw-bit-closure/README.md) contracts cover their complete original typed domains, independent initial-memory references and full memory frames. No finite-input, noalias, positive-size or known-output premise was added. The former covers nine independent same-format storage pairs; the latter preserves every fp16 bit pattern in both existing source versions.

From the repository root, use fresh output directories:

```sh
python3 -B bench/current_source_catalog.py prepare-baseline \
  --package bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/metadata-baseline \
  --output /tmp/current-raw-copy-metadata-baseline
python3 -B bench/current_source_catalog.py --root . \
  --baseline /tmp/current-raw-copy-metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/retention-query-source-proof/catalog-supplement.json \
  --check bench/review-checkpoints/2026-10-15/retention-query-source-proof/current-source-catalog.json
python3 -B bench/fp_value_extensions.py --check
python3 -B bench/deferred_numerical_experiments.py --check
python3 -B bench/legacy_source_adapter.py --root . --tools-root . \
  --catalog bench/review-checkpoints/2026-10-15/retention-query-source-proof/current-source-catalog.json \
  --output /tmp/current-raw-copy-legacy-source-replay
```

The preceding raw-copy package preserved all 191 Python inputs, its old 889 public records, 37 extension rows and 60 evidence packages. Existing recursive Lake registration adds eight modules without changing configuration. Root independently replayed 52 metadata tests and generation. The generator binds 17 installation targets and 4,187 inputs, including postguards for all 24 historical task files. Build evidence combines 3,826 revalidated records with eight independently compiled modules and 44 named standard-axiom checks; it is not a new whole-library compilation.

There are now 146 new public entries: 142 value contracts, two frames, one identity and one protocol. The new six-module source extension preserves the preceding 891 public statements; its separate receipt does not claim a fresh full-corpus build. The original 41 classification and its six unresolved task states are unchanged. The remaining Llama row now describes the existing ∀FloatDType contract; fp32 is only the default experimental profile.

The [previous dequant acceptance](review-checkpoints/2026-10-15/dequant-int4-source-compiled-software-metadata/README.md) and [task explanation](tritonbench_g/matmul_dequantize_int4/SOFTWARE-ACCEPTANCE.md) remain unchanged. Its [capture tool](review-checkpoints/2026-10-15/dequant-int4-native-characterization/README.md) passed independent CPU/offline checks; native execution is NOT_RUN. Native byte/address ABI, device instruction behavior, allocations and cross-program/alias scheduling are separate obligations.

[Official comparator library results](review-checkpoints/2026-10-15/comparator-canonical-library/README.md), [samples](review-checkpoints/2026-10-15/comparator-initial-samples/README.md) and [storage-maintenance snapshot](review-checkpoints/2026-10-15/comparator-storage-maintenance/README.md) retain their exact scopes. The full replay targets frozen F3's **365 files / 872 headlines**; the later **21 declarations**, including the two Retention contracts, are separate. No live-progress assertion or full-corpus success is inferred here.

The original scaled-FP4 source gap and native functional correctness remain open. Historical 129 “no extra numerical law” and 43 experiment records do not imply GPU correctness. See the [experiment handoff](review-checkpoints/2026-10-14/next-value-repair/native-experiment-handoff/EXPERIMENT-HANDOFF.md) and [daily evidence index](review-checkpoints/2026-10-15/README.md).
