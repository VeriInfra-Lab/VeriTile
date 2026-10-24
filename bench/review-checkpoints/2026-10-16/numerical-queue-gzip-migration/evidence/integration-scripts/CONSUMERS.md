# Consumer audit and proposed changes

| Consumer | Candidate treatment | Reason |
| --- | --- | --- |
| `bench/deferred_numerical_experiments.py` | Install reviewed gzip writer/checker; also reject a dangling plain symlink | Exactly one active representation; full-stream decompressed text and Markdown checks; collect/authority AST unchanged |
| `bench/check_original_program_scope.py` | Change QUEUE to `.json.gz`; factor `load_active_queue` with plain-path rejection | Standalone scope checks must reject ambiguous active storage, independently of later CI gates; acceptance predicates unchanged |
| `bench/tests/test_numerical_queue_storage.py` | Install seven reviewed tests with explicit local bench import path | Tests work independently of test-discovery ordering |
| `bench/tests/test_numerical_queue_migration.py` | Add seven scope-loader regressions | Covers dual paths, old-only path, dangling plain name, wrong reference/hash, corrupt gzip and duplicate JSON |
| `bench/original_program_scope.json.gz` | Only `/deferred_queue/path` and `/deferred_queue/sha256` change | All bytes outside this block, all task rows, task reviews and legacy statuses remain identical |
| Active queue Markdown | Regenerate from unchanged current authority | Its link must name the active compressed file; no rule/task content changes |
| `bench/current_source_acceptance.md`, `bench/original_program_scope.md` | State approved three-version goal, label 8/175 as legacy, link policy/map and gzip, preserve old summaries | Progress language must match the user's Real/FP32/BF16 plus required-types goal without inventing completion counts |
| `bench/adapter_common.py` | No change | Existing deterministic gzip writer and EOF/CRC/duplicate-key-validating reader are already reviewed |
| `bench/exact_numerical_bindings.py`, `bench/exact_scope_gate.py` and reviewed extensions | No change | Source-derived numerical authority, callbacks and task rules remain identical. The explicit temporary CLI export `--output DIR` remains plain JSON and is not the active queue; do not use the active task directory as its output |
| `bench/current_source_audit.py` and `.github/workflows/bench-audit.yml` | No source change; pin and verify existing call chain | CI invokes the current global runner; the runner always invokes producer `--check`, and the scope checker now independently checks storage ambiguity |
| `current_source_catalog.py`, `current_value_metadata.py`, catalog/supplement | No change | No proof/source/task inventory changes are part of storage migration |
| Historical checkpoints and receipts | No rewrite | Old `.json` path/hash references refer to their frozen historical evidence, not the active queue |
| Fixed-97e11a9 primary-version prototype and its inventory/exporter | No install or rewrite | They hardcode a different commit and old paths. A fresh coherent base-plus-overlay input snapshot is produced for a later separately reviewed sidecar refresh |

The approved policy is pinned at `bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md`, SHA `10a59ed41d81aee0e5bb82b81a998116564a8dd7c378516a0a4edf894b07be85`. The eight-task map is pinned at that checkpoint's `evidence/three-version-inventory/eight-task-mapping.md`, SHA `c944d1962f2ee7a6c7bc828d73af96e3be2732f66dc75a8c71d8843f777b423d`. Both become part of the explicit future plan and snapshot without changing their contents.

Live consumer searches excluded historical checkpoint payloads deliberately: their old paths remain valid provenance. The active path references found were the producer, scope checker, two progress documents, generated queue Markdown, and the temporary exact-authority CLI export. Source acceptance documents that link to the stable queue Markdown require no path change.
