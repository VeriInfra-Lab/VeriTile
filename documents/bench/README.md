# Benchmarks and final proofs

- [Final selected corpus and validation commands](current_source_acceptance.md): 183 TritonBench-G tasks, their complete retained proof dependencies, worked examples and current regressions.
- [Final source and contract review](../../validation/contracts.json): mathematical formulas, complete FP program pairs, conditions and numerical dependencies.
- [Python source correspondence](BodySourceCorrespondence.md): explicit source bindings and remaining translation gaps.
- [Imported benchmark audit](IMPORTED_KERNEL_AUDIT.md): 65 bug-detection and 269 pretraining challenges, with their proof/backend obligations still unsolved.

The shared framework is in `VeriTile/`. Each selected task identifies its actual programs, independent Real specification and FP equivalence statement. Compilation, source correspondence, proof replay and numerical/native evidence remain distinct checks.

## Current entry points

| Purpose | Entry |
| --- | --- |
| Select each task's source revision and public files | `validation/task_progress.json` |
| Read the mathematical targets, FP pairs and conditions | `validation/contracts.json` |
| Check source/review/correspondence bindings | `bash scripts/bench/audit_tritonbench_g.sh --source-only` |
| Check sources and build/replay the library | `bash scripts/bench/audit_tritonbench_g.sh --global-only` |
| Build and replay the complete retained corpus | `bash scripts/bench/audit_tritonbench_g.sh` |
| Export current public specification cards | `python3 scripts/spec_sheet.py` → `Logs/spec-sheets/` |

The shared Lean lexer is `scripts/bench/lean_import_scan.py`; declaration extraction is
maintained in `scripts/spec_sheet.py`. Source adapters and the final
inventory use those same implementations. `scripts/bench/port_layout.py` supports the
legacy filename-based tools; it does not select the current task revisions.
`scripts/bench/check_ports.sh` without arguments forwards to the current full-corpus trust
gate; explicit task names retain its legacy filename-based interface.

Dated task audit notes retain their original scope and receipts. Current
selection comes from the inventories above. The generated cards formerly
under `bench/tritonbench_g/_spec-sheets/` are available through the [historical index](spec-sheets/README.md).

`bench/` contains exactly `examples/`, `bug_detection/`, `pretrain_kernels/`
and `tritonbench_g/`. Tools are in `scripts/bench/`, tests in `tests/`, shared
records in `validation/`, and this documentation in `documents/bench/`.
Each TritonBench-G task keeps its single current repair at the task root.
