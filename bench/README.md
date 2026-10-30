# Benchmarks and final proofs

- [Final selected corpus and validation commands](current_source_acceptance.md): 183 TritonBench-G tasks, their complete retained proof dependencies, worked examples and current regressions.
- [Final source and contract review](validation/contracts.json): mathematical formulas, complete FP program pairs, conditions and numerical dependencies.
- [Python source correspondence](BodySourceCorrespondence.md): explicit source bindings and remaining translation gaps.
- [Imported benchmark audit](IMPORTED_KERNEL_AUDIT.md): 65 bug-detection and 269 pretraining challenges, with their proof/backend obligations still unsolved.

The shared framework is in `../VeriTile`. Each selected task identifies its actual programs, independent Real specification and FP equivalence statement. Compilation, source correspondence, proof replay and numerical/native evidence remain distinct checks.

## Current entry points

| Purpose | Entry |
| --- | --- |
| Select each task's source revision and public files | `tritonbench_g/task_progress.json` |
| Read the mathematical targets, FP pairs and conditions | `validation/contracts.json` |
| Check source/review/correspondence bindings | `bash bench/audit_tritonbench_g.sh --source-only` |
| Check sources and build/replay the library | `bash bench/audit_tritonbench_g.sh --global-only` |
| Build and replay the complete retained corpus | `bash bench/audit_tritonbench_g.sh` |
| Export current public specification cards | `python3 scripts/spec_sheet.py` → `Logs/spec-sheets/` |

The shared Lean lexer is `lean_import_scan.py`; declaration extraction is
maintained in `../scripts/spec_sheet.py`. Source adapters and the final
inventory use those same implementations. `port_layout.py` supports the
legacy filename-based tools; it does not select the current task revisions.
`check_ports.sh` without arguments forwards to the current full-corpus trust
gate; explicit task names retain its legacy filename-based interface.

Dated task audit notes retain their original scope and receipts. Current
selection comes from the inventories above. The generated cards formerly
under `tritonbench_g/_spec-sheets/` are available through its historical index.
