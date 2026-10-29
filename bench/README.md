# Benchmarks and final proofs

- [Final selected corpus and validation commands](current_source_acceptance.md): 183 TritonBench-G tasks, their complete retained proof dependencies, worked examples and current regressions.
- [Final source and contract review](validation/contracts.json): mathematical formulas, complete FP program pairs, conditions and numerical dependencies.
- [Python source correspondence](BodySourceCorrespondence.md): explicit source bindings and remaining translation gaps.
- [Imported benchmark audit](IMPORTED_KERNEL_AUDIT.md): 65 bug-detection and 269 pretraining challenges, with their proof/backend obligations still unsolved.

The shared framework is in `../VeriTile`. Each selected task identifies its actual programs, independent Real specification and FP equivalence statement. Compilation, source correspondence, proof replay and numerical/native evidence remain distinct checks.
