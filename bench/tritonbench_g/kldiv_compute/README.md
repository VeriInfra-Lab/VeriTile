# kldiv_compute

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

- Source file: `kldiv_compute.py`
- Corpus: TritonBench-G v1
- Status: source, precision, real IO/frame, safety, full FP equivalence and execution witnesses pass. The final dependency-aware build passes 3538 jobs; public axiom checks use only standard Lean axioms, with no FP assumptions. Three contracts are registered as `test-gap`; independent replay remains pending. See [source_scope.md](source_scope.md) for modeling boundaries.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.
