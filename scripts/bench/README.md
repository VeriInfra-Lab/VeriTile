# Benchmark tools

Run from the repository root:

```sh
bash scripts/bench/audit_tritonbench_g.sh --source-only
python3 -B -m scripts.bench.final_inventory --check
python3 -B -m scripts.bench.check_body_corpus --check
python3 -B -m unittest discover -s tests/python -p 'test_*.py'
```

`final_inventory` selects current repaired task entries, enforces the four
benchmark collections, checks imports and binds reviewed source files.
`check_body_corpus` separately preserves Python/Lean correspondence gaps.
`check_imported_kernel_review` guards the 334 imported challenge findings.

## Paste a Triton body without writing a JSON manifest

The `rms_matmul_rbe` layout pilot keeps one Python module for the repaired
source and its comparison candidate. The Lean entry points are `Kernels.lean`,
`RealCorrect.lean`, `FPEquiv.lean`, `IO.lean` and `Contract.lean`. Supporting
modules live in `Proofs/`. Existing split ports may still use `Correct.lean`
as their Real entry.

Keep the repaired Python reference and paste its function body into the Lean
`triton { ... }` source entry. The existing input/IO call declaration supplies
pointer and runtime scalar types outside the pasted body. No `tl.region`,
`tl.param`, cast deletion, or extra statements are needed in that body.

From the repository root, check the directory directly:

```sh
python3 -m scripts.bench.generate_jit_correspondence bench/tritonbench_g/rms_matmul_rbe
```

The tool discovers complete Python JIT functions and Lean source quotations,
matches their bodies, extracts parameter/constexpr/call declarations and helper
links, and runs the existing correspondence checker. It never imports or
executes Triton, Torch, or the source module. It does not read the task's
`jit-correspondence.json` during discovery and does not write one by default.
The corpus source gate also discovers pairs automatically when that file is
absent. Shared example references use the existing collection-level
`source-references.json`; `--python path/to/shared.py` adds an external reference
when needed. All task-local Python JITs remain part of the coverage check.

The standardized TritonBench-G and examples directories use automatic discovery.
The corpus inventory records the checked pairs and source hashes under
`validation/`. The generator's `--write` and `--check` options remain available
for older directories and isolated test fixtures that use a saved snapshot.
Updating source metadata does not accept a proof receipt.

Input types come from the actual input/IO declaration, including different
typed invocation variants. Unannotated Python pointers cannot determine their
storage dtype from a function body alone. Host/input compatibility and the
mathematical specification remain proof obligations. Generation never marks a
task complete or substitutes for Real/FP proofs and independent replay.

If several same-ABI declarations contain the identical body and helper links
cannot distinguish them, the tool reports the ambiguity. A short
`--bind 'path/source.py::function=path/Kernels.lean::definition'` selects a pair;
no repeated signature or dtype JSON is required. Missing stages, removed casts,
different statement order, unresolved helper links and unsupported syntax fail
the source check.

`audit_tritonbench_g.sh` without arguments builds and audits the complete
corpus; `--global-only` checks sources and library replay. The full corpus
CI audit runs daily at 04:00 Asia/Singapore or on manual dispatch.
`check_ports.sh` without arguments forwards to the current corpus audit;
named arguments select one task and all its retained Lean helpers.

Shared records are in [validation](../../validation/). Legacy catalog APIs
consume explicitly supplied frozen packages and do not select the live
benchmark implementation. See the [current workflow](../../documents/bench/current_source_acceptance.md).
