# Working in VeriTile

Read [AGENTS.md](AGENTS.md) for kernel fidelity, specification boundaries and
module responsibilities. This guide points to current tools and inventories.

## Repository map

- `VeriTile/`: shared Lean semantics, DSL, numerical models and proof interfaces.
- `bench/examples/`: self-contained worked examples with separate source,
  independent Real correctness, FP equivalence and required proof helpers.
- `tests/lean/` and `tests/python/`: Lean and Python regressions.
- `scripts/bench/`: source-correspondence and standalone trust-audit helpers.
- `library/`: optional dependency view excluding the worked-example collection.
- [VeriTileBench](https://github.com/VeriInfra-Lab/VeriTileBench): selected repaired
  tasks, imported challenge interfaces, benchmark examples and validation records.
- `documents/`: maintained references; `documents/archive/` holds past designs.
- `scripts/`: shared development and verification commands.
- `experiments/floating_point/`: numerical evidence and experiment tooling.
- `site/`: documentation website; generated references come from `documents/`.
- `verso/`: an independent presentation project with its own Lean toolchain.

VeriTileBench's `validation/task_progress.json` selects its public task entries;
its historical contract review and current source inventories stay in that
repository. Source correspondence, Lean proofs, numerical evidence and native
execution remain distinct results; preserve pending classifications.

## Build and verification

Use the target list in [lakefile.toml](lakefile.toml). Common commands are:

```sh
lake build                                      # shared lite library
lake build VeriTile VeriTileMath VeriTileFull    # complete shared library
bash scripts/check-core-examples.sh             # three source/Real/FP pairs
python3 -B -m unittest scripts.test_library_dependency_profile
python3 scripts/check_docs.py                    # links and generated references
```

The core workflow builds the library and the three worked-example pairs on
PRs/main pushes; its nightly/manual job runs the full library artifact audit.
Corpus builds, source snapshots and per-module comparator replay run in a
VeriTileBench checkout. See [the benchmark validation guide](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/current_source_acceptance.md)
for its daily scope and commands. Official comparator tools and isolation are
required for independent proof acceptance; see [scripts/README.md](scripts/README.md).
Judge commands by their exit codes, not a successful-looking output tail.

## Editing

Use [CodeOrganization.md](documents/CodeOrganization.md) for the directory
and dependency map, [ProofConventions.md](documents/ProofConventions.md) for
Lean style, and [MAIN_THEOREM_CONVENTIONS.md](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/MAIN_THEOREM_CONVENTIONS.md)
for public theorem responsibilities. Task files may reuse explicit helpers;
they are not isolated one-file transcriptions.

Keep original Triton statements, casts, precision, masks and control flow
visible in `Kernels.lean`. Preserve public theorem conditions and complete
program scope when reorganizing helpers. Optional `Misc/` applications must
not become dependencies of public entries or their required `Proofs/` modules.

Edit reference documents under `documents/`, then run
`python3 site/scripts/sync-docs.py`. Do not edit their generated site copies.
Generated specification cards go to ignored `Logs/spec-sheets/` through
`python3 scripts/spec_sheet.py`; dated cards are available in the fixed
snapshot linked by `documents/bench/spec-sheets/README.md`.
