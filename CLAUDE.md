# Working in VeriTile

Read [AGENTS.md](AGENTS.md) for kernel fidelity, specification boundaries and
module responsibilities. This guide points to current tools and inventories.

## Repository map

- `VeriTile/`: shared Lean semantics, DSL, numerical models and proof interfaces.
- `bench/tritonbench_g/`: one current repaired implementation and its proof modules per task.
- `bench/examples/`: self-contained worked examples with separate source,
  independent Real correctness, FP equivalence and required proof helpers.
- `tests/lean/` and `tests/python/`: Lean and Python regressions.
- `validation/`: current selections, source reviews and validation records.
- `scripts/bench/`: benchmark inventory, audit and source-correspondence tools.
- `bench/bug_detection/` and `bench/pretrain_kernels/`: 334 imported challenges
  whose proof/backend obligations remain open.
- `documents/`: maintained references; `documents/archive/` holds past designs.
- `scripts/`: shared development and verification commands.
- `experiments/floating_point/`: numerical evidence and experiment tooling.
- `site/`: documentation website; generated references come from `documents/`.
- `verso/`: an independent presentation project with its own Lean toolchain.

`validation/task_progress.json` selects the current public task entries.
`validation/contracts.json` records their reviewed specifications and
conditions. `validation/source.json` binds the retained source tree.
Source correspondence, Lean proofs, numerical evidence and native execution
remain distinct results; preserve pending classifications.

## Build and verification

Use the target list in [lakefile.toml](lakefile.toml). Common commands are:

```sh
lake build                                      # shared lite library
lake build VeriTile VeriTileMath VeriTileFull    # complete shared library
bash scripts/bench/audit_tritonbench_g.sh --source-only  # source and review checks
python3 -B -m unittest discover -s tests/python -p 'test_*.py'
python3 scripts/check_docs.py                    # links and generated references
```

`scripts/bench/audit_tritonbench_g.sh` runs the complete corpus build and trust/replay
checks. `--global-only` checks sources and the library without corpus replay.
Official comparator tools and their sandbox are required for proof acceptance;
see [scripts/README.md](scripts/README.md). Judge commands by their exit codes,
not by a successful-looking tail of their output.

PR/main-push CI runs source/regression and library gates. The full retained
corpus and all draft interfaces run daily at 04:00 Asia/Singapore (20:00 UTC),
or on manual dispatch. Run the applicable complete proof checks locally before
merging proof changes. Never run the corpus scanner concurrently with tests
that create deliberately invalid temporary benchmark directories.

## Editing

Use [CodeOrganization.md](documents/CodeOrganization.md) for the directory
and dependency map, [ProofConventions.md](documents/ProofConventions.md) for
Lean style, and [MAIN_THEOREM_CONVENTIONS.md](documents/bench/MAIN_THEOREM_CONVENTIONS.md)
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
