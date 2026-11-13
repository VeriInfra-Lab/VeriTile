# Automatic proof attempts

`scripts/prove.sh` runs the `lean4` Claude Code plugin's `/lean4:autoprove`
command, then uses [Lean's official comparator](https://github.com/leanprover/comparator)
to decide whether the requested theorems were proved.

## Setup

Use Linux with Landlock ABI v4 or newer and a working systemd user service. Install
Go 1.24+, the project's Lean toolchain, Python 3, and ripgrep (`rg`).
The Claude Code `lean4` plugin is needed only for agent-driven proof attempts
through `scripts/prove.sh`; the artifact and bench audits do not need it.
Build the judge tools once, outside the project (to avoid copying them into each
judging workspace):

```bash
scripts/setup-comparator.sh /tmp/veritile-proof-tools
export PATH="/tmp/veritile-proof-tools/bin:$PATH"
lake build VeriTile VeriTileFull
```

The installer pins upstream comparator commit
`2a00b30df5e9173e70c4e4ec669fdf03da3163b9`, which targets Lean 4.29.0,
its lockfile's lean4export revision, and landrun commit
`5283024a2f49b28046c3b4a06d7d775c058d4d80`. Revisit these pins when upgrading
Lean. The landrun pin preserves the CLI argument handling expected by this
comparator version. `COMPARATOR_BIN` may specify another compatible official comparator
executable; `landrun` and `lean4export` must be on `PATH`.

The prerequisite check uses comparator's `--best-effort` filesystem settings
and separately requires strict Landlock TCP mediation. This supports ABI v4
while keeping filesystem and network isolation mandatory. Run the checks from
the host shell when an enclosing workspace sandbox denies access to the
systemd user bus.

For the pinned Lean 4.29 checker, the installer backports the upstream
[string-literal replay fix](https://github.com/leanprover/lean4/commit/94e0a0e4f6f4c3fa207aa7738fdd198067b4b3de).
It checks the exact checker source before and after applying the patch. The
fix replays `String.ofList` and `Char.ofNat` before declarations that reduce
string literals; these implicit dependencies are absent from ordinary
constant traversal. Both definitions are still checked by the kernel.
Statement comparison, axiom checks, kernel replay and sandboxing remain enabled.

## Usage

```bash
scripts/prove.sh path/to/Task.lean --theorem MyKernel.correctness
scripts/prove.sh path/to/Task.lean --theorem MyKernel.correctness \
  --theorem MyKernel.equivalence --max-cycles 20 \
  --prompt "Try induction on n first."
```

Replace the schematic path and names with your task. The file must be inside
this project. **`--theorem` is required** and repeatable: the caller selects the
fully qualified declarations to judge, before the agent runs. The default
limit is five proof cycles. Only those selected theorems and their dependencies
are judged; this is not a claim that every declaration in the file is proved.

## Judgment and logs

Before proof search, the runner copies the original file, project Lean sources,
Lake configuration, and dependency/build cache into an independent temporary
workspace. This needs temporary disk space for `.lake` (reflinks are used where
supported). After the agent exits, only its resulting source file is admitted
to that workspace. The agent's build artifacts and changes to imports or
configuration are not copied back into the judge.

Comparator builds and compares `ComparatorChallenge` and `ComparatorSolution`,
checks the selected statements and their referenced definitions, checks for
unpermitted axioms, and replays the exported proof in Lean's kernel. The allowed
axioms are exactly `propext`, `Quot.sound`, and `Classical.choice`. The runner uses
upstream's systemd AF_UNIX restriction along with comparator's landrun sandbox.
It never falls back to matching agent text or compiler output.

- Exit 0: official comparator accepted all selected theorems.
- Nonzero: rejection, missing tools, unavailable sandbox, invalid arguments, or
  another execution failure. An agent's success report cannot override this.

Each attempt writes `Logs/<basename>_<unique-id>/` with the original challenge,
candidate source, comparator configuration and output, agent JSON stream,
input-source hashes, and `result.json` containing the two exit codes and the
comparator binary hash. The temporary copy of the build cache is removed.

The starting sources, dependencies, tools, and host must be trusted. This
wrapper runs a local coding agent with filesystem access; it does not isolate
a hostile agent from the host. Use separate solver/judge machines for that
threat model. Comparator compares declaration names exactly: private or
module-generated names in a statement can differ between the two modules and
be rejected even for a valid proof. Prefer public definitions for task statements
and their specification/kernel dependencies.

## Regression checks

With the judge tools on `PATH`, run:

```bash
python3 scripts/test_prove.py
```

These tests use the real comparator and a deterministic fake agent (no API
calls). They cover a valid proof, changed statements and definitions, extra
axioms, remaining `sorry`, deleted targets, and multiple requested theorems.

Proof search still depends on the installed `lean4` plugin version; record it
when reporting benchmark results. The previous benchmark setup used version
4.4.9 of the plugin.

## Source and documentation tools

`bash scripts/check-core-examples.sh` checks Python/Lean source correspondence
and both Real and FP proof entries for VectorAdd, RowWiseSum, and LogExp.
For corpus commands in this guide, use a checkout of
[VeriTileBench](https://github.com/VeriInfra-Lab/VeriTileBench) with its locked Lake dependencies.

`python3 scripts/check_docs.py` checks local file links and same-repository
current-main GitHub source/workflow URLs in all tracked Markdown documents,
and verifies generated reference pages. Pinned historical references remain
external so archived paths are not judged against a newer checkout. It does not fetch external
URLs. Edit references under `documents/`, then run
`python3 site/scripts/sync-docs.py` to regenerate the website copies.

`python3 scripts/spec_sheet.py bench/examples/VectorAdd/RealCorrect.lean`
prints a core example card. Use `--out DIR` to write cards to an ignored output
directory. In VeriTileBench, invocation without files selects its current
public declarations and writes to `Logs/spec-sheets/`. Full source paths distinguish
nested/repaired revisions. Generated cards are presentation aids; their lexical
flags do not establish specification independence or proof acceptance.

The declaration parser lives in `scripts/spec_sheet.py`; the standalone lexical
mask lives in `scripts/bench/lean_import_scan.py`. Source adapters and final inventory
checks share those implementations rather than keeping separate copies.

## Artifact checker

`scripts/check-artifact.sh` is the local release/CI gate for the Lean artifact.
It runs `lake build VeriTile VeriTileFull`, rejects Lean `sorry` warnings, runs
the official comparator on every proven library target in the manifest, checks declared axioms
against `scripts/artifact-axiom-whitelist.txt`, validates the per-kernel
registry in `scripts/kernel-manifest.tsv`, and checks README example links for
drift. It also resolves documented public API names with Lean through
`site/scripts/check-doc-api.py`.

`scripts/kernel-manifest.tsv` is the source of truth for public kernel/example
metadata: file, theorem symbol, theorem kind, verification status, source,
static config, label, and notes. See `documents/KernelManifest.md` before
adding a new public example or benchmark port.

```bash
scripts/check-artifact.sh
```

## Shared comparator gate

The core artifact checker and standalone comparator, together with VeriTileBench's
`scripts/bench/check_ports.sh` and `scripts/bench/audit_trust.sh`, all
require the official comparator and the sandbox tools installed above. Missing
tools, export failures, and comparator rejections fail the gate. CI installs
the same pinned tools through `.github/actions/setup-comparator`.

The VeriTile workflow builds the library and the three worked examples on
pull requests and main pushes. Its nightly or manual library audit runs the
full artifact checker. VeriTileBench runs its full corpus build and comparator
replay daily at 04:00 Asia/Singapore (20:00 UTC) and on manual dispatch.

In a **VeriTileBench checkout**, `scripts/bench/audit_tritonbench_g.sh` covers
all gates and the whole corpus by default. To reproduce its CI stages separately:

```bash
scripts/bench/audit_tritonbench_g.sh --global-only
# Daily full scope (run indices 0–63):
AUDIT_TRUST_SHARD_COUNT=64 AUDIT_TRUST_SHARD_INDEX=0 scripts/bench/audit_trust.sh --final-inventory
# Optional public-entry audit (run indices 0–15):
AUDIT_TRUST_SHARD_COUNT=16 AUDIT_TRUST_SHARD_INDEX=0 scripts/bench/audit_trust.sh --ci-targets
```

Run every index for the selected shard count for a complete corpus audit. Both shard
variables are required, and sharding cannot be combined with named targets or
`--global-only`. A global-only result or a single shard is not a completed
audit. The existing sharded aggregate command also remains available: setting
the two variables on `scripts/bench/audit_tritonbench_g.sh` runs the global gates and
the selected shard together.

Back in the **VeriTile checkout**, the core checks are:

```bash
python3 scripts/check_comparator.py --library
python3 scripts/check_comparator.py --file bench/examples/VectorAdd/RealCorrect.lean --trust
python3 scripts/test_check_comparator.py
```

Routine checks freeze the current trusted project sources and build cache in
an independent temporary workspace. Comparator exports and replays the current
proofs against that source snapshot (`source-replay` mode). This is not a check
that statements stayed unchanged from a previous Git revision; specification
changes still require review. `prove.sh` retains its separate comparison against
the original task captured before the agent runs.

The library gate selects `proven` rows under `VeriTile/` from the manifest
and recursively checks their theorem dependencies. It does not inventory every
unreferenced library theorem. Standalone
gates enumerate every theorem object retained in the file's Lean environment,
including private and macro-generated names, and expose checked aliases to
comparator. Definition-only fixtures report zero theorem targets and run a
separate trivial sentinel through comparator; this does not establish a kernel
correctness claim. Anonymous `example` commands that Lean does not retain as
theorem declarations are compilation tests, not inventoried proof targets.

The trust gate also retains its headline, statement, and circular-spec checks.
The aggregate bench audit performs compilation and comparator replay in one
trust pass. Batch workers share one independent snapshot, with separate output
modules, and concurrency accounts for export/replay memory. Raw `lake build`
remains the build step; use these gates for proof acceptance.

VeriTileBench's aggregate audit additionally checks its final inventory, source
correspondence and imported-challenge review before proof acceptance. Run its
`test_audit_gates.py` sequentially with the corpus scanner: those regressions
create deliberately invalid temporary task directories.

### Source correspondence and compatibility fixtures

Current examples use automatic discovery from the Python function and the
shared `Kernels.lean` body:

```bash
python3 -m scripts.bench.generate_jit_correspondence bench/examples/VectorAdd
```

It checks statements, casts, indexes, masks, source parameters and helper links;
it does not execute Triton or prove mathematical correctness. Bind input types
and constexpr parameters outside the pasted body. The authoring standard is in
[AGENTS.md](../AGENTS.md) and the
[translation guide](../site/src/content/docs/cookbook/translating.md).
Collection-wide commands run in VeriTileBench, which owns the task inventory
and validation receipts. A passing correspondence check is separate from
Real/FP proof acceptance and GPU evidence.

The retained `test_source_*.py` suites also exercise older explicit
`Source-audit` adapters: constexpr branches, runtime scalar ABI bindings,
materializations, pointer dtypes and extern/helper calls. Their frozen evidence
is isolated under `scripts/fixtures/`; it is not a task-local JSON requirement
or a current production registry under this core repository's `bench/`.
See the [materialization fixtures](fixtures/source_materializations/README.md),
[uniform fixtures](fixtures/source_uniform_materializations/README.md) and
[control fixtures](fixtures/source_control_materializations/README.md).
Each adapter checks its selected source view and exact metadata/evidence;
unselected branches, other precisions, native ABI admission and hardware
behavior remain outside that comparison. Changed casts, stale source hashes,
unsupported rebinding or ambiguous helper links must fail rather than acquire
an exemption. New authoring retains the complete original function body.

Logs under `Logs/comparator-check-*` record source hashes, selected targets,
generated source, comparator configuration, diagnostics, binary hash, and exit
status. Temporary build-cache copies are removed when a batch finishes.
