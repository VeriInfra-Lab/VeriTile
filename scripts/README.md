# Automatic proof attempts

`scripts/prove.sh` runs the `lean4` Claude Code plugin's `/lean4:autoprove`
command, then uses [Lean's official comparator](https://github.com/leanprover/comparator)
to decide whether the requested theorems were proved.

## Setup

Use Linux with Landlock support and a working systemd user service. Install
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

`bash scripts/bench/audit_tritonbench_g.sh --source-only` is the shared source-check
entry for local work and CI. It checks the final inventory, Python/Lean body
correspondence and imported-challenge review without building or replaying
proofs. The full and library-only proof gates are described below.

`python3 scripts/check_docs.py` checks local file links in all tracked Markdown
documents and verifies generated reference pages. It does not fetch external
URLs. Edit references under `documents/`, then run
`python3 site/scripts/sync-docs.py` to regenerate the website copies.

`python3 scripts/spec_sheet.py` exports the current reviewed public readers to
ignored `Logs/spec-sheets/`. Use `--out DIR` to change the output location or
pass Lean files to print individual cards. Full source paths distinguish
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

The artifact checker, `scripts/bench/check_ports.sh`, and `scripts/bench/audit_trust.sh` all
require the official comparator and the sandbox tools installed above. Missing
tools, export failures, and comparator rejections fail the gate. CI installs
the same pinned tools through `.github/actions/setup-comparator`.

The Bench audit workflow checks the final source and contract inventory,
builds/replays the library on pull requests and main pushes. Full corpus builds
and replay, including all 334 benchmark interfaces, run daily at 04:00
Asia/Singapore (20:00 UTC), and on manual runs with `full_audit=true` (the
default). Each of the 64 corpus workers builds its complete dependencies, then
checks axioms/specifications and runs official comparator replay. The exporter
recursively includes the proof dependencies of each selected theorem.

The daily run covers all 8,428 retained corpus modules, including helper
theorems unused by public results. Its final gate requires the global job,
draft build and all shards to succeed; failed, cancelled or skipped jobs fail
it. PRs require the global gates and explicitly skip corpus/draft jobs. A PR
pass does not claim an exhaustive per-module audit. Separate concurrency groups
prevent PR or main pushes from cancelling a daily run. Before merging proof
changes, run the complete applicable proof checks locally.

Local `scripts/bench/audit_tritonbench_g.sh` runs still cover all gates and the whole
corpus by default. To reproduce the CI stages separately:

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

```bash
python3 scripts/check_comparator.py --library
python3 scripts/check_comparator.py --file bench/examples/VectorAdd/Correct.lean --trust
python3 scripts/test_check_comparator.py
```

Routine checks freeze the current trusted project sources and build cache in
an independent temporary workspace. Comparator exports and replays the current
proofs against that source snapshot (`source-replay` mode). This is not a check
that statements stayed unchanged from a previous Git revision; specification
changes still require review. `prove.sh` retains its separate comparison against
the original task captured before the agent runs.

The library gate selects `proven` library rows from the manifest. Standalone
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

The final aggregate checks `scripts/bench/final_inventory.py`, source correspondence
and the imported benchmark review before the build and proof gates. Generic
source-parser regressions remain in `scripts/test_source_*.py`; they test
extraction and rejection behavior without supplying archived corpus evidence.

The allocation regressions run the production zeros/full gate and check that
dtype attributes resolve consistently while changed shapes, fill values,
precisions and reassigned pointers still fail.
The dtype regressions retain declared pointer types through self-advances and
reject unknown rebindings and arithmetic expressions that are not direct loads.
The helper regressions validate explicit same-file single-return arithmetic
JIT expansion. The helper list is bound to the exact Lean selector and argument
map; every macro argument must be fixed or explicitly listed as a symbolic
dtype (branch selectors cannot be symbolic). The expanded expression must
match its Lean assignment as an AST. Unknown
calls, side effects, rebinding, namespace aliases and ambiguous assignments
remain unresolved. This comparison neither exempts a port nor certifies its
unselected source profiles or numerical semantics.

`Source-audit materializations:` records a runtime integer binding followed by
a symbolic arange-size identity adapter. It pins the complete original Python
file, exact Lean selector and every macro argument, explicit region storage,
the integer metadata, and reviewed adapter source bytes. It preserves the
original parameter list and every original statement: the comparison Python
body gains the same two identity assignments. Mandatory extraction then compares
the complete straight-line statement/expression AST, including every `tl.rand`
argument, strict comparison, eager division, mask and store. Computed seeds or
sizes, unrecorded metadata, unsupported control flow, changed adapter sources,
and implicit profile composition are rejected. Its adapter dependency is
`VeriTile/Triton/DSL/RandomProfile.lean`; changed library bytes require renewed
review. The tests retain the actual compiled source as text fixtures and use an
explicit source-only harness, without fabricated correctness entries. All 24
original storage/seed configurations are checked by production source gates.
These comparisons do not certify native ABI admission, GPU execution or a port.

`Source-audit pointer-dtypes:` binds a direct Python pointer storage comparison
to the exact Lean selector and complete macro arguments. The reviewed registry
`bench/tritonbench_g/source_pointer_dtype_profiles.json` retains the original
full-source hash, both persistent-matmul native compile records, and their TTIR.
The profile verifies every native pointer against selected Lean region types,
and checks the actual TTIR signature and hashes. It checks all original branches
for rebinding, indirect uses and unsupported dtype guards before selecting only
an exact `pointer.dtype.element_ty == tl.<dtype>` if arm. Runtime guards remain.
No metadata means no dtype-branch selection. These profiles currently reject
composition with other Python source rewrites; they do not create boolean
constexprs, add exemptions, or certify native integer/IEEE semantics or launches.

In the same checkout, run `test_audit_gates.py` and a full corpus audit
sequentially. The regression suite creates deliberately invalid temporary ports
under `bench/tritonbench_g/`, which a concurrent full scan would also select.

Logs under `Logs/comparator-check-*` record source hashes, selected targets,
generated source, comparator configuration, diagnostics, binary hash, and exit
status. Temporary build-cache copies are removed when a batch finishes.

Explicit boolean source profiles use a `Source-audit constexpr:` JSON object with
`python` parameter values and the exact selected `lean.source`/`lean.args`. Every
key must be an original `tl.constexpr` parameter used only as an `if`/`if not`
guard. Metadata never derives a profile from defaults or dtype names. Unselected
branches remain outside that comparison view; runtime parameters and precision
differences remain checked. The profile regressions exercise production gates,
malformed metadata, mutated branches and non-guard uses.

The runtime scalar regressions cover explicit Python-float/fp32 ABI views.
Each view pins the original Python source, selected Lean macro arguments, and
reviewed Triton 3.7.1 sm90 compilation evidence in
`bench/tritonbench_g/source_runtime_abi_profiles.json`. Only supported scalar
Name loads receive the ABI conversion in the comparison view; tensor casts
remain visible. Integer objects, stale evidence, rebindings, and unsupported
uses fail extraction. The recorded offline compilations do not launch kernels
or certify other values, shapes, runtime types, or Lean numeric contracts.

The recorded extern regressions cover Pow's explicit rank-one order parameters
and its same-file helper, separately from the pure arithmetic helper whitelist.
The view binds the complete original Python source, conditional import chain,
exact Lean macro arguments and reviewed native record in
`bench/tritonbench_g/source_extern_profiles.json`. The two integer-zero order
parameters must come from the original wrapper's `(0,)` launch arguments.
Validation retains all twelve original parameters, checks both original
branches and compares the complete selected body as an AST. The helper's
existing fp32 cast remains explicit; no extra runtime ABI conversion is added.
This profile records fp32 `__nv_powf` dispatch for Triton 3.7.1/sm90 with block
size 32. It does not certify other imports, dtypes, native numerical results,
or a complete Pow contract. Invalid metadata, changed casts, pointer fields,
bindings or evidence fail the mandatory extraction gate, even for a port with
an existing translation marker.
