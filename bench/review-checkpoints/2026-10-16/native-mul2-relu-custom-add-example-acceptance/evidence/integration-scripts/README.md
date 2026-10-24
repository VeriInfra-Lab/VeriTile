# Four-task integration script candidates

These scripts are prepared for root review, not executed integration. They cover
only the four packages in `specs.json`: triton_mul2, relu_triton_kernel,
vector_addition_custom and add_example. Expected resulting source scope is 8/183
accepted, 175 pending. Add_value and sin_kernel remain later-batch candidates.

The script package was prepared entirely in this fresh /tmp directory. No
repository or index mutation was performed. `preflight.json` records receipt and
artifact checks; `lint-report.json` records AST/syntax/mapping checks;
`prepare-plan-report.json` exercises only the explicitly bounded read-only
prefix of prepare.py against the committed repository. These are not build or
integration-success receipts.

## Baseline and storage

Runtime HEAD is captured from a clean worktree; no future commit hash is assumed.
The final read-only plan observed 97e11a9a6cea8d5dfe35db643a1492d79e423cdf, following
the proof checkpoint commit 94c62a4bc4f188cfeb869add00b479a5dbefb7cc. The complete
predecessor checkpoint, storage checkpoint, metadata baseline and validator files
are hashed and required unchanged throughout every phase.

The installed gzip helper is pinned to producer receipt
48b2f06eb4b0194e8a13c7c03fda82e9ca50cf88f1a4fd9701fc854ad9b6ef0e and helper source
f553afe1345a77c499a77278c5b8a91942917d1c08745fc1e1679e6c67e9c4b7.
The producer receipt is copied unchanged as gzip-storage-receipt.json.

Paths are explicit: --baseline-ledger defaults to
bench/original_program_scope.json; --ledger defaults to
bench/original_program_scope.json.gz. Predecessor catalog/supplement/root review
default to their existing plain .json names. No extension or existence fallback
is used. New catalog, supplement, combined review and scope reviews use .json.gz.
All frozen producer/root/peer receipts remain their exact plain JSON bytes.

Prepare archives the exact old plain ledger as
`evidence/baseline-ledger.original.json` and also makes a compressed object
snapshot. Rebind creates the new active gzip ledger and removes the old active
plain path. The two previous progress documents are retained with exact bytes
as `evidence/baseline-progress.md` and
`evidence/baseline-current-source-acceptance.md`; both active documents then point
to the new catalog/ledger and report 8/175. Historical conclusions are linked as
historical snapshots, rather than left as competing current summaries.

Frozen .lean evidence becomes .lean.txt; generated Lean object suffixes and .pyc
become .bin. Adapter transformation adds only a deterministic mapping assignment
at each nested bind; removing that assignment/helper must restore the exact
original AST. All original path/hash/collision/public/whole-queue checks remain.
The authority dispatcher is also checked by restoring its previous AST.

## Phase sequence

Use a fresh runs-root. The following commands each run separately; inspect each
actual exit/log before advancing. Every command accepts --root, --packages,
--baseline-ledger, --ledger and the three explicit predecessor metadata names.
Defaults match the observed repository. Do not edit scripts after prepare:
preparation pins every script hash for the later phases.

```bash
NEXT_SCRIPTS=/tmp/veritile-standardization-20261015/next-source-integration-scripts-wip
NEXT_RUNS=/tmp/veritile-standardization-20261015/next-source-integration-runs

python -B "$NEXT_SCRIPTS/record_phase.py" --phase prepare --run-dir "$NEXT_RUNS/001-prepare"
python -B "$NEXT_SCRIPTS/record_phase.py" --phase install --run-dir "$NEXT_RUNS/002-install"
python -B "$NEXT_SCRIPTS/run_checks.py" --phase generate --run-dir "$NEXT_RUNS/003-generate"
python -B "$NEXT_SCRIPTS/record_phase.py" --phase rebind --run-dir "$NEXT_RUNS/004-rebind"
python -B "$NEXT_SCRIPTS/run_checks.py" --phase verify --run-dir "$NEXT_RUNS/005-verify"
python -B "$NEXT_SCRIPTS/finalize.py" --runs-root "$NEXT_RUNS" --selected-verify "$NEXT_RUNS/005-verify"
python -B "$NEXT_SCRIPTS/stage.py"
```

Prepare/install/rebind failures are recorded by record_phase.py. Generate and
verify retain their own actual subprocess manifests and logs. Each retry must
use a fresh numbered run directory. Preserve failed runs. A partially executed
mutating phase is deliberately not automatically rolled back or silently
repeated; root must inspect it. A killed run left as `running` blocks finalization
until its interruption is explicitly documented without inventing an exit code.

Verify runs all eight specified Lean build targets, complete bench/tests
unittest discovery (with both bench and bench/tests in PYTHONPATH), current
sharding tests, and the global audit. It returns nonzero when any actual command
returns nonzero, including the expected incomplete-scope global result. Finalize
independently requires successful build/test exits and the exact current
before/after input hashes. It permits only the exact 175-pending scope failure
and the explicit unavailable-official-comparator failure; actual comparator
success is also allowed. No job or test-success count is hardcoded. It retains
all run directories, including failures and generated baseline source archives.

Stage requires an initially empty index, exact validated bytes, only allowed
changes/deletion, no historical active source/object suffixes, and every staged
file below 100 MiB. It force-adds exact paths including receipt-bound ignored
artifacts, verifies every index blob against the reviewed worktree, checks
whitespace outside immutable evidence, and adds a staging receipt. It does not
commit or push.

## Preservation and review details

The unchanged validators recheck all 957 prior public declaration records,
153 queue rules, 1,642 exact provenance bindings, eight callback tables, previous
supplement declarations and registered modules. Existing four accepted rows
change only the catalog review references; the four new rows retain prior
reopening/history fields. All 175 unrelated ledger rows remain object-identical.

The new queue estimate is 94,505,619 bytes when conservatively including every
producer/root/peer archived input. With an extra 1 MiB reserve it is 95,554,195
bytes, below 104,857,600. This is a serialization estimate, not source-authority
validation; generation, finalize and stage independently check the actual file.
All 10,615 retained frozen artifact destinations passed hashes and per-file
size checks. The largest frozen artifact is 80,085,497 bytes.

Add_example did not contain a producer supplement-task proposal. Its constructed
source view explicitly selects original add_kernel, FiniteSource lowerBody and
runProgram, and FiniteHost sourceHost/sourceAllocated, with task_local_exact_ast,
sequentially_consistent_schedule and no ordinary-shared-DSL admission. The trust
boundary is the exact substantive root semantic-review text. Root reviewed this
selection against the prior complete nine-module review. Other task metadata
uses the existing ReLU row and frozen mul2/custom task proposals.

## Changes after initial root script review

- Added progress_docs.py and generated Markdown previews in lint-output; it
  replaces outdated active progress conclusions with 8/175 and explicit gzip
  catalog/ledger links while preserving both previous documents exactly.
- prepare.py snapshots both old document hashes and their archive destinations;
  common.py verifies those archived bytes and includes current_source_acceptance.md
  in the allowed active/staged/input set; rebind.py writes both current documents.
- All other script functionality was already included in the root's 12-script
  review. The final hash manifest is candidate-receipt.json.
