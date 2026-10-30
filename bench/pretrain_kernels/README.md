# Pretraining kernel equivalence problems

The catalog contains **269 cases**: 261 LOCAL cases and 8 TE cases. Forward,
input-gradient, weight-gradient, routing, optimizer, and attention variants are
separate cases. Each task has its own directory under `tasks/`; there is no
shared task module or cross-task Lean import.

All 269 entries contain **Lean problem drafts**. Drafts still require semantic
review and backend binding; compilation alone does not establish either. No
case is currently claimed as a proved equivalence, and no new numerical rule
is admitted by this collection. Each `task.json` records these statuses.

The [collection audit](../../documents/bench/IMPORTED_KERNEL_AUDIT.md) records confirmed conversion
defects and remaining proof obligations separately from compilation.

A Lean draft uses the same file responsibilities as the other benchmarks:

- `Kernels.lean`: two explicit arithmetic programs in the Triton DSL.
- `IO.lean`: typed inputs, complete scratch/output layout, coordinate
  equivalences, and execution schedule.
- `Contract.lean`: the Real formula and allowed input domain.
- `Correct.lean`: the complete Real execution, safety, output, and frame goal.
- `FPEquiv.lean`: the conditional `original ≡[R] candidate` problem.
- `Proofs/Backend.lean`: the outstanding connections to native implementations.
- `Misc/Validation.lean`: small interface checks.

Draft programs are explicit arithmetic lowerings. In particular, an attention
lowering is not yet a certified implementation of an external fused attention
kernel. Native tiling, reduction order, layout, launch configuration, and
rounding connections remain outstanding wherever the task records a pending
binding. The `Allowed` argument and rule set in the FP problem do not discharge
these connections.

Dynamic capacities and document boundaries are public input conditions.
Where a capacity depends on floating metadata, the FP goal takes an explicit
decoder and includes the decoded capacity condition. Its connection to the
selected numerical models remains pending. Tasks with a non-real literal also
expose a separate finite Real specialization and an unsolved connection goal.

Check the exact inventory and all draft modules with:

```sh
python bench/pretrain_kernels/check.py
```

The check requires all seven Lean modules in every task and rejects cross-task
imports. To build all modules through Lake, run `lake build PretrainKernels`.
For isolated per-task builds followed by a joint import, run:

```sh
python bench/pretrain_kernels/check.py --lean /path/to/lean \
  --library-path /path/to/compiled/base/library --record
```

The base library contains compiled VeriTile and Mathlib modules. It must not
contain this collection's task products. `--verify-record` checks whether the
recorded compilation evidence still matches the current files.
