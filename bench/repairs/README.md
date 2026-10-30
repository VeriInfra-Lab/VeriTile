# Explicit-contract source revisions

These revisions follow the input-contract and proof organization approved for
[`add_example`](tritonbench_g/add_example/README.md). Each task owns its Python
source, `Kernels.lean`, public `Contract.lean` when needed, independent
`Correct.lean` and `FPEquiv.lean`, supporting `Proofs/`, optional `Misc/` results, and a scope/validation
README. Shared Tensor metadata and memory bindings belong in `VeriTile/Torch`.
The shared [GPU memory model](../../documents/GpuMemoryModel.md) records the
load/store alignment rules and their normative references; kernel-specific
lowering observations stay with the corresponding task.

The review entry points have these responsibilities:

| File | Contents |
| --- | --- |
| `Kernels.lean` | Shared readable kernel source and implementation variants. |
| `Contract.lean` | Input conditions and their check semantics. |
| `IO.lean` (when shared) | Explicit source, runtime argument bindings, typed channels, windows, masks and launch schedule used by both entries. |
| `Correct.lean` | Independent Real mathematical formula, named public IO, and a short correctness specification. |
| `FPEquiv.lean` | Equivalence of two actual kernel implementations (`original ≡[R] candidate`), with explicit precision/model boundaries and a short proof. |
| `Misc/TensorBridge.lean` | Optional Tensor applications, input-binding corollaries and public check-correspondence helpers, as ordinary theorems. |
| `Misc/NativeCorrect.lean` | Retained single-source FP/native observations; these do not replace implementation equivalence. |
| `Proofs/` | Supporting proofs and implementation details needed by these entries. |
| `Misc/` | Other optional host results, candidate applications and audits such as `Misc/PublicChecks.lean`. |

The primary entries and their required helpers do not import optional `Misc/`
entries, directly or indirectly. Shared definitions required by the main proofs
stay in the source, IO, contract or `Proofs/` layers.
Their statements use geometry, scalars, logical arrays and shared IO judgments.
Raw `State`, `run`, `exec`, `Cell`, memory-read propositions and existential
post-states belong in `Proofs/`; hiding the same proposition behind a task-local
`IO` or `Correct` name does not meet this rule. Public IO data may live in a
shared task `IO.lean`; it must expose the actual source and ABI, channels,
windows, masks, precision and full schedule. A complete judgment connects
logical input bindings, actual source-access safety, successful execution,
independent outputs and the exact exterior memory frame.

Actual Tensor objects, metadata flags, wrapper admission and returned Tensor
identities belong in `Misc/TensorBridge.lean`. Retain prior Tensor and operational
results using the same source-proof foundation. Supporting proofs must not
import either public reader entry, including through another task. Preserve
the input domain, full grid/stage scope, output coverage and memory frames. Distinct kernels,
stages and supported source branches can have distinct main specifications.
A separate checked application theorem is optional for a task that never had
one; do not duplicate mathematical correctness in `FPEquiv.lean` to add one.
Neither single-source FP32 correctness nor host/Tensor proofs are additional
per-task acceptance requirements.
The [ReLU example](tritonbench_g/relu_triton_kernel/README.md) demonstrates this
separation; [Lion](tritonbench_g/adam_update_triton/README.md) demonstrates runtime
scalar arguments and two in-place outputs. Optional flat-memory corollaries and
source-identity checks are ordinary auxiliary theorems, not extra primary specifications.

The Algorithm regional IO and Structural `ProgramIO` preserve regional memory
semantics without adding disjoint-input assumptions. Their source-access safety
is proved, not supplied as a numerical premise. Flat-pointer transport remains
an explicit stronger result. A custom native runner needs an actual shared
source connection; an arbitrary execution callback is not an IO adapter.

The single [task checklist](../tritonbench_g/task_progress.json) distinguishes
retained historical acceptance from the current revision. The earlier checkout
contains accepted original programs or separate repairs for all 183 identities;
those results retain their exact source, input scope and numerical conditions.
They do not automatically certify changed files here. An unregistered work
directory or a successful compilation alone is not an accepted whole task.
The checklist counts public IO format completion separately from retained proof
acceptance. The earlier kernel/Tensor split did not itself complete this format.

Current revisions use justified explicit input checks whose domain agrees with
the public theorem. They reuse applicable whole proofs and preserve source
precision (FP32 when production input dtype is unspecified). They do not add
copies, arbitrary layouts, device transfers, or dtype variants merely to expand
support. Original intentional materialization and required integer/quantized
types are preserved. Concrete source defects and their counterexamples remain
separate from the revised program's correctness result.

Numerical deferrals state the operation, precision, domain and outstanding
experiment in the task README and public numerical premise. Structural proof
gaps cannot be deferred this way. GPU execution, offline compilation and CPU
replay are different kinds of evidence and are reported separately. Shared
launcher grid-product compatibility and historical cleanup remain in
[`bench/TODO.md`](../TODO.md).

Run `python bench/task_progress.py --check` to check the checklist's structure
and rendered table. This does not reprove kernels or certify numerical laws.
