# kldiv_ops launched-kernel specification

The selected source is the unchanged `kldiv_ops_repaired.py`, SHA-256
`5a610a722d441f13cccdf9ef8443f994046b26bd1fbec5f858c12d1f8809a20d`.
The public `Contract.lean`, `Correct.lean`, and auxiliary `Misc/NativeCorrect.lean`
retain the two launched kernel specifications in namespace `VeriTile.Bench.Repairs.KldivOps.Kernel`.

Forward writes `t * (log(max(t, eps)) - y)` for a raw target and
`exp(t) * (t - y)` for a log target. `none` writes every matrix element;
`sum`, `mean`, and `batchmean` all write one unnormalized sum per row. The later
Python reduction and normalization are outside this kernel result. Backward
writes `-t` or `-exp(t)` directly into the supplied destination. The cotangent
multiplication and scalar-one return branch happen later in Python.
`Contract.loss`, `forwardTarget`, and `backwardTarget` define these mathematical
formulas independently of the source interpreter. Real log and division use
Lean's total mathematical operations; native exceptional words are described
separately, without encoding arbitrary reals into FP32 bytes.

## Exact selected admission

The wrapper is unchanged. It checks Tensor arguments, strided layout, FP32,
CUDA on one device, contiguous layout, resolved negative views, and alignment
of every active pointer. Prediction and target have matching rank-two shapes.
`log_target` is a Python bool, forward reduction is one of the four supported
strings, and epsilon is a Python float. There is no tensor-value guard;
exceptional FP32 words and all Python epsilon encodings remain admitted.

Backward additionally admits only a scalar cotangent or the exact destination
shape, and requires nonempty destination backing storage to be disjoint from
both target and cotangent. Read-only inputs may alias each other. Nonzero
storage offsets, unused strides of degenerate dimensions, and both empty axes
remain admitted. No copying, new size cap, source guard, or source dtype change
was introduced. `python_forward_checks_iff_preconditions` and
`python_backward_checks_iff_preconditions` retain the exact checked predicates.
Forward output shape, dense layout, alignment, backing, and separation describe
the existing fresh output allocation. `Dispatch` describes signed PID
representability of an ordinary successful launch, rather than a new guard.

## Executed program and physical result

`Kernels.lean` is the complete readable typed forward/backward DSL. Its external
epsilon declaration supplies only a register type and emits no instruction or
value; the execution binds the actual ABI scalar. Literal typed-AST equalities
connect it to the preserved `Proofs/HistoricalKernels.lean` proofs.

`Proofs/KernelExecution.lean` defines an `Option` execution that runs each
complete typed source row against the current shared physical memory and
commits its stores before the next row. Its success is proved. The source
retains the full chunk loop, tail masks, both target branches, FP32 per-chunk
reduction, sequential FP32 accumulator initialized by `0.0`, and final store.
The explicit remask is only on the reduced branch. Runtime epsilon is bound
through the scalar ABI; it is not substituted by an ideal real constant.

`IO.lean` declares the actual forward/backward sources, full typed input and
output spans, masks, no private scratch, and the complete row schedules. The
runtime epsilon register stays explicit. Backward declares only its actual T
input; Y/T may alias in forward, with output separation retained.

`Correct.forward/backward` use the shared `ProgramIO.Implements` judgment with
independent logical-array formulas. The judgment connects actual source access
safety, successful full-grid execution, every typed output and the exterior
frame. The ordinary `FPEquiv.forward/backward` auxiliaries in
`Misc/NativeCorrect.lean` use that same IO relation with native FP32
observations, the original epsilon ABI conversion and ordered compiler
reductions. `FPEquiv.lean` records the remaining whole-task implementation
pair connection described below. Public statements contain no raw state,
source runner or memory predicate. The previous operational results remain ordinary helpers in
`Misc/RawCorrect.lean` and `Misc/RawFPEquiv.lean`.

`Proofs/SafetyForward.lean` and `Proofs/SafetyBackward.lean` prove actual evaluated
load/store bounds through both signed chunk loops. Forward safety includes the
post-loop scalar store even with zero columns. `Proofs/IOSchedule.lean` identifies
the shared invocation schedule with the existing proven execution; the logical
IO adapters reuse those execution/results proofs rather than re-proving them.

`TensorBridge.Correct.forward/backward` and
`TensorBridge.FPEquiv.forward/backward` retain the complete original Tensor
applications. They compose actual Tensor bindings into this execution and
prove every output value, native pointer equality, aligned pointers and all four backed
bytes, signed PID/tail-mask/terminal-increment facts, and the complete memory
complement. The frame includes both raw reads and every observed byte outside
the destination, including padding and partial-byte allocation tails. Source
loads stay stable through write prefixes by the selected output separation;
forward prediction/target aliasing is preserved.

The FP theorem compares the executed source with an independent ordered native
word graph. The graph keeps operand roles, FP32 epsilon binding, distinct
literal-minus-one multiplication and unary negation, the compiler-selected
binary addition tree, and sequential chunk carry. The shared IO result proves every output word and the typed frame; the retained
Tensor bridge additionally proves all output bytes, including NaNs, infinities, signed zeros, and subnormals under that
same primitive interpretation. Primitive meanings and compiler tree selection
are ordinary backend model parameters. There is no numerical transformation,
all-word cross-primitive equality assumption, or deferred experiment in these
public kernel results. See `numerical-obligations.json`.

## Preserved history and verification

The previous public endpoints are preserved byte-for-byte under
`Proofs/HostContract.lean`, `Proofs/HostCorrect.lean`, and
`Proofs/HostFPEquiv.lean`; their names and whole-wrapper conclusions remain
available on explicit import. `README.host.md` and
`numerical-obligations.host.json` preserve their scope and optional numerical
ports. The original source and conditional candidate are unchanged.

The original defects remain recorded: reduced inactive raw-target lanes can
introduce NaN when epsilon is zero; non-unit logical strides can disagree with
flat source loads; unresolved negative target views change backward semantics.
The existing CPU checker reproduced all three, passed 38 admission rejections,
five ordinary failure prefixes, and 182 actual source/reference/frame cases.
These are source controls, not GPU conformance or numerical-law validation.
The new Lean endpoints and supporting lemmas use standard-base axiom audits.

### Implementation-equivalence review (2026-10-21)

The complete forward/backward source-to-native results are ordinary auxiliary
theorems in `Misc/NativeCorrect.lean`; their public declaration names, runtime FP32
epsilon ABI, full row/chunk schedules, reduction modes and frames are retained.
The existing original-task candidate commutes only the exact signed output-row
product. `Proofs/TypedPreservation.lean` already preserves its backward pairs,
but no corresponding full repaired runtime-epsilon forward candidate connection
has been proved. Backward-only equivalence cannot complete this whole task.
No implementation-equivalence headline is claimed until that bridge exists.
