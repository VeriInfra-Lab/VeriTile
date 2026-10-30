# Fast RMSNorm: kernel proof boundary

The selected source is `fast_rms_layernorm_repaired.py` (FP32), as recorded in
`task_progress.json`. The production Python and all four kernel ASTs are
unchanged. `Kernels.lean` contains the original function bodies for ordinary/Gemma
forward, the in-place input gradient and the increasing-row weight gradient.
Each body is copied into `triton { ... }` without dtype declarations or Lean
syntax inside it. Checked call signatures supply pointer/scalar types and values
outside the body; original docstrings, assignments, casts and control flow remain.
`fast_rms_layernorm_index_commuted.py` supplies the explicit Python candidate;
only the X row-stride multiplication order and the corresponding JIT names/calls
change. Its fourth (weight-gradient) JIT body is unchanged.
The original source remains unchanged and defective: backward overwrites caller
upstream storage and returns no weight gradient. The regression checker reproduces
both defects.

For a row of width N, let `r = 1 / sqrt(sum(x[j]^2)/N + eps)` and let `v[j]` be
`w[j]` ordinarily or `w[j]+1` in Gemma mode. The independent targets are
`y[j] = x[j]*r*v[j]`,
`dx[j] = r*g[j]*v[j] - r^3*x[j]*sum(g[k]*v[k]*x[k])/N`, and
`dw[j] = sum_rows(g[row,j]*x[row,j]*r[row])`.
Their definitions precede and do not call a kernel interpreter. The derivative
interpretation requires positive variance-plus-epsilon; the source contract
permits every finite Python float epsilon, including zero and negative values.
FP exceptional behavior is retained separately, without identifying it with Real
arithmetic.

The wrapper admits strided, default-contiguous FP32 CUDA tensors on the current
CUDA device with no unresolved negative views. X has positive last dimension
N ≤ 65536 and at most 2^31−1 rows. W can have any shape with N elements, including
a scalar at N=1. X/W have distinct backing identities; their nonempty pointers
are 16-byte aligned. Empty X never queries its pointer. Gemma must be a bool.
Backward G has the saved device and element count; its shape may differ and its
pointer may be unaligned. The exact guard equivalences remain in `Contract.lean`;
`observeTensor` connects its flags with actual Tensor metadata queries.

X/W remain direct views. Harmless singleton strides, empty leading dimensions,
aligned storage-offset windows and upstream aliases remain supported. Different
storage handles exposing overlapping read-only X/W bytes are also supported;
`KernelStorage.Placement` separates only buffers involved in a write. Backward
retains its existing private G-to-dX clone because the inherited dX kernel writes
in place. No new copy, dtype conversion, input restriction or output copyback was
introduced. dW reads that clone before dX modifies it. With no rows, every kernel
launch is skipped and dW is zero-filled.

The mathematical endpoints are `Correct.kernel_forward` and `Correct.kernel_backward`.
`Misc/NativeCorrect.lean` retains ordinary single-source theorems under the original
`FPEquiv.kernel_forward` and `FPEquiv.kernel_backward` names. The Real entries use shared
`PreparedAlgorithmProgramIO.Implements`/`ImplementsOn`; native entries use shared
`ProgramIO.Observes`/`ObservesOn`. They expose finite input/output channels and
independent formulas. Their conclusions include actual source access safety,
successful execution of the complete ordered grid sequence, every output and
the exact regional frame. Forward exposes Y and saved r; backward exposes
in-place DX and DW and runs the weight grid before the input-gradient grid.

`IO.lean` retains the `Ports`, `RealIO` and `NativeIO` namespaces for logical
shapes, roles, active inputs, actual sources, ABI arguments and full schedules.
It also contains the `Reference` channel/domain adapters and the native formula
needed by the main FP proof. The independent Real formulas remain in
`Proofs/RealReference.lean`; optional word observation adapters are in
`Misc/NativeReference.lean`. The native domain still admits arbitrary symbolic
terms, not only word leaves.
For zero rows, DW is an explicit preinitialized input with value zero and the
source schedule is empty. Nonempty backward imposes no initial DW condition.
`PhysicalIO.real_empty_domain` and `native_empty_domain` derive these empty
conditions from the original `zeroWeight` initializer. The exact prior raw
initialization/physical results remain ordinary supporting theorems.

Forward also takes `EpsilonABI.Binding`: the actual finite Python binary64 word,
its runtime FP32 argument, and the local round-to-nearest-even conversion law.
`RuntimeForward` binds that word to the distinct `eps` source register and uses
an independent epsilon input in its reference. Signed zero is preserved without
changing literal padding zeros. Each forward body has checked runtime-epsilon and mathematical-literal
instantiations with identical Python body text. The shared call-signature
frontend supplies the scalar binding and emits no declaration statement. `RuntimeForward.Source.runtime_body` and `runtime_length` prove the
typed statement lists coincide outside the inverse-statistic assignment. The
Real/optional historical path keeps the literal specialization.
The Real backward saved-r relation is exactly the independent relation
established by forward, assuming saved X has not changed. Native backward uses
the current saved words without claiming an adjoint for changed inputs.
`Proofs/IOExecution.lean` equates the new shared schedules' memory results with
the existing complete grids, including both backward stages. `IOSafety` and
`NativeIOSafety` establish access safety along those same actual executions;
the native proof includes all weight-gradient loop iterations.

`Proofs/PhysicalIO.lean` binds every original regional heap, and every
`KernelTensor.lift` physical reader, to these exact finite channels and original
allocation capacities. It adds no heap representation or read-only alias
restriction. Tensor applications retain their four-byte physical readback and
complete exterior-byte frames using the existing storage transport.

The ordinary theorems in `Misc/NativeCorrect.lean` connect source execution to
independently defined ordered scalar graphs, lowered through explicit native primitive instructions. Reduction trees
have proved complete lane coverage, operation order and cast/load/store edges
remain visible, and dW forces split arithmetic. Runtime words are erased from
the compiler-facing shape, with a recursive payload-independence theorem; only
evaluation retains those bits. Compiler choices, FMA contraction,
FP32 literals, rsqrt and scalar instruction meanings are model boundaries rather
than deferred experimental laws. No program execution or whole-output equation is
assumed, and no finite-input or reassociation law is needed. See `NUMERICS.md`.

Historical host behavior and failure/outcome theorems are preserved in
`Misc/OptionalHostCorrect.lean` and `Misc/OptionalHostFPEquiv.lean` under
`OptionalHostCorrect`/`OptionalHostFPEquiv`; the historical `Misc/RevisionAudit.lean` imports
these explicitly. Their historical placement domain is unchanged; the broader
read/read-overlap kernel result uses `KernelStorage`. They are optional and do
not define the default kernel claim.

Run `python bench/repairs/tritonbench_g/fast_rms_layernorm/check_contract.py` for
source pins and CPU execution of the actual kernel ASTs against independent
float64 autograd. It checks rejected inputs, both modes, both gradients, original
defects, singleton/offset layouts, upstream aliases, empty inputs and full caller
backing storage. This is regression evidence; GPU conformance was not run.

`Misc/TensorBridge.lean` contains the unchanged complete mathematical/native Tensor
applications and Python/physical-address corollaries. `Proofs/RawCorrect.lean`
and `Proofs/RawNative.lean` retain the former raw heap/byte statements exactly as
ordinary theorems. `Proofs/EntryInputs.lean` retains native byte interpretation
and zero encoding. Main dependency closures do not import `TensorBridge`;
supporting `Proofs/` modules do not import reader-facing entries. `Misc/Audit.lean`
checks public entries and Tensor applications together, while
`Misc/KernelAudit.lean` checks only supporting foundations.

Current targeted Lean checks cover all new IO, safety, reference, binding and
composition modules, both public entries, retained raw/physical readers and
audit consumers. Direct noncircular-reference, public statement-surface and
axiom-base gates are included. No GPU experiment or independent numerical
replay was run in this formatting revision.

`FPEquiv.kernel_equivalent` compares the complete original program with the
preserved X row-stride commutation candidate through shared
`ProgramIO.EquivalentOn`. The forward grid uses the actual runtime epsilon
register in both regular and Gemma mode. Backward retains the complete disjoint
feature-owner weight-gradient grid first, then every in-place input-gradient
row. The dW source, split arithmetic and row reduction order are unchanged.
Both programs are safe and succeed, active typed outputs agree, and both
complete exterior frames hold. Empty rows retain the existing explicit zero-DW
input domain. No finite-word restriction or experimental rewrite is introduced.

The body-correspondence manifest checks all eleven source/profile mappings,
including the runtime/literal epsilon instantiations. It rejects omitted casts,
changed branches, extra body directives and candidate mutations. Integer metadata
and row loops retain the original signed casts. `LegacyKernels.lean` is only a
historical presentation; the actual IO sources come from the copied bodies.
`Proofs/RealSource.lean` derives their final-memory equivalence to the earlier
algebraic presentation without equating register files or ASTs. The signed
weight loop has its own execution, value, frame and safety proofs.
`Proofs/SourcePair.lean` compares the complete copied source/candidate bodies
and proves both execution and access-safety equivalence. `Proofs/IOPair.lean`
composes their actual complete schedules. The independent Real targets and
optional native/Tensor/host applications retain their existing domains.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.
