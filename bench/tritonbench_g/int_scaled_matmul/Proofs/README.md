# int_scaled_matmul

The repaired original is [int_scaled_matmul.py](../int_scaled_matmul.py).
[Sources/Body.lean](../Kernels.lean) contains both complete Python JIT bodies,
including the unscaled kernel's docstring, mutable `GROUP_M`, `add.u32` assembly,
all boundary checks, masks, pointer updates and plain stores. The scaled kernel
retains `range(K, 0, -BLOCK_K)`, `dtype=ACC_TYPE`, both rematerialized indices,
`mask.shape` and the unused `stride_s1n` argument. Invocation types and argument
values are supplied outside each pasted function body. The native call uses
INT8 A/B, INT32 C, FP32 scales and the original INT32 accumulator default.

[Kernels.lean](Support/KernelEntryAdapters.lean) only adapts typed parameters to those copied bodies.
It contains no other complete kernel quotation. The old extra input/output casts
and replacement natural-counter loop have been removed. In the execution proof,
a descending loop is represented by its signed mirror counter; every source
statement, including reconstruction of `k`, remains in the executed body.
The historical `steps` proof parameter bounds admitted iterations and never
replaces the actual source loop bound. Extra masked zero-product iterations are
proved to leave the carry unchanged.

The two existing optimization candidates are materialized in
[int_scaled_matmul.py](../int_scaled_matmul.py). One removes
only `c = accumulator` and stores the accumulator directly. The other removes
only the second `rm`/`rn` assignments and reuses their unchanged initial values.
[candidate-provenance.json](../../../../validation/tasks/int_scaled_matmul/candidate-provenance.json) pins both file hashes.
A whole-module inverse AST check verifies that restoring those statements and
JIT/launch names reproduces the original program. Their complete Lean bodies
are in the same source file; no extra implementation variant is authoritative.

The optional frontends preserve the entry value of a reassigned integer
constexpr, resolve the explicit integer dtype default, derive `mask.shape`
from the existing typed prefix, and interpret known INT8 storage reads on the
mathematical integer memory channel. This last lowering retains the original
load mask and padding and is checked for arbitrary integer carriers; no extra
byte-range input premise is introduced. Ambiguous pointer rebindings fail.

[Correct.lean](../RealCorrect.lean) states the independent full-K signed matrix product
and Real row scaling with toward-zero conversion. [FPEquiv.lean](../FPEquiv.lean)
compares the two actual candidates, including the signed32 carry, FP32 suffix,
all outputs, source access safety and exterior memory frames. The existing
host/native/compiler bridges remain separate from the ideal Real statement.
See [Contract.lean](../Contract.lean) for invocation conditions and
[source correspondence inventory](../../../../validation/source-correspondence.json) for all four source pairs.

The [validation records](../../../../validation) distinguish source correspondence,
local proof checks and exact-current independent acceptance. Compilation alone
is never recorded as completion.
