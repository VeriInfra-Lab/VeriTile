# Checked direct `bgmv_expand_slice`

The repaired wrapper checks its input contract and launches on the supplied
contiguous buffers. It returns `None` and updates only active rows of the selected
output slice. The original Python is unchanged. The widened historical kernel
AST is unchanged; only its host wrapper and supplied strides changed.

The original kernel reads metadata as `indices + batch`, which ignores a strided
metadata view. The checked wrapper requires contiguous metadata and validates
its contents before launch. It accepts `-1` or an in-range adapter index. Sentinel
rows retain their original output bytes.

The selected profile is FP32 X, FP16 W, FP32 Y, and int32 or int64 indices. W has
shape `[L,N,K]` or `[L,1,N,K]`; the latter uses the original squeeze alias. X/Y are
rank two, K is positive, the metadata batch count fits both, and the output slice
fits Y. All operands share a CUDA device and use contiguous storage without lazy
negative/conjugate views. Aligned nonzero storage offsets and aliases among read
operands are accepted. Y must have a distinct backing from each read operand.

Executable checks bound scalar strides and launch axes, and cover signed64 byte
offsets for every formed lane, including masked lanes. Active accesses are proved
inside ordinary input storage windows. This removes the historical need for
padded scratch allocations. The wrapper performs no tensor snapshots, layout
conversion or copyback. Empty B/N returns before metadata reads, squeeze, device
context or launch. Nonempty execution uses the inputs' actual CUDA context.

`Contract.lean` defines the ordered Python predicates and proves their equivalence
to `InputContract`. The checked public statements are:

- `Correct.checked_real`: complete direct source execution, an independent K-term
  GEMV plus original Y when requested, and the logical frame.
- `FPEquiv.checked_fp`: complete typed source execution, the same supplied output
  object, `None`, all four active output bytes, and every untouched byte.
- `FPEquiv.checked_caller_bytes`: full original X/W/index backing preservation.

`Proofs/Direct/Bounds.lean` derives active capacities and signed64 formed offsets
from the checks. `Physical.lean` derives actual pointer words, full backing frames,
empty execution and the frame after a prefix of source stores. The latter models
an interrupted launch without claiming numerical success or exception atomicity.
Input/allocator bindings state only actual tensor windows, allocation identity,
physical separation, and squeeze alias metadata. Compiler bindings pin the whole
production AST, selected scalar reduction trees, and primitive-expression
lowering. No whole-result numerical premise is used. See `NUMERICS.md`.

The four root Lean entries are `Kernels`, `Contract`, `Correct`, and `FPEquiv`.
Correct and FPEquiv import independent helper paths; neither public entry imports
the other. Historical padded/snapshot proofs remain under `Proofs/`, with their
old public statements in `LegacyCorrect.lean` and `LegacyFPEquiv.lean`. They are
retained proof credit, not the checked direct wrapper's contract.

Run `python bench/repairs/tritonbench_g/bgmv_expand_slice/check_contract.py`.
The checker runs the actual wrapper and kernel AST with CPU operations, verifies
direct identities, canonical strides, dtype/AST pins, rejection before writes,
full backing frames, and partial-write context restoration. It covers both add
modes, K tails, sentinel/all-sentinel rows, empty shapes, both weight ranks,
metadata widths, aligned offset views and aliases among read operands.
This is structural/CPU evidence, not native GPU numerical conformance.
