# FP32 forward RoPE kernel

The selected canonical source is `rope_transform_repaired.py`, SHA256
`daddf5be173d6b04740e71f405f4ed46a8efdce54b2fd75f882c7dc878c7d9eb`.
This proof update leaves Python and its guards unchanged. `Kernels.lean` quotes
the complete original JIT, with both branches, all four Q/K loads, coefficient
loads, casts and stores. `KernelContract.source_binding` identifies it with the
retained successful typed-source proof by definitional equality. The selected
Python entry passes `BACKWARD_PASS=False`; that forward branch is the public
kernel theorem's scope.

The actual kernel arguments are prepared contiguous FP32 Q/K Tensors
`[B,S,H,D]`, with independent positive head counts and positive even D.
Coefficient Tensors are contiguous FP32 arrays exactly `[S,D/2]`. All four
arguments share a CUDA device and have separate nonempty backing intervals.
Valid nonzero storage offsets are included. Empty B or S submit no programs;
unused empty data pointers are not constrained by Tensor pointer laws. The
launch passes the logical packed Q/K stride `H*D` and table stride `D/2`
explicitly, including singleton axes with otherwise unusual Tensor strides.
Admission preserves the existing padded signed-i32, u64 pointer and 1,048,576
padded tile-element checks. It does not assume source success or an output.

The independent forward pair is `(a*c-b*s, b*c+a*s)` for the current supplied
coefficients. No unit-circle premise is imposed. The backward branch computes
its fixed-coefficient input adjoint `(a*c+b*s, b*c-a*s)` and is generally not an
inverse. Both inputs to each pair, and both Q/K tiles, are loaded before stores.
All Q/K coordinates are updated in the accepted even-D domain. Odd dimensions,
zero heads, strided coefficients and overlapping nonempty backings remain
outside the existing canonical domain.

`Correct.lean` exposes one public `rope_forward_correct` specification:
`realIO` implements the independent logical rotation polynomial. `IO.lean`
lists Q/K as in-place FP32 input/output channels, C/S as read-only tables, the
complete row schedule, the exact cleared-register/pid/numPids launch setup,
dense windows, masks and empty scratch footprint. `ProgramIO.Implements`
proves actual access safety, successful execution, every typed Q/K output and
the complete exterior cell frame. Zero B/S produces an empty schedule.
The modeled grid completes rows sequentially; equivalence to GPU CTA scheduling
remains the existing explicit runtime boundary.

`Misc/NativeCorrect.lean` retains the ordinary `rope_forward_fp32` source-to-native observation
using the same full-grid IO and unrestricted original `Tagged` inputs.
The shared `ProgramIO.Observes` maps only the output to words, preserving the
identity input binding and arbitrary original provenance tags. It uses concrete shared FP32 primitives under
`FP.Config` and the existing source-specific ordered graph
`low = fma(c,a,neg(mul(s,b)))`, `high = fma(s,a,mul(c,b))`.
Both products place the coefficient first; low contracts the left product and
high contracts the right. FP32 load/store casts and all FP32 words are retained.
This does not invent a second optimized kernel or assume native=Real.

The former raw-state theorems, including the no-launch result and native full
grid result, retain their exact names and statements in `Proofs/RawResults.lean`.
`Proofs/AccessSafety.lean` proves bounds for the actual reached coefficient/input
loads and all four stores. `Proofs/IOGrid.lean` reuses the original source
execution, readback and frame results to connect every logical input/output.

`TensorBridge.rope_forward_kernel_correct` and
`TensorBridge.rope_forward_kernel_fp32` retain the full actual Tensor
applications and their address bounds. Actual input bytes and Tensor pointers
feed the same typed execution from which every Q/K
output word and all four bytes are read. Local initial-memory/store commuting
proofs justify the physical interpretation. The full exterior byte frame
includes the coefficient allocations, backing prefixes/suffixes and every
unwritten address. The pure little-endian codec reuses the frozen fast RoPE
proof; Tensor and primitive helpers reuse the completed backward RoPE work.

Explicit model boundaries are the shared Tensor metadata/storage/indexing API
laws, primitive FP32 semantics under cfg, source-specific compiler lowering,
Python-to-readable-DSL correspondence, and runtime grid scheduling. There is no
local real-exactness or numerical replay premise. Host preparation, allocator
behavior, returned-object identity, compiler internals and GPU-to-instruction
conformance are outside the public kernel endpoints. Prior host theorems are
preserved in `Proofs/HostContract`, `Proofs/HostCorrect` and
`Proofs/HostFPEquiv`; no new wrapper or allocator proof was added.

Validation comprises targeted Lean builds and the combined standard-base
axiom/noncircularity audits in `Proofs/KernelMisc/Audit.lean` (operational helpers)
and `Misc/Audit.lean` (public/Tensor entries). The existing checker verifies
the unchanged JIT AST, eight CPU copy/reuse/empty paths and twelve rejection
paths, and reproduces the original S=1 row-stride defect. Four additional
actual-JIT CPU layout cases check all outputs and backing frame in reverse row
order. Branch, nonunit, formed-index and two fused-rounding counterexamples are
recorded. Four offline Triton 3.7.1 profiles check all ordered Q/K stored graphs
and masked FP32 accesses: forward/backward on SM80 (`D=6,Hq=3,Hk=1`) and SM90
(`D=34,Hq=5,Hk=3`). These checks support the explicit source/compiler boundaries;
they are not universal numerical premises. The offline cases supply explicit
ASTSource signatures and do not verify the signature chosen by the actual
runtime binder. No GPU was executed.

The public IO entries have strict statement-surface and independent-formula
checks. The complete Tensor endpoints, byte observations, pointer bounds and
physical frame remain ordinary results in `Misc/TensorBridge.lean`. Both layers use
the existing source execution/output/frame foundation; regional IO does not
stand in for physical address/byte transport. Targeted public/helper/Tensor
builds pass. No new GPU run, numerical experiment or independent replay was
performed in this formatting pass.

## Implementation-pair review (2026-10-21)

`Kernels.indexShared` is a complete readable optimization candidate that
reuses the existing coefficient lane vector and half-width mask in the Q/K
address and mask tiles. It removes repeated `arange` generation and comparisons;
no measured performance claim is made. Both direction branches, all FP32
instructions, casts, loads and stores remain explicit and unchanged.

`FPEquiv.source_candidate_equivalence` compares the original and candidate as
actual complete forward FP32 row schedules with `≡[[]]`. The exact source
execution and trace-safety proofs hold for every scalar backend and both source
directions. The complete-grid theorem retains every original `Request`, every
Tagged input including nonfinite words, all Q/K output cells, safe successful
execution for both schedules and both exact exterior raw-cell frames. No
floating rewrite rule or new numerical validation obligation is used.
`Misc/NativeCorrect.lean` retains the independent source-to-native word formula,
and `Correct.lean` retains the independent Real forward rotation polynomial.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/Audit.lean`
- `Misc/NativeCorrect.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
