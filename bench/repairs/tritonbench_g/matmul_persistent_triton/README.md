# Persistent matmul repair

Read Kernels.lean, Correct.lean and FPEquiv.lean. This separate candidate fixes
false clamped-tail hints, K=0 uninitialized output and source integer/pointer
formation hazards. It retains both actual FP16/E4M3 profiles, fused dot with
FP32 supplied accumulator, K order, masks, output conversions and default
fusion. The independent Real target is ordinary matrix multiplication; empty
outputs and K0 are covered and every caller allocation is preserved.

The complete contract supports raw/resolved matching rank-two CUDA tensors,
nonnegative strided/broadcast backing, nonzero offsets and A/B aliasing.
`matmul_persistent` now checks that supported metadata before any device
query or allocation, including natural element alignment for nonempty inputs.
`Checked.admitted_input` derives the legacy input contract from those guards
and ordinary valid-storage/live-object facts; both public entries provide
`checked_complete`. No logical negative/conjugate resolution copy is added.
Stored negative values are allowed. Ordinary allocation derives fresh C and
its exact shape/stride/offset; empty C row stride is max(N,1). No arbitrary
small-shape/native-safe restriction is added. Empty views do not need a
dereferenceable offset; backing/pointer checks are conditional on nonemptiness.

Proofs/Bounds and ControlBounds establish actual integer/address bounds;
Execution/OutputCoverage/GridOrder derive evolving-memory grid results;
Real/HostReal assemble the independent full formula and frame. Host handles
compatibility, A.device property lookup, allocation, empty/K0/positive paths
and exact returned allocation. Original schedule/K mathematics is reused.

FPEquiv uses one actual resident artifact/AST/ABI/tape and exact reaching IDs.
Its only numerical callbacks are output cast and a single fixed ISA instruction.
The explicit catalog covers WGMMA and MMA variants; recorded offline evidence
is SM90 and other target instruction mappings are not asserted. DotPlans proves
complete four-fragment K coverage. CompilerCorrect is an explicit ordinary
implementation boundary to the fixed complete source interpreter, not a new
proof of the compiler or a generic typed-DSL interpreter translation.

All 32 restored closure/checked modules compile with standard-base axiom audits.
`python bench/repairs/tritonbench_g/matmul_persistent_triton/check_contract.py`
passes 22 executions of the actual repaired source through a CPU facade,
including both dtype profiles, ragged tails, persistent iterations, short final
groups, non-current devices, offsets, broadcasts, input aliases, empty outputs
and K=0. Nine unsupported-input cases reject before host side effects. These
controls use diagnostic NumPy dot and Torch casts; GPU arithmetic is not
validated. Historical acceptance and numerical scope are unchanged; no current
acceptance record is written by this task owner.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `f6f67cac011b1ec78872fe4a0282e59ddb9e9c05f9889e1aad2657b52b206755`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(a.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
