# Copied Python function bodies

`Kernels.lean` contains all 73 forward and 122 backward statements of
`fifth_order_sph_harmonics_repaired.py`, and the complete corresponding
power-reuse candidates from `candidate_kernels.py`. `jit-correspondence.json`
checks all four bodies. Original casts, coefficient assignments, masked
loads/stores and initial-gradient accumulation remain in their source order.
Call signatures and invocation bindings supply pointer dtypes, constexpr
parameters and runtime strides outside the pasted bodies. Runtime strides use
the generic integer carrier; the explicit source int64 conversions are retained.

Ordinary floating constant assignments create FP32 scalar tensors. The shared
frontend records this precision in execution, including tuple assignment;
inline weak constants remain inline. Exact negative zero is retained as an FP32
bit pattern instead of being collapsed to rational zero. A finite literal is rejected if direct
exact-decimal-to-FP32 conversion would disagree with Python's intervening
binary64 rounding. This follows Triton's ordinary assignment conversion in
[`visit_Assign` / `_sanitize_value`](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/compiler/code_generator.py)
and [`to_tensor_type`](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/semantic.py).

The execution proofs interpret the actual copied programs. Forward's memory
connection reuses the independently specified eleven channel stores; it does
not identify the complete register state with an older unsigned-index view.
Backward likewise connects actual signed indices and stores to the existing
ordered polynomial specification. Neither connection erases floating casts or
assumes an arbitrary floating-point conversion is an identity.

Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

Current FPEquiv pair (2026-10-21): the complete original grids are compared
with the new `forwardPowersSource` / `backwardPowersSource` candidates. No prior
alternate implementation existed. The candidates reuse the identically rounded
left-associated powers: `x*x*x*x*x`, `x*x*x*x`, `x*x*x`, `x*x` become successive
assignments of square, cube, fourth and fifth power; y and z are analogous.
Forward removes eighteen repeated multiplications per lane and backward removes
nine, without changing any multiplication tree, precision or rounding point.
This is a source operation-count reduction; no measured speedup is claimed.
`FPEquiv.kernel_equivalent` proves actual original/candidate `≡[[]]` with shared
`ProgramIO.Equivalent`: both complete grids are safe, succeed, have identical
active typed outputs and preserve both exterior memory frames. No numerical
relaxation or experimental rule is used. Native backward accepts arbitrary old
gradient words, retaining its original read/add/store behavior.

# Fifth-order spherical-harmonic kernel proof

The authoritative source is `fifth_order_sph_harmonics_repaired.py`. Its forward
computes the original eleven exact decimal degree-five polynomials. Backward
computes their Jacobian transpose and retains the three initial gradient loads
and source `+=` operations. `Correct.lean` uses explicit logical IO and the shared
`ProgramIO` judgment. `FPEquiv.lean` proves the full two-program power-sharing transformation;
optional source-to-native theorems are in `Misc/NativeCorrect.lean`.

The original backward has 29 inconsistent coefficient occurrences affecting 21
of 33 partial derivatives. At `(1,0,0)` with upstream channel zero equal to one
and all other channels zero, its x-gradient error is exactly
`-1/20000000000000`. The independent monomial table in `Proofs/Polynomial.lean`
proves all 33 derivatives; `Proofs/Defect.lean` proves the counterexample. All 29
repaired constants round to the same FP32 words as the originals. This exact
Real defect is not a claim of an observed GPU or FP32 coefficient difference.

## Prepared tensor domain

Coordinates are resolved FP32 CUDA `[rows,3]` tensors with exact strides `(3,1)`.
The destination is `[rows,width]` with strides `(width,1)` and an eleven-channel
slice satisfying `column+11 ≤ width < 2^31`; the default is width eleven and
column zero. The accessed coordinate and destination spans are disjoint. Actual
storage offsets are retained. The supplied destination is initially not marked
as requiring gradients.

Every power-of-two block size from 1 through 1024 is supported, with default 64.
The direct grid is `ceil(rows/block)` and must fit signed int32; the actual
coordinate/output extents and padded-lane mask expressions must fit signed
int64. No artificial row cap is imposed. Empty inputs run no blocks.

Backward upstream has the exact output shape and any nonnegative signed-int64
row and column strides. Dense, transposed, sliced and zero-stride views are read
directly. The prepared gradient is dense `[rows,3]`, with disjoint backing and
actual active values equal to positive zero. Padding is arbitrary. Every active
byte is aligned, inside its observed storage and below the 64-bit address limit.
The source's unused `mask` argument is preserved.

`KernelTensor.Binding` connects each queried view to an actual shared
`VeriTile.Torch.FP32.Tensor`, its `TensorLaws`, metadata, storage and physical
index map. `KernelEntries.Forward` and `Backward` package those ordinary facts;
they contain no execution or output premise. The upstream binding does not
require contiguity. Output theorems read through actual Tensor physical indices,
and native byte theorems use actual Tensor storage addresses.
`Misc/TensorBridge.lean` also exports the native mask and pointer binding helpers.
Neither primary entry imports this optional module.
The shared FP32 storage counts complete words with `elements = byteCapacity/4`;
unused trailing bytes do not impose a new divisibility restriction on admission.

## Public statements

`IO.lean` exposes both actual source grids, their FP32 channels, exact strided
windows, masks and launch order. Forward reads `[rows,3]` coordinates and writes
the eleven-channel destination slice. Backward reads coordinates, all eleven
strided upstream channels and the prepared `[rows,3]` gradient; that last input
aliases the output. Every program of the original finite grid is retained,
including the empty-grid case.

The main interfaces take `Parameters` (dimensions, block geometry and strides)
and the existing eleven-channel fit bound. Independent polynomial/Jacobian and
concrete native formulas are named in `IO.lean`. Real backward's logical domain
requires exactly the active initial gradient values to be zero. Native backward
retains arbitrary initial gradient words in its source-ordered addition.
There are no raw states, address heaps or execution predicates in these entries.

`TensorBridge.real_forward`, `real_backward`, `native_forward`, and
`native_backward` retain the complete prior Tensor statements and use the same
source execution foundation through `Misc/KernelRealResults.lean` and
`Misc/KernelNativeResults.lean`. The existing input-reference and output-index
lemmas connect those raw statements to actual Tensor coordinates. Native backward's
raw target retains its initial gradient loads; its Tensor bridge specializes
them to the same prepared positive zeros as before.

`Correct.forward` and `Correct.backward` each apply one complete IO composition
theorem in `Proofs/IOCorrect.lean`. Shared `ProgramIO.Implements`/`ImplementsOn`
connect logical input bindings, actual source access safety, successful full-grid
execution, all eleven polynomial or three Jacobian-transpose outputs, and the
untouched tagged-memory frame. `Proofs/IOSchedule.lean` identifies the shared
launch schedule with the existing complete source-grid execution; `IOResults`
reuses that result rather than re-proving execution. `KernelSafety` and
`IOSafety` discharge actual load/store safety for each complete phase.

The ordinary `FPEquiv.forward` and `backward` theorems in `Misc/NativeCorrect.lean`
each contain one source-to-native observation statement and a one-call composition proof.
They prove full source execution
and every native word against the existing independent source-ordered FP32
target; they do not introduce a second kernel or numerical transformation. The optional
`TensorBridge.forward_bytes` and `TensorBridge.backward_bytes` in
`Misc/TensorBridge.lean` additionally prove all four bytes at each actual destination
Tensor address and every byte outside the store set. Native backward retains
the initial positive-zero addition, including its signed-zero behavior.

`KernelScalar` fixes concrete scalar implementations and discharges the old
primitive-local numerical conditions internally. Constants have exact rational
rounding proofs. FP32 add/subtract/multiply, negation, casts and exact loads are
model semantics, including overflow and subnormal behavior. Both actual Python
launches explicitly set `enable_fp_fusion=False`; the selected model uses
separate nearest-even rounding with no FTZ. There is no deferred arithmetic
oracle in a public theorem.

`Native`, `Access` and `Lowering` prove pre-arithmetic PID/lane widening, signed
formed masks, complete modular 64-bit pointer expressions, active storage
bounds and input/output separation. `Grid` proves complete program coverage,
unique active ownership and full frames for every admitted block size.

The boundary is conformance of native Triton/compiler/GPU execution to this
concrete model. CPU interpretation and offline PTX checks provide evidence for
that boundary; they are not an axiom that establishes it. Special floating-point
values follow the declared model, including its NaN policy.

## Evidence

The following historical CPU/PTX measurements were recorded in the optional
`.veritile-cache/task-deliveries/fifth_order_sph_harmonics/` delivery bundle.
They are not a dependency of the Lean source or proof build, and this source
migration does not claim to have rerun them:

- Full normalized source/DSL AST equality: 73 forward and 122 backward
  statements, including all 12 explicit upstream-stride widening casts.
- Exact coefficient/derivative checks: eleven unchanged forward polynomials,
  all 33 derivatives and 29 same-word coefficient corrections.
- Direct canonical Triton CPU source checks: 52 forward configurations and
  208 backward configurations across all eleven blocks, with 205 actual kernel
  launches and eleven empty configurations. Forward matches the source-ordered
  Torch FP32 expression exactly. Backward is compared to FP64 polynomial
  autograd with `atol=3e-3, rtol=2e-4`. Full input backing, destination exterior
  columns and backing, and gradient exterior backing are checked.
- Twenty existing same-source SM90 offline profiles were copied and rechecked,
  covering both phases, blocks 1/64/128/1024, maximal width/high column and
  manually supplied `ASTSource` signatures with i32/u32/i64/constant stride ABIs.
  These profiles are not JIT argument-binding traces. All 40 pinned TTIR/PTX artifacts use 64-bit
  addresses and separate non-FTZ FP32 add/multiply, with no FMA instruction.

No GPU execution was performed. The frozen delivery packet records exact source,
proof-object, build-launcher, dependency and evidence hashes.

## Historical host evidence

The previous host/allocator/device/autograd statements are preserved unchanged
in `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean` and `README.host.md`.
Their descriptions of deferred local arithmetic refer to that older interface.
The task-root `source-binding.json`, `validation.json`,
`coefficient-validation.json` and `cuda-lowering.json` are also historical
provenance. In particular, the old public Lean hashes and numerical deferrals
in `source-binding.json`, and the 84/16 case counts in `validation.json`, do not
describe the new public entries. Current proof pins and checks are recorded in
the committed review checkpoints. The historical delivery packet remains optional provenance.
Host return identity/arity, allocator behavior and device guards are outside the
new kernel acceptance boundary; the Python implementation and guards are
unchanged by this proof revision.

The power-sharing connection is in `Proofs/SourcePair.lean`. Its proof-only
statement decomposition is checked against both complete ordinary `triton`
quotations. It proves exact execution equality and trace safety for arbitrary
structural algebras by preserving the FP32 operation trees and commuting only
writes to distinct local registers. `Proofs/IOPair.lean` transports the existing
successful native IO certificate through every actual grid invocation, preserving
the axis-zero-only launch setup, masks, strides, channels and exterior frames.
The main dependency closure does not import `Misc/`.
