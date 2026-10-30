# Integer and row-scaled matrix multiplication

This revision handles both `int_matmul_kernel` and `int_scaled_matmul_kernel`.
It preserves INT8 inputs, INT32 accumulation/output, and the scaled kernel's
FP32 suffix. No tensor copies, extra dtypes, general-stride implementation,
allocator, or autotuner are introduced; the caller still supplies Config and C.

## Checked contract

A is M×K, B is K×N, and C is M×N. All are rank-two strided CUDA tensors on
the same device, default-contiguous, and without unresolved negative flags.
A/B are INT8 and C is INT32. The scaled API additionally requires an M×1 FP32
scale tensor with the same layout/device/view properties. All INT8/FP32 values
are admitted. Contiguous storage offsets, read-only input overlap, and unusual
irrelevant singleton strides are included. The launch uses equivalent canonical
dense strides; it does not pass a singleton's unused stride through source
integer arithmetic. C must not overlap an accessed A/B/scale byte window.

Config must be the supplied Config class with exactly BLOCK_M, BLOCK_N,
BLOCK_K, GROUP_M and builtin integer fields. BM/BN are powers of two at least
16, BK is a power of two at least 32, each pairwise tile product is at most
2^20, and GROUP_M is positive. Warps are one of 1/2/4/8/16/32, stages are
positive, and CTAs equal one. Ordinary successful compilation/launch resource
requirements still apply; these guards are not a proof of device resources.

For nonempty output, set `gm=ceil(M/BM)`, `gn=ceil(N/BN)`, `gk=ceil(K/BK)`,
and `pm=gm*BM`, `pn=gn*BN`, `pk=gk*BK`. Each of the following must be below
2^31:

```
M+BM-1, N+BN-1, K+BK-1, GROUP_M*gn, gm*gn,
pm*K+pk+BK, (pk+BK)*N+pn, pm*N+pn
```

These are source-specific cdiv, grouped mapping, padded coordinate, and
pointer-after-final-advance envelopes. They are actual Python checks and named
Lean premises, not hidden allocation/storage laws. Each nonempty tensor's
actual data pointer must align to its element size and its byte-window end must
be at most 2^64. C's window must be separate from the read-only windows.
M=0 or N=0 returns C before pointer/envelope queries; K=0 is also supported.

`Correct.python_checks_iff_preconditions` proves exact dynamic admission;
`Proofs/Contract.lean` proves the typed check equivalence to `InputContract`.
Tensor/config classification rejects unsupported objects before their metadata
is treated as a valid tensor/config. The logical contract does not depend on
the order of the guards.

## Source repairs and the overflow witness

The unscaled block-pointer loads use explicit zero padding. The scaled wrapper
sets EVEN_K from divisibility by BK rather than parity. False `multiple_of` and
`max_contiguous` assertions on modulo-wrapped ragged rows/columns are removed.
The scale/output suffix uses the supplied strides, which the checked host binds
to canonical dense strides. Empty output returns without launching.

Both kernels use one explicit inline PTX `add.u32` for each INT32 carry update.
The dot remains INT8→INT32 and is seeded at zero. This additional minimal repair
is required by observed Triton 3.7.1 SM90 lowering: for BM=BN=16, BK=32, warps=4,
stages=2, `acc += tl.dot(a,b)` was fused into
`mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` carrying the previous
accumulator. With M=N=1, K=131072, and all A/B values -128, the exact sum is
2^31. That fused recurrence saturates to 2147483647; the intended modular32
result is -2147483648. The input passes the source envelope. This is an
instruction-semantics counterexample from the observed lowering, not a GPU run.

The candidate keeps the dot's zero seed and a separate `add.u32` in both TTIR
and PTX. Checked tile products imply BK≤65536. A signed8 product has absolute
value at most 16384, so every tile prefix has absolute value at most 2^30.
`Proofs/SaturatingTile.lean` proves saturation inactive for any partition of
these tile products into MMA instructions, proves equality to the typed
`wordDot`, and proves the explicit add implements the modular carry update.
`IntegerInstructions` executes those bounded saturating groups and explicit
word adds; both `CheckedHost` and `CheckedNative` use that executor. Its
program/grid refinement is composed into the public result. This resolves
the split/fused integer issue structurally. No overflow input
restriction or deferred whole-dot equality is used. The candidate is separate
from the unchanged benchmark source.

## Independent results and proof chain

`LogicalInputs.integerExpected` is the full K sum of products of the original
logical signed8 tensor values, projected modulo 2^32 and read as signed32. It
contains no tile schedule, kernel invocation, or execution result. This is an
integer-format mathematical specification, not an unbounded Real-output claim.

`LogicalInputs.scaledExpected` applies FP32 promotion to that signed32 value,
multiplies by the original logical FP32 row scale, and performs the signed32
conversion. Undefined software conversions retain the local conversion port.
The strongest native public API uses `CheckedNative.Instructions.signed32`,
which returns a `BitVec 32`; even undefined conversion results therefore retain
the output word type. No finite-scale or defined-cast admission guard is added.

The public entries are:

- `Correct.int_matmul_checked`: full guarded instruction/byte call, independent
  modular mathematical result, and actual C storage readback.
- `FPEquiv.int_scaled_matmul_checked_fp32`: full guarded software-profile call,
  including all scale words and an explicit conversion fallback.
- `FPEquiv.int_scaled_matmul_checked_fp32_if_local`: complete finite native
  instruction/byte call transported through three scalar numerical laws at
  nearest-even FP32 with input/result flush disabled, including actual C readback.

Each result returns the exact supplied C object, states its shape/device/INT32
and contiguous/resolved metadata, gives every output coordinate, and preserves
all logical memory outside those C cells. The mathematical and native endpoints
also take ordinary `Laws .int32 c` and the initial C-word binding to the common
byte reader. They execute actual four-byte stores and prove exact word readback
through C’s original data pointer and storage offset, all four bytes of every
result, and every exterior byte. The empty branch preserves every byte and the
exact C. No output allocation or copied-input snapshot is assumed.

`Kernels.lean` owns source definitions and imports no proof module. Both reader
entries reuse the historical whole proofs through `Proofs/Whole.lean` and
`Proofs/HostBindings.lean`; FP does not import the mathematical reader.
Historical defined-cast results remain available with their original premises.

`FiniteIndex` proves signed BitVec32 grouping, coordinates, descending K masks,
offsets, terminal advances, and checked byte-address extension. `FiniteExecution`
executes the finite masks, offsets, loop, integer arithmetic, and stores.
`FiniteSource` connects it to the complete typed source interpreter.
`FiniteGrid` composes source execution with grouped-grid ownership and coverage.
`CheckedHost` derives launch conditions from the exact checked contract and
composes the real host branches. `LogicalInputs` connects source reads to the
original logical tensor values using a common physical byte reader, including
read-only storage aliases. `CheckedNative` transports the complete native source
through only local scalar FP laws. `OutputBytes.render_initial` proves the
initial representation is the original physical memory, and `render_write`
proves one logical source store equals four little-endian byte stores.
`ByteExecution` composes that store simulation through the exact finite masks,
offsets, tile fold and full grid. `ByteHost` composes the checked host and empty
branches and proves physical C readback and the exterior byte frame. Its
`output_addresses` derives alignment and UINT64 bounds from the explicit guards;
`input_frame` derives physical input-byte preservation from checked separation.
These are memory semantics proofs, not proofs of CUDA runtime internals or
machine instruction scheduling.

Ordinary trusted boundaries are Tensor metadata/contiguity/storage semantics,
typed primitive and PTX instruction semantics, successful compilation and
race-free launch, and typed load/store representation. The Tensor laws contain
no source-specific index envelope or assumed matmul result. Python/Lean binding
is checked by a focused source audit, not a verified Python compiler.

The three deferred numerical conditions are signed32→FP32 nearest-even
promotion on all signed32 inputs; FP32 multiplication at the pinned profile on
all FP32 words; and FP32→signed32 toward-zero conversion where the independent
conversion is defined. Undefined conversions keep the actual native word.
The integer reduction and saturation issue, masks, offsets, loop coverage,
host branches, and return/frame results are proved rather than deferred.
GPU numerical conformance remains unvalidated. Shared launcher compatibility
and historical cleanup remain the separate `bench/TODO.md` items.

## Validation

All 47 retained/new Lean modules pass the final dependency-ordered Lean 4.29.0
rebuild and standard-base axiom audit, with no `sorry` or new axiom.
The original Python SHA256 remains
`f54f5d858707e1a132f958abc72e946d366be1c3df9881df360bc7838acbf92f`.

The focused check is:

```
python bench/repairs/tritonbench_g/int_scaled_matmul/revision001/check_contract.py
```

It checks the exact reviewed AST changes, actual guards, canonical launch
arguments, original Config behavior, and source/proof import direction.
There are 61 rejection cases before launch/mutation and 50 CPU-interpreted calls:
K tails including the original parity bug, group tails, empty/K=0, offsets,
singleton strides, read-only aliases, output overlap, pointer alignment/window
limits, the source-envelope boundary, all-word scales, output identity/frame,
and full-K modular overflow. Triton's interpreter executes real CPU backing
storage with a mocked CUDA context. Its unsupported inline assembly is handled
only by an exact wrapping-add adapter for the audited `add.u32` instruction.
This is not a GPU numerical experiment.

`check_cuda_lowering.py` performs eight offline SM90 compilations: two old
fusion witnesses and six repaired profiles (both kernels, both EVEN_K paths,
16×16 and 64×64 tiles). It checks zero-seeded INT8 MMA, separate modular adds,
and scaled `cvt.rn.f32.s32`, `mul.f32`, `cvt.rzi.s32.f32`. Both mma.sync32 and
wgmma32 profiles were observed. The saved IR/PTX goes to
`/tmp/veritile-int-scaled-lowering-audit` by default. CUDA execution is unavailable;
these checks establish the inspected lowering, not universal compiler correctness.
