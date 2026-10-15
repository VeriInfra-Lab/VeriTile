# Triton attention: full accepted source-stage values

This addition proves the complete accepted preprocess, forward and backward
source stages from independently supplied initial memory. It leaves the old
`TritonAttention.lean` file and its four public specification texts unchanged.
It does not prove that the old ideal-real attention formula is the result of
actual floating-point execution. In particular the old
`R.round .fp16 = id` hypothesis remains part of that historical theorem; none
of the new value contracts uses it.

## Accepted range and source correspondence

* Preprocess: arbitrary row/head tile dimensions, arbitrary initial payloads,
  row offsets `pid0 * rows + i`, `DO / L`, `Out * (DO / L)`, axis-1 reduction,
  both complete stores and frame. The two output regions are distinct and the
  output offsets injective, as in the old accepted theorem.
* Forward: arbitrary accepted contiguous-head layout and program IDs; complete
  raw loop `range(0, (pid0 + 1) * BLOCK_M, BLOCK_N)`, causal mask, actual
  boundary masks, all online maximum/denominator/numerator updates, and Out/L/M
  stores. `BLOCK_N > 0` and divisibility of the stop by BLOCK_N are retained.
  Output regions are distinct and output offsets injective. The theorem also
  handles out-of-bound lanes using the supplied typed `undef` oracle, so it
  requires neither the old in-bounds restriction nor `undef = 0`.
* Backward: arbitrary accepted contiguous-head layout, arbitrary head/batch
  strides and initial memory, all `num_block` outer iterations and every
  inner query block from the key-block index to `num_block`. It includes all
  QK/p/dp/ds recomputation, repeated DQ read/add/write, DK/DV accumulation,
  actual masked output stores and frame. Only BLOCK_M must be positive.
  The independent recurrence tracks memory effects, so even aliases are
  interpreted rather than excluded or replaced by an output premise.

`FPShape` proves the three typed bodies erase to the complete old accepted
source bodies. This is a source/control bridge, separate from the value proof.
As in the accepted old backward port, BLOCK_N is an ignored slot: both score
axes use BLOCK_M, and query pointers are rebuilt for each outer block instead
of expressing the Python signed pointer rewind. The Python host uses
BLOCK_M = BLOCK_N = 128. This package does not claim that arbitrary independent
Python BLOCK_N values or arbitrary layouts outside that port are covered.
Ignored old parameters are universally quantified in the projection bridges.

`FPSource` exposes physical region types through DTYPE and OUT_DTYPE. OUT_DTYPE
is instantiated as `.fp32` for L/M/Delta/DQ; Q/K/V/Out/DO/NewDO/DK/DV use DTYPE.
It exposes implicit native storage conversions at the actual stores. Forward
Out and backward DK/DV retain the original explicit `.to(fp16)` before the
store conversion. Preprocess DO/L and Out operands retain their fp32 register
casts; DQ's fp32 load/register conversion is explicit. The `.real` tag in a
register denotes the frontend carrier under recorded fp32 compute precision,
not fp64 storage or mathematical real arithmetic.

## Proof and software roles

Six public specifications distinguish the three arbitrary-backend source
contracts from three concrete-scalar profiles. The successful execution,
independent memory recurrence, readbacks and frame are parameterized over the
whole accepted loops; the fixed examples do not establish their range.
`preWrite`, `forwardWrite`, and `backwardFinalMemory` pass the independent
reference check excluding the source kernel, exec, run and step.

The software carrier is a genuine 32-bit word. fp16 and bf16 cells use the low
16 bits; their conversion decodes that format and rounds directly to the
target format. It never widens to fp32 and then silently treats narrowing as
identity. fp32 scalar add/sub/mul/div use the existing IEEE-value software
operations, and maximum uses maxnum including NaNs and signed zero. Reduction
plans contain explicit trees and a proof that their leaves are exactly the
axis elements; a nonempty reduction does not introduce an extra zero seed.
These declared semantics have no finite-input or non-NaN precondition.

Preprocess is closed software for each of fp16/bf16/fp32 storage: its result
is proved independent of every `External` callback. Forward and backward have
partial scalar closure and remain **backend value contracts**: `hostLiteral`
provides the Real host scale's explicit bit conversion except the declared
canonical +0/1 software constants, `exp`
provides each exponential result, and `dot` receives operands, dimensions,
storage/output dtypes, compute context and DotOptions. Unsupported f8 casts
remain `otherCast`; the generic backend theorem over all six FloatDType tags
is not native support for all those tags. `.real` is an abstract channel.

The accepted source port represents `sm_scale` by `ℝ`, rather than an
independent runtime floating word. The new contracts retain that full old
parameter range; they do not restore distinctions already erased by that
port (host -0 versus +0, or host NaN payloads). The software profile maps Real
0/1 to canonical fp32 +0/1. Binding arbitrary native host bit patterns requires
a runtime-word source interface in addition to the pending ABI relation;
no claim covering such lost distinctions is made by these source theorems.

The backward source exponential has context `none`, with its subtraction,
multiplication and storage-to-register cast inside explicit fp32 scopes.
Forward exp and scalar operations have `some fp32`. This distinction is
retained in the callback signature and must be bound at the actual native
site. No equality between these profiles is assumed.

Nonempty software witnesses cover a two-column preprocess (NewDO = [2,3],
Delta = 5), two forward KV blocks (Out = half 1.5, L = 2, M = 0), and two
backward key blocks with three inner iterations (DV = [2,1], DK/DQ = 0).
Their dot is explicitly sequential fp32 multiply/add after source-format
conversion; the reached exp values are exp(±0)=1 and exp(-Inf)=0. Unreached
exp inputs have a stated NaN extension. They are supporting examples, not a
general exponential implementation or tensor-core model.

## Native evidence and remaining obligations

The offline probe extracts all three original Python kernel bodies with
identical Python ASTs, without candidate source repairs. Triton 3.8.0,
CUDA SM90, BLOCK_M = BLOCK_N = 128, head = 32, fusion false/true:

* All six fp16/bf16/fp32 preprocess configurations compile.
* Both fp16 forward and both fp16 backward configurations compile.
* Eight bf16/fp32 forward/backward configurations fail: the original explicit
  fp16 weights multiply a differently typed V/DO operand. The formal generic
  callback does not turn those rejected native profiles into valid kernels.

These are ten offline successes and eight recorded failures, with no GPU
execution. Native artifacts, source/options and hashes are in the package's
`native-probe/receipt.json`.

Forward PV and backward gradient accumulation are compiled to dotAcc graphs,
even with `enable_fp_fusion=false`. Those graphs absorb the existing
accumulator; the source contracts separately compute dot then add. The
source theorem does not prove their universal bit equality. An independent
compiled-graph contract and native dot accumulation binding remain required
for a native end-to-end result. Compiler removal of a source zero-add also
needs its exact signed-zero/NaN policy considered. These are graph/semantic
boundaries, not laws to assume universally from a few samples.

With default fusion enabled, PTX additionally contracts preprocess product /
first reduction-add into FMA, forward denominator rescaling / addition into
FMA, and backward `qk * scale - m` into FMA. Fusion-disabled profiles have no
scalar FMA, but still have the dotAcc differences above. The preprocess closed
software theorem is therefore a source-graph result; it is not automatically
the default-fusion native graph. The observed native division is
`div.full.f32`, and exp lowers through `ex2.approx.f32`. Instruction excerpts
and counts are retained separately.

Remaining native bindings are specific: host scale ABI conversion; exp
lowering/approximation and context; dot input precision, reduction/accumulation
schedule and exceptional values; reduction trees; scalar division versus the
native divide lowering; cast/FTZ/NaN/signed-zero policies; and the observed
compiled accumulator graph. Valid machine addresses/launch dimensions are
inherited input constraints, not new numerical experiment obligations.
No CUDA result or universal backend/native correspondence is claimed here.
