# rmsnorm_fused: checked canonical FP32 repair

This task reuses the complete historical FP32 repair, its ordered native
reduction/FMA model, and its explicit source-store-to-byte proof. The original
Python remains unchanged. The repair supports forward inference with contiguous
strided CUDA FP32 input and FP32 vector weight on the same device, zero storage
offsets and no lazy negative/conjugate flags. The final feature width is between
1 and 16384; all leading dimensions and empty leading dimensions are retained.
Epsilon is a finite positive built-in Python float. Nonempty calls require
four-byte-aligned actual input pointers, a CUDA grid of at most `2**31-1` rows,
and an extent within signed-64 byte addressing.

The canonical layout/offset guards state the inherited supported storage domain;
no copies are introduced to support additional layouts. Read-only input and
weight aliases are allowed. The original metadata-only reshape is retained.
The output is the exact fresh contiguous `empty_like` allocation, with the same
logical shape and device. Empty leading dimensions allocate and return it
without launching. Nonempty launches select the input owner's CUDA device.

## Concrete source defects and repair

The original kernel uses the input's reshaped row stride to address the fresh
output and ignores weight stride. For `x = ones(2,4)[:,::2]`, the input row stride
is 4 but `empty_like(x)` has packed row stride 2. The original second output row
therefore stores at offsets 4 and 5 in a four-element result. Dense transposes
can collide; reshape copies can similarly disconnect input and output layout.
The guard rejects unsupported layouts, and the repaired launch passes the
canonical logical stride N, including singleton rows with unusual unused strides.
The row PID is widened to int64 before multiplication. Both loops, masked weight
load, extra input remask, FP32 casts, reductions, sqrt/divisions, multiplication
order, 64 KiB feature limit, block selection, warp heuristic and default fusion
remain unchanged. The source check compares every numerical AST statement.

## Contract and complete results

`Kernels.lean` imports only source definitions. The helper implementation is in
`Proofs/`; public `Correct.lean` and `FPEquiv.lean` are independent. `Contract.lean`
names representation, shape, canonical layout, epsilon, feature, extent, grid
and alignment conditions. `checkInput_iff` proves exact ordered admission,
including Tensor classification before attribute access. `geometry_shape`,
`geometry_size` and `geometry_rows` derive source geometry from the observed
shape. The bounded block function equals the original next-power-of-two/cap
choice over the admitted widths; the Python check exhaustively compares all
16384 possible widths. No unchecked shape, grid or offset callback is assumed.

`Correct.wrapper_correct` constructs complete execution over any permutation of
the full row grid and proves each independent Real output:

```text
y[row,k] = x[row,k] / sqrt(sum(j < N, x[row,j]^2) / N + eps) * weight[k]
```

It returns the exact allocated output descriptor, input shape, FP32 dtype and
owner device, and proves successful context restoration plus every caller cell
outside the output footprint. The formula is independent of kernel execution.
This is the cast-erased Real interpretation, not an equality with rounded FP32.

`FPEquiv.wrapper_fp32` runs the complete typed source to an explicit symbolic
operation graph and evaluates the selected ordered native reduction tree and
FMA schedule. It proves every returned byte, all four bytes per output word,
and every caller byte outside the output footprint, including trailing bytes.
Its independent coordinate expression preserves source operand/cast/load order.
The source-to-byte connection is proved one store at a time, through scatter,
row and whole-grid composition. It is not a whole-output materialization law.
No scalar numerical equality is required for this structural theorem.

The graph permits only actual adjacent multiply/add contractions. Tree and
fusion/operand selectors inspect shapes rather than words or epsilon. Complete
reduction trees include every lane, with explicit neutral padding. No fixed
reduction tree, associativity, commutativity, split-equals-FMA or native-to-Real
law is assumed. NaNs, infinities, subnormals and signed zeros in input/weight
remain admitted as unrestricted raw FP32 words.

## Storage, native addresses and local numerical work

`Host.Preparation` states same-call allocation metadata, freshness and frames,
with no source success or output formula. `Observes` binds those descriptors to
the checked shape, device and actual data pointer. Ordinary live storage spans
start at the exact input, weight and allocated output pointers. `NativeIndex`
proves signed32 PID interpretation, its int64 widening, lane sign extension,
masks, every formed signed64 relative offset, exact native pointer values and
all four active bytes inside the pointer space. Actual allocator output
alignment is an ordinary boundary law. Region identity denotes the same live
allocation used by the explicit byte-store model; physical pointer/storage
representation and faithful Python/Triton launch are ordinary trusted APIs.
There is no verified CUDA/Python/Triton implementation claim.

The seven deferred local numerical families are literal conversion (0, 1, N
and epsilon), FP32 add, multiply, divide, sqrt, identity cast and FMA. The
`reached_primitives` theorem excludes unused generic algebra ports. The separate
`scalar_equivalence` theorem shows how validating those local operations implies
full byte equality, retaining the same native tree/schedule. Their concrete
instruction conformance must address actual reached operands, FTZ/subnormal
behavior, nonfinite words and runtime epsilon conversion. Structural execution,
coverage, shape geometry, native indices, storage and host composition are
proved and are not numerical deferrals.

The host epsilon annotation is the inspected finite binary64 float's exact Real
value; FP evaluation retains its conversion to FP32. Positive source epsilon
does not imply that every FP intermediate stays finite or positive. No such
assumption or unconditional FP-to-Real agreement is made.

## Validation

All 16 Lean source/helper/public modules compile with Lean 4.29.0. The public
whole-call and native-pointer results pass `#axiomsClean`, using only `propext`,
`Classical.choice` and `Quot.sound`. The reused graph/byte helpers also retain
their noncircularity audits. No historical evidence tree is copied.

```sh
python /tmp/veritile-add-example-build.py bench/repairs/tritonbench_g/rmsnorm_fused/Correct.lean bench/repairs/tritonbench_g/rmsnorm_fused/FPEquiv.lean
python bench/repairs/tritonbench_g/rmsnorm_fused/check_contract.py
```

The exact source/guard binding check passes 32 rejections, seven admitted host
cases and 12 CPU Triton-interpreter controls. It covers every block threshold,
rank-one and higher-rank inputs, empty leading dimensions, unusual contiguous
singleton strides, owner-device selection, pointer alignment, NaN admission,
feature/grid/extent boundaries and the original gapped-output counterexample.
Host cases mock CUDA ownership/allocation/context over real CPU Torch metadata;
synthetic huge shapes exercise limits without allocating huge tensors. Exact
interpreter cases use x=±1, weight=2 and eps=3, making every expected output
exactly ±1 and testing address/mask/store coverage across the feature range.
Torch is 2.13.0+cu130, Triton is 3.7.1, and CUDA is unavailable. No GPU execution
or new offline compiler/native numerical validation was performed.
