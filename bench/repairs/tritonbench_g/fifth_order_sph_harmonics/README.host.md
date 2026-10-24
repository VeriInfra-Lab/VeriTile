# Checked fifth-order spherical-harmonic repair

This separate FP32 implementation preserves the eleven exact decimal forward
polynomials in the original source and repairs their backward Jacobian transpose.
The original Python is unchanged. This is a first-derivative custom-autograd API;
`mask` retains its original unused behavior.

## Defects and target

The original backward contains 29 inconsistent coefficient occurrences, affecting
21 of 33 partial derivatives. At coordinates `(1,0,0)`, with upstream channel zero
one and every other channel zero, its x gradient differs from the true derivative
by exactly `-1/20000000000000`. `Proofs/Defect.lean` proves this counterexample;
`Proofs/Polynomial.lean` independently proves all 33 derivatives. The target is
the original decimal polynomials, not ideal square-root-normalized harmonics.
All 29 repaired coefficient occurrences round to the same FP32 words as the
original constants. No FP32 coefficient difference or GPU-observed numerical
failure is claimed.

The repair also fixes backward return arity, preserves forward block/column
parameters in backward, and reads the actual two upstream strides. The original
launch helper produced approximately one block per coordinate row. The repaired
wrapper uses `ceil(rows / block_size)` directly. It performs no coordinate,
output, or upstream copies and uses no segmented launcher.

## Checked interface

Use `FifthOrderSphericalHarmonicRepaired.apply(coords, output_tensor=None,
mask=None, block_size=64, col_offset=0)`.

- Coordinates are resolved, strided FP32 CUDA tensors with shape `[rows,3]`
  and exact strides `(3,1)`. Actual data pointers and storage offsets are retained.
- A supplied destination has shape `[rows,width]`, exact strides `(width,1)`,
  the same CUDA device, and `col_offset+11 <= width < 2**31`. It must initially
  have `requires_grad=False`. It is marked dirty and returned by object identity;
  unselected columns keep their original bytes. Autograd may update its version
  and gradient flags. Coordinates and destination dense accessed spans must be
  physically disjoint, including when they share an allocation.
- The default destination has width eleven and requires column offset zero.
- Block size is a strict Python integer, a power of two from one through 1024.
  Column offset is a strict nonnegative Python integer.
- The direct CUDA grid must be smaller than `2**31`. Coordinate/output extents
  and every padded-lane mask offset must fit signed int64. The checker evaluates
  the actual products; there is no arbitrary fixed row-count cap.
- Backward upstream has the exact returned shape/device/dtype. Its two strides
  may be any nonnegative signed-int64 values, including zero. Every active byte
  must be naturally aligned and inside its actual storage. Transposed, sliced,
  and `.sum()` zero-stride upstreams are read directly.
- Backward allocates fresh `zeros_like(coords)`, retains the original three
  zero-gradient loads and `+=` operations, and returns exactly as many gradient
  entries as arguments supplied to `.apply`. `once_differentiable` explicitly
  restricts this API to the first derivative. Ordinary saved-Tensor version
  checks protect the actual saved coordinate object.
- Empty tensors launch no kernel. Each nonempty phase runs under the coordinate
  device guard, which restores the previous device.

Both launches set `enable_fp_fusion=False`. This is an explicit ordered FP32
profile and can differ bitwise from the original compiler's default FMA profile.

## Proof boundary

`Kernels.lean` contains source declarations only. The unchanged original forward
source declaration was moved to the original task's `Kernels.lean`; its old
`Proofs/TypedSource.lean` import path and qualified declarations remain available.

`Correct.lean` independently exposes full-launch Real forward/backward theorems
and exact guard/domain equivalence. Real inputs use a separate arbitrary-Real
carrier and are never encoded as finite floating words.

`FPEquiv.lean` independently exposes returned-object/arity and every output-byte
result against the ordered software FP32 reference. `Proofs/Initialized.lean`
connects the backward reference to original caller inputs and literal zero words,
using successful fresh-allocation storage observations. The only deferred laws
are local reached FP32 literal, add, subtract, multiply, and sign operations.
There is no kernel-result, execution, layout, address, memory, or frame premise in
that numerical domain.

The structural chain executes the complete source, proves masked channel stores,
all program coverage and ownership, and replays those literal stores on an
arbitrary initial byte heap. Four-byte stores commute with the source memory
projection. Input and destination reads agree with the evolving byte heap at
every program prefix. Every actual coordinate, output, upstream, and gradient
pointer is bounded through checked storage metadata; signed masks and complete
modular 64-bit pointer expressions are proved. Initial destination bytes and
unselected columns are not replaced by canonical values. Allocator and
version-indexed Tensor observations model ordinary successful Torch API behavior.

## Validation

- `python test_coefficients.py`: eleven unchanged forward polynomials; all 33
  repaired exact derivatives; 29 coefficient replacements with unchanged FP32
  words; the exact original counterexample; pinned original source hash.
- `python test_revision.py`: 84 actual Triton CPU source cases, 16 actual Torch
  autograd cases, 18 guard rejections, supplied output identity and padding,
  empty inputs, arities one through five, saved-coordinate mutation detection,
  and direct-launch metadata controls beyond `2**31` rows. The only CPU adapter
  replaces CUDA residency/device context checks; kernels and launches are actual
  Triton interpreter execution, with no launcher compatibility adapter.
- `python check_cuda_lowering.py --output-dir /tmp/fifth-order-checked-lowering`:
  20 offline SM90 compilations across both phases, blocks 1/64/128/1024, the
  largest admitted row width, and i32/u32/i64/constant upstream stride ABIs.
  PTX uses 64-bit addresses and separate non-FTZ FP32 operations.

No CUDA device execution or numerical CUDA conformance was performed. Offline
compilation and CPU execution do not discharge the local numerical laws.
`source-binding.json`, `validation.json`, `coefficient-validation.json`, and
`cuda-lowering.json` record current source/evidence bindings.
