# Preserved original-source findings

The selected verification target is `int8_quantization_repaired.py`. The original
`bench/tritonbench_g/int8_quantization/int8_quantization.py` remains unchanged;
none of the repaired theorems asserts equivalence to its defective wrapper.

The independent finite mathematical target partitions each input's flattened
batch into consecutive `block`-row groups. For Q, transform each element by
`channels^(-1/2) * 1.44269504`; for K, leave it unchanged. For each block let `m`
be the maximum absolute transformed value, with padded lanes zero. The scale
is `m/127`; a zero maximum produces zero bytes. Otherwise the INT8 target is
nearest-away rounding of `127 * transformed / m`. The total real target also
zeros a whole block when a transformed value exceeds FP32's finite range. The
FP32 graph separately retains all primitive rounding, underflow, invalid-value
classification, clamping, and conversion steps; it is not equated with real
arithmetic.

Concrete original counterexamples:

* Q shape `(1, 2, 1)` and K shape `(1, 3, 1)` return a K output allocated by cloning
  Q's `(1, 2, 1)` output, and launch K with Q's length. The third K element is
  neither allocated nor processed. Independent shapes are admitted by the repair.
* Q shape `(1, 3, 1)` and K shape `(1, 2, 1)` launch three K rows against a two-row
  input: the third row is an active out-of-bounds read.
* An all-zero full block sets `scale=0`, then evaluates `0/0`, and converts the
  NaN result to int8. The repair's safe denominator and usability branch give
  exact zero bytes and a zero scale.
* A partial last block has masked-off loads with no `other=0`. Those undefined
  lanes enter the maximum reduction. The repair explicitly supplies zero padding.
* A strided view such as a transpose of the final two dimensions can retain its
  rank while the original pointer formula ignores its strides. The repair rejects
  noncontiguous inputs before allocation.
* Q and K on different CUDA devices still allocate K scales on Q's device and
  clone Q's output for K. The repair allocates each pair on its own input device.
* Non-power-of-two channels or block sizes are passed directly to `tl.arange` in
  the original. The repaired flattened tile uses a power-of-two width and two
  masks, admitting non-power-of-two lengths, channels, and blocks.
* NaN, infinity, or Q-prescale overflow reaches undefined/target-specific float
  to int conversion in the original. The repair zeroes an invalid whole block.

The original can also overflow narrow offset arithmetic. The selected source
widens program IDs and lanes before arithmetic, chunks the grid at `2^31-1`,
and checks input and scale extents below `2^61` elements. Native pointer proofs
retain byte scaling and bounds for every active lane. No old source theorem has
been relabeled as a theorem about the selected repair.
