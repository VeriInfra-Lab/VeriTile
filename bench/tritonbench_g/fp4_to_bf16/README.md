# fp4_to_bf16

The original unscaled kernel is now imported through the shared Triton DSL. The software bit-semantics theorem covers packed uint8 e2m1 input to bf16 output, all active bytes and tail masks in its declared profile. `tl.interleave` is supported in this implementation.

Public proof entries: [FPCorrect.lean](FPCorrect.lean).

Native GPU correspondence remains unestablished. Offline compilation and Lean proofs have separate meanings; current scope and experimental obligations are recorded in the [previous 184-task snapshot](../../review-checkpoints/2026-10-14/next-value-repair/final-acceptance/tasks-184.tsv) and [experiment handoff](../../review-checkpoints/2026-10-14/next-value-repair/native-experiment-handoff/EXPERIMENT-HANDOFF.md).

The installed [N21 runner](../../../experiments/floating_point/value_validation/fp4_unscaled/README.md)
now has a [v2 supplement](../../review-checkpoints/2026-10-14/next-value-repair/fp4-unscaled-experiment-runner-v2/README.md):
26 CPU checks and two original-kernel SM90 offline compilations passed; the
GPU attempt is `NOT_RUN` with zero launches. Its 21 configurations use block
16/512, exact original JIT bodies, all 256 packed bytes, nonzero pointer offsets,
tail masks and full input/output allocation checks. The oracle compares raw
bf16 bits with the high nibble first. Physical aliases and internal operands
are not covered. This delivers a concrete collector and reference, not GPU
validation, and does not close N21 or the 37 whole-job native work items.

The previous import-status note is preserved in the [documentation history](../../review-checkpoints/2026-10-14/next-value-repair/documentation-update/before/bench/tritonbench_g/fp4_to_bf16/README.md).

A new [host-grid theorem](HostCorrect.lean) composes the actual admitted kernel
for every program in the original `ceilDiv(numel, 512)` launch. It models a
contiguous packed-uint8 tensor of any nonscalar shape, the fresh bf16 allocation
and doubled final dimension, all active output cells, arbitrary initial output
memory, tail masks, and the complete memory frame. Each output is proved to
decode to its signed exact rational e2m1 value, including both zero signs. The
reference reads only the original input bytes; no source execution or supplied
output is used to define it. The original per-program alias theorem is retained.

This addition proves the sequential logical launch in the existing structural
semantics. Python allocation/shape lowering, native index-width specialization,
physical tensor provenance and GPU concurrent-launch refinement are still
external boundaries. The original wrapper checks contiguity and CUDA placement
but does not assert an input dtype; this theorem states the documented packed
uint8 domain explicitly and does not claim arbitrary torch dtypes. It includes
zero-element logical grids without asserting native zero-grid launch legality.
The new modules are registered separately from the frozen checkpoint contracts;
the historical acceptance catalogue has not been relabelled.
