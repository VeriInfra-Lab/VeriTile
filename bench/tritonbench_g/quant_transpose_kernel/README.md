# quant_transpose_kernel

The current [checked revision](../../repairs/tritonbench_g/quant_transpose_kernel/README.md) preserves this source and proves the native arithmetic refinement from explicit finite-index guards. The accepted original outcomes and wider conditional results remain below.

The original Python wrapper leaves input precision unspecified; this task selects
FP32, matching all four source examples. Its output is explicitly `torch.int8`.

- `Kernels.lean` contains source surfaces, typed kernel slices, grouped schedule
  and wrapper metadata shared by the mathematical and floating proofs.
- `Correct.lean` contains `QuantTransposeReal.quantize_global_transpose_real_source_outcomes`,
  an independent Real specification and proof for the complete wrapper.
- `FPEquiv.lean` contains the existing finite/native structural acceptance,
  FP32 llrint/int8 stage and pre-rounding FP32 stage contracts.
- `Proofs/` contains the supporting host, grid, scalar and memory proofs.

Real and floating scalar interpretations share `QuantizationSource.scalar`
from `quantize_global/Kernels.lean`, preserving `127 * (input * inverse)`, then
llrint64 and modular int8 narrowing. The Real proof also reuses that example's
mathematical memory and host helpers. Its task-level import closure contains no
floating proof. Zero reciprocal and out-of-range llrint are mathematical undefined
observations; no native exception or rollback is asserted.

The two identical autotune entries, partial-group row rotation, actual strided
input, source-ordered host failures and output/absmax tuple are preserved.
Floating numerical primitive relations remain explicit; the layout migration
does not establish native numerical conformance or change acceptance status.
Historical entry files remain compatibility imports with unchanged declaration
names. Their path/hash-bound metadata must be migrated before registration.
