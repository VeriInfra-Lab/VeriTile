# matmul_dequant_int4

- Name:

kernel_real

Statement:

VeriTile.Spec.Real (ReaderIO.view f p ⊨ fun xs row column =>
      ((Source.nibble (xs.weight row column) row.val -
        Source.nibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column)

- [matmul_dequant_int4_grid_y_checked.py](matmul_dequant_int4_grid_y_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original two-axis dequantization grid, all rows*columns outputs, tails, both FP16/FP32 scale/output profiles, exact exterior frame.

## Conditions

- KernelGrid.Valid is unchanged: positive block rows/columns, outputRowStride=columns, outputColumnStride=1, output disjoint from Q/Z/S regions. Unit column stride is an original premise, not a new input restriction.
- ProgramIO simultaneous logical input binding enforces consistency of repeated packed/grouped addresses; ordinary access bounds remain in the judgment.
- Selected Contract adds exactly the already-selected grid-y check while preserving original guard order/domain.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_dequant_int4 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
