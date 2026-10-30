# int8_dequant_matmul

kernel_real: columnScale[col] * (rowScale[row] * sum_k(signed8(A[row,k])*signed8(B[k,col])) / 16129) + optional bias[col].

- [int8_dequant_matmul_scalar_cast.py](int8_dequant_matmul_scalar_cast.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every finite M×N cell of the complete grouped tile/split grid; each address retains its independent complete split order, all bias/even/tail branches and exterior frame.

## Conditions

- Shape/Config, Source.Admitted c and selected integer instruction plans; unchanged source configuration domain.
- IOLayout.domain Quote.integerRealOps: split-K C input is the typed FP16 zero; unsplit C is not required as an input.
- The accurate IntegerDotPerAddressProgramIO.ImplementsOnCells judgment retains actual typed input bindings, bounds, locality, source traces, coverage, output cells and frames.

Exact FP program pairs and numerical dependencies are recorded in the [int8_dequant_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
