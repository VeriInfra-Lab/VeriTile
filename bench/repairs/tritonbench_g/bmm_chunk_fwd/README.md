# bmm_chunk_fwd

Full logical-K A*B-transpose per returned chunk: signed sequence mismatch, padded row/column positions and the causal upper triangle yield zero.

- [bmm_chunk_fwd_repaired.py](bmm_chunk_fwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete returned dense output, original ordinary main grid and actual causal cleanup launches; empty outputs use the original empty schedule. All original configurations and sequence/causal branches remain.

## Conditions

Public conditions:

- c belongs to Host.configurations
- Parameters retains its checked shape/source metadata
- AlgorithmProgramIO binds only the exact active read windows and signed sequence IDs
- Successful actual main/cleanup execution, source safety, all outputs and exact exterior frame stay in Implements

Contract domain source:

Contract.InputContract records tensor/layout/dtype/shape/sequence/stride/chunk/causal/extent/grid/alignment/separation checks; IO.logicalInputs uses the prepared contiguous views. No source result or output-equality premise is added.

Exact FP program pairs and numerical dependencies are recorded in the [bmm_chunk_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
