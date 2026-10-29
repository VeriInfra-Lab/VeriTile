# chunk_cumsum_vector

For the same feature column, inclusive sum over logical time k≤t and k<T of the initial input, with time/feature coordinates made explicit.

- [chunk_cumsum_vector_repaired.py](chunk_cumsum_vector_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One actual feature-tile invocation with its row/feature Program coordinates, running the complete time loop; all nine source configurations and tail feature/time masks. It is not relabeled as a simultaneous whole-feature-grid result.

## Conditions

Public conditions:

- Geometry.OutputLayout h
- c∈configurations
- p : Geometry.Program h
- The singleton prefixLaunch explicitly binds the actual source program IDs
- RegionalScheduledImplements preserves actual source safety/success, all active logical outputs, bounded regional accesses and exact exterior raw-cell frame

Contract domain source:

Contract.mathematical is the independent inclusive prefix. Contract retains FP32 rank-4 contiguous CUDA input/output choices, row/feature-grid limits and alignment. Geometry/output layout and source strides remain visible rather than replaced by a generic output-existence premise.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_cumsum_vector contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
