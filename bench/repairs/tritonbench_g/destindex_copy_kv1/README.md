# destindex_copy_kv1

destindex_copy_kv1_kernel_real: complete original scatter schedule implements the logical payload copy.

- [destindex_copy_kv1_repaired.py](destindex_copy_kv1_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete arbitrary-order row grid for the selected shared kv1 source; destindex_copy applies this common source to either repaired plane independently. Every head and dimension is preserved; metadata remains a distinct integer channel.

## Conditions

Conditions:

- The schedule membership is exactly p<count (arbitrary complete order, retaining duplicate invocations allowed by that premise)
- Admissible explicitly requires nonnegative injective signed destination metadata
- Canonical ProgramIO bounds/readback and write map provide storage capacity, access safety and the complete exterior frame

Domain source:

Shared destindex_copy_kv1/IO.lean Admissible/inputChannels/outputChannels/memory; each selected Contract retains signed-int32 bounds/uniqueness, shape, storage separation and binding. destindex_copy Contract has independently configured nope/rope planes.

Exact FP program pairs and numerical dependencies are recorded in the [destindex_copy_kv1 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
