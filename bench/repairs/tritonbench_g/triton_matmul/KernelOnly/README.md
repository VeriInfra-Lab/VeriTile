# triton_matmul

For each logical output (i,j), sum over every k of A i k * B k j.

- [triton_matmul_repaired.py](../triton_matmul_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete actual grid in every permutation of range(programs g.request), grouped tiles, tails and all K iterations; FP16/E4M3 storage and FP32 accumulation/source final storage cast retained.

## Conditions

- ids.Perm (List.range (programs g.request)); g.C≠g.A and g.C≠g.B
- Geometry supplies logical rows/columns/inner, storage and actual strides; ProgramIO public input/output windows provide bounds/bindings
- Contract.InputContract and direct existing-output OutputABI have physical window/separation binding; no allocator, numerical output or run premise

Exact FP program pairs and numerical dependencies are recorded in the [triton_matmul contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
