# matmul_triton1

Current repaired Python implementation: [matmul_triton1.py](matmul_triton1.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun (a,b) row col => ∑ k : Fin innerSize, a row k * b k col

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.

`Kernels.lean` contains the repaired Python body and its mask-caching candidate,
including all runtime `uint64` conversions, original names and statement order,
and `z += tl.dot(..., allow_tf32=false)`. Argument bindings and pointer types
are outside the pasted body. There is no separate complete Lean implementation.

`Proofs/Source*` and `Proofs/Mask*` execute these two bodies directly. The public
Canonical/Ready/StorageBounds/geometry/separation conditions are retained, with
full-grid success, access safety, output values and exact raw memory frames.
`SourceFiniteOperands` also connects actual loaded tiles to the modular uint64
operand/address model; it supplies no whole-kernel executor.

`Sources/DotProfile.lean` names the selected compiler execution profile for the
accumulator-fed local dot. It does not assert an IEEE identity between a
separately rounded dot plus add and a fused dot. The opaque primitive contains
only a vector dot and its carry, with the original operands and options kept in
the source execution. Pure mathematical sum, geometry and scatter definitions
remain proof helpers.

The original kernel body is pasted with its Python indentation, colons and
comments. `jit-correspondence.json` checks its complete Python AST and the
external call signature; no casts or runtime statements are normalized away.
