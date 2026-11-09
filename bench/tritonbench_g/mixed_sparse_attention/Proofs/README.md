# mixed_sparse_attention

Current repaired Python implementation: [mixed_sparse_attention.py](../mixed_sparse_attention.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

independent occurrence-list weighted quotient with base-two exponent and exact coefficient 1.44269504; duplicate columns/blocks retained

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Both original Python functions are pasted in `Kernels.lean`, at the retained FP16 and BF16 dtype specializations. The candidate fuses only the sparse-column zero/select pair. Caller metadata supplies FP32 tensors, signed32 metadata and runtime scalar arguments.

The public mathematical and structural theorems retain their original natural metadata domain. The checked native boundary uses the existing nonnegative signed32 guards; `Proofs/MetadataReadViews.lean` proves exact word decoding and natural read agreement for every selected metadata role. No native signed32 representation is claimed for arbitrary unbounded natural inputs.
