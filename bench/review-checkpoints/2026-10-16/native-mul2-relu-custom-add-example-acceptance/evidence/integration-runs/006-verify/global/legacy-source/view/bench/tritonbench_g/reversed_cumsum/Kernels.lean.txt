import bench.tritonbench_g.reversed_cumsum.Proofs.Source

/-!
Source-audit target: chunk_global_reversed_cumsum_vector_kernel
Source-audit Lean: reversedVectorSource%
Source-audit args: version=original

The imported macro contains the entire original typed JIT body and its
comparison-reversed candidate. Both retain the signed reverse loop,
unspecified padding, independent storage casts, dot and every carry update.
Historical prepared-input slices are isolated in Proofs/LegacyReal.lean.
-/
