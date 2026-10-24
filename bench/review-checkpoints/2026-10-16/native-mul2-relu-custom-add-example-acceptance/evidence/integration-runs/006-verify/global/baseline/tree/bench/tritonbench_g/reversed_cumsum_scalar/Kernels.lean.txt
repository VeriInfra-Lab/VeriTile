/-
Source-audit target: chunk_global_reversed_cumsum_scalar_kernel
Source-audit Lean: reversedScalarSource%
Source-audit args: version=original
-/
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Source

/-! Public complete typed kernels. The original retains every source
statement; the candidate only commutes the input pointer's row-times-T
natural multiplication. The source audit checks the macro in Proofs/Source. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Kernels

abbrev original := TypedSource.original
abbrev indexCommuted := TypedSource.indexCommuted

end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Kernels
