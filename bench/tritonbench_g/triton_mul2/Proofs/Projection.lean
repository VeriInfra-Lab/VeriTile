import bench.tritonbench_g.triton_mul2.Kernels
import bench.tritonbench_g.triton_mul2.Proofs.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TritonMul2.Projection
open VeriTile.Triton

/-- Precision erasure is used only for the mathematical interpretation.
Floating-point equivalence executes the typed source directly. -/
theorem original_projection (X O : RegionName) (n B : Nat) :
    (sourceKernel X O n B).toAlgorithm? =
      .ok (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_kernel X O n B).toAlgKernel := rfl

theorem commuted_projection (X O : RegionName) (n B : Nat) :
    (commutedSourceKernel X O n B).toAlgorithm? =
      .ok (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_commuted_kernel X O n B).toAlgKernel := rfl

theorem inplace_projection (X : RegionName) (n B : Nat) :
    (inplaceSourceKernel X n B).toAlgorithm? =
      .ok (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_inplace_kernel X n B).toAlgKernel := rfl

theorem inplace_commuted_projection (X : RegionName) (n B : Nat) :
    (inplaceCommutedSourceKernel X n B).toAlgorithm? =
      .ok (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_inplace_commuted_kernel X n B).toAlgKernel := rfl

/-- Compatibility with the retained finite-index source lowering. This is
the algorithm body projection, not the floating-point rewrite source. -/
theorem original_body (X O : RegionName) (n B : Nat) :
    (sourceKernel X O n B).body =
      (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_kernel X O n B).body := rfl

theorem inplace_body (X : RegionName) (n B : Nat) :
    (inplaceSourceKernel X n B).body =
      (VeriTile.Bench.TritonBenchG.TritonMul2.mul2_inplace_kernel X n B).body := rfl

#axiomsClean original_projection
#axiomsClean commuted_projection
#axiomsClean inplace_projection
#axiomsClean inplace_commuted_projection
#axiomsClean original_body
#axiomsClean inplace_body
end VeriTile.Bench.Repairs.TritonMul2.Projection
