import bench.repairs.tritonbench_g.triton_mul2.Kernels
import bench.repairs.tritonbench_g.triton_mul2.Proofs.LegacyBody
import VeriTile.Meta.StatementAudit

/-! Full typed AST equality for all four actual native bodies. -/
namespace VeriTile.Bench.Repairs.TritonMul2.SourcePreservation
open VeriTile.Triton

theorem sourceKernel (X O : RegionName) (n B : Nat) :
    TritonMul2.sourceKernel X O n B = LegacyBody.sourceKernel X O n B := by rfl
#axiomsClean sourceKernel

theorem inplaceSourceKernel (X : RegionName) (n B : Nat) :
    TritonMul2.inplaceSourceKernel X n B = LegacyBody.inplaceSourceKernel X n B := by rfl
#axiomsClean inplaceSourceKernel

theorem commutedSourceKernel (X O : RegionName) (n B : Nat) :
    TritonMul2.commutedSourceKernel X O n B = LegacyBody.commutedSourceKernel X O n B := by rfl
#axiomsClean commutedSourceKernel

theorem inplaceCommutedSourceKernel (X : RegionName) (n B : Nat) :
    TritonMul2.inplaceCommutedSourceKernel X n B = LegacyBody.inplaceCommutedSourceKernel X n B := by rfl
#axiomsClean inplaceCommutedSourceKernel

end VeriTile.Bench.Repairs.TritonMul2.SourcePreservation
