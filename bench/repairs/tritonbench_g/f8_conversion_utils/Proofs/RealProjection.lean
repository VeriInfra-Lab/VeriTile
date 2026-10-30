import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.KernelContract
import VeriTile.Triton.Memory.KernelSpec.Masked

namespace VeriTile.Bench.Repairs.F8ConversionUtils.PublicReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.F8ConversionUtils
open Checked
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

/-- Source erasure commutes with the existing whole-kernel proof, including
both decode stores. The original typed source is unchanged. -/
theorem source_body (p : Source.Profile) (Y X : RegionName) (N B : Nat) :
    (Source.kernel p Y X N B).eraseDType.toAlgKernel.body =
      (RealProof.original p.direction.doubleStore Y X N B).toAlgKernel.body := by
  rw [← real_source]
  exact RealSource.body_eq p.direction p.encoder p.format Y X N B

theorem source_projection (p : Source.Profile) (Y X : RegionName) (N B : Nat) :
    (Source.kernel p Y X N B).eraseDType.toAlgorithm? =
      .ok (Source.kernel p Y X N B).eraseDType.toAlgKernel := by
  cases p <;> simp [Source.kernel,Source.toHalf,Source.toFP8,TypedSource.toHalf,TypedSource.toFP8,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind]

end
end VeriTile.Bench.Repairs.F8ConversionUtils.PublicReal
