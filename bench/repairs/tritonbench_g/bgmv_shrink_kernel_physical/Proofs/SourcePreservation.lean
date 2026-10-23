import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.LegacySurface
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.SourcePreservation
/-- The complete original kernel, including interface and source AST, is
unchanged by exposing runtime scaling and replacing the previous generator. -/
theorem original (h : Call) : Kernels.source h=LegacySurface.source h := by
  cases e : h.storage <;> simp only [Kernels.source,LegacySurface.source,e]
  all_goals rfl
#axiomsClean original
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.SourcePreservation
