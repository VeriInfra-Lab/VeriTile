import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Kernels
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.LegacySurface
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourcePreservation
/-- Complete interface and AST preservation for every constexpr mask branch. -/
theorem original (p : Source.Parameters) :
    Source.kernel p=LegacySource.kernel
      ⟨p.A,p.B,p.O,p.M,p.N,p.K,p.TM,p.TN,p.TK,p.GROUP,p.divM,p.divN,p.divK⟩ := by
  rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> rfl
#axiomsClean original
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourcePreservation
