import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Kernels
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.LegacySurface
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourcePreservation
open VeriTile.Triton
/-- Proof-only view of the original tuple assignment. Its two RHS registers
are distinct from both destinations, so the checked frontend emits the pair in
an always-taken block. This view is not the authoritative source kernel. -/
def tupleView : ComputeStmt → ComputeStmt
  | .ifThenElse condition yes no => .ifThenElse condition [.ifThen (.constBool Bool.true) yes] no
  | statement => statement

/-- The raw body uses the original unused Python iterator `_`. The archived
manual quotation named it `_i`. All memory ports and all instructions agree after restoring that
source iterator and the explicit tuple block. Current execution and safety proofs operate directly
on the raw kernel and its `_` loop; no whole-state equality to `_i` is asserted. -/
theorem original (p : Source.Parameters) :
    let old : LegacySource.Parameters :=
      ⟨p.A,p.B,p.O,p.M,p.N,p.K,p.TM,p.TN,p.TK,p.GROUP,p.divM,p.divN,p.divK⟩
    (Source.kernel p).inputs = (LegacySource.kernel old).inputs ∧
    (Source.kernel p).outputs = (LegacySource.kernel old).outputs ∧
    (Source.kernel p).surfaceBody = (LegacySource.prelude old).map tupleView ++
      [.forRangeDyn "_" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1)
        (LegacySource.body old)] ++ LegacySource.epilogue old := by
  rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> exact ⟨rfl,rfl,rfl⟩
#axiomsClean original
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourcePreservation
