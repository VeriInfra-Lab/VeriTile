import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardProjection
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton

theorem natural_range_safe (bounds : RegionBounds) (idx : RegName)
    (start stop : Nat) (body : List Stmt) (u : BlockState) :
    Stmt.forLoopTraceSafe bounds idx start stop body u ↔
      Stmt.forRangeTraceSafe bounds idx start stop 1 body u := by
  rw [Stmt.forLoopTraceSafe, Stmt.forRangeTraceSafe]
  by_cases h : start < stop
  · simp only [Nat.one_ne_zero, h, ↓reduceIte]
    apply and_congr_right
    intro _
    cases stepStmts body (u.setReg idx .nat [] (Tile.scalar start)) with
    | none => rfl
    | some v => exact natural_range_safe bounds idx (start+1) stop body v
  · simp only [Nat.one_ne_zero, h, ↓reduceIte]
termination_by stop-start

#axiomsClean natural_range_safe
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
