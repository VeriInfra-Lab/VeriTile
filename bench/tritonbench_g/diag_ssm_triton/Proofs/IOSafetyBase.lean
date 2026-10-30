import bench.tritonbench_g.diag_ssm_triton.Proofs.RegionCorrectness
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_BackwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexForwardSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexBackwardSafety
import VeriTile.Triton.Memory.RegionalKernelIO

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem loop_iff (bounds : RegionBounds) (idx : RegName) (cur stop : Nat)
    (left right : List Stmt)
    (he : ∀ s, stepStmts left s = stepStmts right s)
    (hs : ∀ s, Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeListR .triv bounds right s)
    (s : BlockState) :
    Stmt.forLoopTraceSafe bounds idx cur stop left s ↔
      Stmt.forLoopTraceSafeR .triv bounds idx cur stop right s := by
  rw [Stmt.forLoopTraceSafe, Stmt.forLoopTraceSafeR]
  by_cases hc : cur < stop
  · simp only [hc,↓reduceIte,hs,stepStmtsR_triv,he]
    cases stepStmts right (s.setReg idx .nat [] (Tile.scalar cur)) with
    | none => rfl
    | some t => exact and_congr_right (fun _ => loop_iff bounds idx (cur+1) stop left right he hs t)
  · simp only [hc,↓reduceIte]
termination_by stop-cur
 decreasing_by omega

theorem loop_range_iff (bounds : RegionBounds) (idx : RegName) (cur stop : Nat)
    (left right : List Stmt)
    (he : ∀ s, stepStmts left s = stepStmts right s)
    (hs : ∀ s, Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeListR .triv bounds right s)
    (s : BlockState) :
    Stmt.forLoopTraceSafe bounds idx cur stop left s ↔
      Stmt.forRangeTraceSafeR .triv bounds idx cur stop 1 right s := by
  rw [Stmt.forLoopTraceSafe, Stmt.forRangeTraceSafeR]
  by_cases hc : cur < stop
  · simp only [hc,↓reduceIte,Nat.one_ne_zero,hs,stepStmtsR_triv,he]
    cases stepStmts right (s.setReg idx .nat [] (Tile.scalar cur)) with
    | none => rfl
    | some t => exact and_congr_right (fun _ => loop_range_iff bounds idx (cur+1) stop left right he hs t)
  · simp only [hc,↓reduceIte,Nat.one_ne_zero]
termination_by stop-cur
 decreasing_by omega

theorem forward_body (bounds : RegionBounds) (x y : RegionName) (batch dim B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (ForwardProjection.loopBody x y batch dim B) s ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmForwardLoopBody x y batch dim B) s := by
  have eq := ForwardProjection.offset_step batch dim B
  simp only [ForwardProjection.offset] at eq
  simp only [ForwardProjection.loopBody,RealReference.diagSsmForwardLoopBody,List.drop,
    Stmt.TraceSafeList,Stmt.TraceSafeListR,Stmt.TraceSafe,Stmt.TraceSafeR,
    Op.SafeAt.eq_def,Op.SafeAtR.eq_def,
    MemAccess.SafeAt,MemAccess.SafeAtR,MaskOpt.SafeAt,MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe,MemAccess.ActiveAddressSafeR,memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR,MaskOpt.Active,MaskOpt.ActiveR,stepStmtR_triv,evalOpR_triv,
    true_and,and_true,ForwardProjection.offset_step]
  simp only [ForwardProjection.offset,Stmt.TraceSafe,Op.SafeAt.eq_def,true_and,and_true,eq]
  rfl

theorem forward_prefix (bounds : RegionBounds) (s l : RegionName) (batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (ForwardProjection.prelude s l batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmForwardPreLoop s l batch dim B) u := by
  have eq := ForwardProjection.mask_step batch dim B
  simp only [ForwardProjection.mask] at eq
  simp only [ForwardProjection.prelude,RealReference.diagSsmForwardPreLoop,List.take,List.drop,
    List.cons_append,List.nil_append,Stmt.TraceSafeList,Stmt.TraceSafeListR,Stmt.TraceSafe,Stmt.TraceSafeR,
    Op.SafeAt.eq_def,Op.SafeAtR.eq_def,
    MemAccess.SafeAt,MemAccess.SafeAtR,MaskOpt.SafeAt,MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe,MemAccess.ActiveAddressSafeR,memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR,MaskOpt.Active,MaskOpt.ActiveR,stepStmtR_triv,evalOpR_triv,
    true_and,and_true,ForwardProjection.mask_step]
  simp only [ForwardProjection.mask,Stmt.TraceSafe,Op.SafeAt.eq_def,true_and,and_true,eq]
  rfl

theorem appendR_iff (bounds : RegionBounds) (left right : List Stmt) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (left++right) s ↔
      Stmt.TraceSafeListR .triv bounds left s ∧
        ∀ t, stepStmts left s = some t → Stmt.TraceSafeListR .triv bounds right t := by
  induction left generalizing s with
  | nil => simp [Stmt.TraceSafeListR,stepStmts]
  | cons st rest ih =>
    cases he : stepStmt st s <;>
      simp [Stmt.TraceSafeListR,stepStmts,stepStmtR_triv,he,ih,and_assoc]

end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
