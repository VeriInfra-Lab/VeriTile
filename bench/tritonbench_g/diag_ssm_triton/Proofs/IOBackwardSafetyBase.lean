import bench.tritonbench_g.diag_ssm_triton.Proofs.IOSafetyBase

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem option_true {α : Type} (o : Option α) :
    (match o with | some _ => True | none => True) ↔ True := by
  cases o <;> rfl

theorem previous_safe (bounds : RegionBounds) (s y : RegionName) (batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafe bounds (BackwardProjection.previousLoad s y batch dim B) u ↔
      Stmt.TraceSafeR .triv bounds (BackwardProjection.previousLoadReference s y batch dim B) u := by
  have h := BackwardProjection.previous_offset batch dim B
  simp only [BackwardProjection.previousOffset] at h
  simp only [BackwardProjection.previousLoad,BackwardProjection.previousLoadReference,
    Stmt.TraceSafe,Stmt.TraceSafeR,Stmt.TraceSafeList,Stmt.TraceSafeListR,
    Op.SafeAt.eq_def,Op.SafeAtR.eq_def,MemAccess.SafeAt,MemAccess.SafeAtR,
    MaskOpt.SafeAt,MaskOpt.SafeAtR,MemAccess.ActiveAddressSafe,MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafe,memAccessActiveAddressSafeR,MaskOpt.Active,MaskOpt.ActiveR,
    stepStmtR_triv,evalOpR_triv,true_and,and_true,BackwardProjection.previousOffset,h]
  have he := BackwardProjection.load_congr (d := .real) y
    (BackwardProjection.previousOffset batch dim B)
    (.sub .nat .scalarR (.ref .nat [B] "offsets") (.constNat (batch*dim)))
    (.maskOther (.ref .bool [B] "mask") ((Op.const 0).broadcast [B])) u
    (BackwardProjection.previous_offset batch dim B u)
  simp only [BackwardProjection.previousOffset] at he
  simp only [stepStmt,he]
  rfl

theorem backward_body (bounds : RegionBounds) (s y gx gy : RegionName)
    (length batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (BackwardProjection.loopBody s y gx gy length batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmBackwardLoopBody s y gx gy length batch dim B) u := by
  have eq := ForwardProjection.offset_step batch dim B
  simp only [ForwardProjection.offset] at eq
  have prev := BackwardProjection.previous_load_step s y batch dim B
  simp only [BackwardProjection.previousLoadReference] at prev
  have ps := previous_safe bounds s y batch dim B
  simp only [BackwardProjection.previousLoadReference] at ps
  simp only [BackwardProjection.loopBody,RealReference.diagSsmBackwardLoopBody,List.take,List.drop,
    List.cons_append,List.nil_append,Stmt.TraceSafeList,Stmt.TraceSafeListR,
    ps,stepStmtR_triv,ForwardProjection.offset_step,prev]
  -- The branch is now shared; only straight-line statements remain different.
  simp only [Stmt.TraceSafe,Stmt.TraceSafeR,
    Op.SafeAt.eq_def,Op.SafeAtR.eq_def,MemAccess.SafeAt,MemAccess.SafeAtR,
    MaskOpt.SafeAt,MaskOpt.SafeAtR,MemAccess.ActiveAddressSafe,MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafe,memAccessActiveAddressSafeR,MaskOpt.Active,MaskOpt.ActiveR,
    evalOpR_triv,true_and,and_true]
  simp only [ForwardProjection.offset,Stmt.TraceSafe,Op.SafeAt.eq_def,true_and,and_true,eq]
  rfl

theorem backward_prefix (bounds : RegionBounds) (l : RegionName) (batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (BackwardProjection.prelude l batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmBackwardPreLoop l batch dim B) u := by
  have eq := ForwardProjection.mask_step batch dim B
  simp only [ForwardProjection.mask] at eq
  simp only [BackwardProjection.prelude,RealReference.diagSsmBackwardPreLoop,List.take,List.drop,
    List.cons_append,List.nil_append,Stmt.TraceSafeList,Stmt.TraceSafeListR,Stmt.TraceSafe,Stmt.TraceSafeR,
    Op.SafeAt.eq_def,Op.SafeAtR.eq_def,MemAccess.SafeAt,MemAccess.SafeAtR,
    MaskOpt.SafeAt,MaskOpt.SafeAtR,MemAccess.ActiveAddressSafe,MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafe,memAccessActiveAddressSafeR,MaskOpt.Active,MaskOpt.ActiveR,
    stepStmtR_triv,evalOpR_triv,true_and,and_true,ForwardProjection.mask_step]
  simp only [ForwardProjection.mask,Stmt.TraceSafe,Op.SafeAt.eq_def,true_and,and_true,eq]
  rfl

theorem backward_post (bounds : RegionBounds) (gs gl : RegionName) (B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (RealReference.diagSsmBackwardPostLoop gs gl B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmBackwardPostLoop gs gl B) u := by
  simp only [RealReference.diagSsmBackwardPostLoop,Stmt.TraceSafeList,Stmt.TraceSafeListR,
    Stmt.TraceSafe,Stmt.TraceSafeR,Op.SafeAt.eq_def,Op.SafeAtR.eq_def,
    MemAccess.SafeAt,MemAccess.SafeAtR,MaskOpt.SafeAt,MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe,MemAccess.ActiveAddressSafeR,memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR,MaskOpt.Active,MaskOpt.ActiveR,stepStmtR_triv,evalOpR_triv,
    true_and,and_true]
  rfl


end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
