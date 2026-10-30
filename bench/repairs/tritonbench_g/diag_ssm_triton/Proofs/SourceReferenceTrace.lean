import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardSafety
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceReferenceTrace
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-! These equivalences concern only the five fixed mathematical fragments
below. TraceSafe and TraceSafeR are not interchangeable for arbitrary kernels. -/

theorem forward_prefix (bounds : RegionBounds) (s l : RegionName) (batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (RealReference.diagSsmForwardPreLoop s l batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmForwardPreLoop s l batch dim B) u := by
  simp only [RealReference.diagSsmForwardPreLoop,
    Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR,
    Op.SafeAt.eq_def, Op.SafeAtR.eq_def,
    MemAccess.SafeAt, MemAccess.SafeAtR, MaskOpt.SafeAt, MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR, MaskOpt.Active, MaskOpt.ActiveR, stepStmtR_triv, evalOpR_triv,
    true_and, and_true]
  all_goals rfl

#axiomsClean forward_prefix

theorem forward_body (bounds : RegionBounds) (x y : RegionName) (batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (RealReference.diagSsmForwardLoopBody x y batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmForwardLoopBody x y batch dim B) u := by
  simp only [RealReference.diagSsmForwardLoopBody,
    Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR,
    Op.SafeAt.eq_def, Op.SafeAtR.eq_def,
    MemAccess.SafeAt, MemAccess.SafeAtR, MaskOpt.SafeAt, MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR, MaskOpt.Active, MaskOpt.ActiveR, stepStmtR_triv, evalOpR_triv,
    true_and, and_true]
  all_goals rfl

#axiomsClean forward_body

theorem backward_prefix (bounds : RegionBounds) (l : RegionName) (batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (RealReference.diagSsmBackwardPreLoop l batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmBackwardPreLoop l batch dim B) u := by
  simp only [RealReference.diagSsmBackwardPreLoop,
    Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR,
    Op.SafeAt.eq_def, Op.SafeAtR.eq_def,
    MemAccess.SafeAt, MemAccess.SafeAtR, MaskOpt.SafeAt, MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR, MaskOpt.Active, MaskOpt.ActiveR, stepStmtR_triv, evalOpR_triv,
    true_and, and_true]
  all_goals rfl

#axiomsClean backward_prefix

theorem backward_body (bounds : RegionBounds) (s y gx gy : RegionName) (length batch dim B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (RealReference.diagSsmBackwardLoopBody s y gx gy length batch dim B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmBackwardLoopBody s y gx gy length batch dim B) u := by
  simp only [RealReference.diagSsmBackwardLoopBody,
    Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR,
    Op.SafeAt.eq_def, Op.SafeAtR.eq_def,
    MemAccess.SafeAt, MemAccess.SafeAtR, MaskOpt.SafeAt, MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR, MaskOpt.Active, MaskOpt.ActiveR, stepStmtR_triv, evalOpR_triv,
    true_and, and_true]
  all_goals rfl

#axiomsClean backward_body

theorem backward_post (bounds : RegionBounds) (gs gl : RegionName) (B : Nat) (u : BlockState) :
    Stmt.TraceSafeList bounds (RealReference.diagSsmBackwardPostLoop gs gl B) u ↔
      Stmt.TraceSafeListR .triv bounds (RealReference.diagSsmBackwardPostLoop gs gl B) u := by
  simp only [RealReference.diagSsmBackwardPostLoop,
    Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR,
    Op.SafeAt.eq_def, Op.SafeAtR.eq_def,
    MemAccess.SafeAt, MemAccess.SafeAtR, MaskOpt.SafeAt, MaskOpt.SafeAtR,
    MemAccess.ActiveAddressSafe, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafe,
    memAccessActiveAddressSafeR, MaskOpt.Active, MaskOpt.ActiveR, stepStmtR_triv, evalOpR_triv,
    true_and, and_true]
  all_goals rfl

#axiomsClean backward_post

theorem loop (bounds : RegionBounds) (idx : RegName) (cur stop : Nat) (body : List Stmt)
    (hs : ∀ u, Stmt.TraceSafeList bounds body u ↔ Stmt.TraceSafeListR .triv bounds body u)
    (u : BlockState) :
    Stmt.forLoopTraceSafe bounds idx cur stop body u ↔
      Stmt.forLoopTraceSafeR .triv bounds idx cur stop body u := by
  rw [Stmt.forLoopTraceSafe, Stmt.forLoopTraceSafeR]
  by_cases h : cur < stop
  · simp only [h, ↓reduceIte, hs, stepStmtsR_triv]
    apply and_congr_right
    intro _
    cases stepStmts body (u.setReg idx .nat [] (Tile.scalar cur)) with
    | none => rfl
    | some v => exact loop bounds idx (cur+1) stop body hs v
  · simp only [h, ↓reduceIte]
termination_by stop-cur

theorem range (bounds : RegionBounds) (idx : RegName) (cur stop : Nat) (body : List Stmt)
    (hs : ∀ u, Stmt.TraceSafeList bounds body u ↔ Stmt.TraceSafeListR .triv bounds body u)
    (u : BlockState) :
    Stmt.forRangeTraceSafe bounds idx cur stop 1 body u ↔
      Stmt.forRangeTraceSafeR .triv bounds idx cur stop 1 body u := by
  rw [Stmt.forRangeTraceSafe, Stmt.forRangeTraceSafeR]
  by_cases h : cur < stop
  · simp only [h, Nat.one_ne_zero, ↓reduceIte, hs, stepStmtsR_triv]
    apply and_congr_right
    intro _
    cases stepStmts body (u.setReg idx .nat [] (Tile.scalar cur)) with
    | none => rfl
    | some v => exact range bounds idx (cur+1) stop body hs v
  · simp only [h, Nat.one_ne_zero, ↓reduceIte]
termination_by stop-cur

theorem appendR (bounds : RegionBounds) (left right : List Stmt) (u : BlockState) :
    Stmt.TraceSafeListR .triv bounds (left ++ right) u ↔
      Stmt.TraceSafeListR .triv bounds left u ∧
        ∀ v, stepStmts left u = some v → Stmt.TraceSafeListR .triv bounds right v := by
  induction left generalizing u with
  | nil => simp [Stmt.TraceSafeListR, stepStmts]
  | cons st rest ih =>
    cases he : stepStmt st u <;>
      simp [Stmt.TraceSafeListR, stepStmts, stepStmtR_triv, he, ih, and_assoc]

theorem forward (p : ForwardLayout) (length : Nat) (bounds : RegionBounds) (u : BlockState) :
    (RealReference.diag_ssm_forward_kernel p.s p.x p.l p.y length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u ↔
      Kernel.TraceSafeR .triv bounds (RealReference.diag_ssm_forward_kernel p.s p.x p.l p.y length p.batch p.dim p.B).toAlgKernel u := by
  simp only [Kernel.TraceSafe, Kernel.TraceSafeR, RealReference.diag_ssm_forward_kernel_toAlg_body,
    RealReference.diagSsmForwardProjectedBody, TraceRewrite.append_iff, appendR, forward_prefix]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro v
  apply imp_congr_right
  intro _
  simp only [Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR, stepStmtR_triv,
    loop bounds "t" 0 length _ (forward_body bounds p.x p.y p.batch p.dim p.B)]
  cases stepStmt (.forLoop "t" length (RealReference.diagSsmForwardLoopBody p.x p.y p.batch p.dim p.B)) v <;> rfl

theorem backward (p : BackwardLayout) (length : Nat) (bounds : RegionBounds) (u : BlockState) :
    (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u ↔
      Kernel.TraceSafeR .triv bounds (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel u := by
  simp only [Kernel.TraceSafe, Kernel.TraceSafeR, RealReference.diag_ssm_backward_kernel_toAlg_body,
    RealReference.diagSsmBackwardProjectedBody, List.append_assoc, TraceRewrite.append_iff, appendR, backward_prefix]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro v
  apply imp_congr_right
  intro _
  simp only [Stmt.TraceSafeList, Stmt.TraceSafeListR, Stmt.TraceSafe, Stmt.TraceSafeR, stepStmtR_triv,
    range bounds "i" 0 length _ (backward_body bounds p.s p.y p.gx p.gy length p.batch p.dim p.B),
    backward_post]
  cases stepStmt (.forRange "i" 0 length 1 (RealReference.diagSsmBackwardLoopBody p.s p.y p.gx p.gy length p.batch p.dim p.B)) v <;> rfl

#axiomsClean loop
#axiomsClean range
#axiomsClean appendR
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceReferenceTrace
