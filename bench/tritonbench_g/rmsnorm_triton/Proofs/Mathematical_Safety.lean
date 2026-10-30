import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_Candidate

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton
open VeriTile.Triton
open Full.ScratchRms Full.Prefix Full.PostStep Full.Wb
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

namespace Safety

theorem append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (before ++ after) s ↔
      Stmt.TraceSafeListR .triv bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeListR .triv bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeListR, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeListR, stepStmtR_triv, stepStmts]
    cases he : stepStmt head s with
    | none => simp
    | some t => simp only [Option.bind_some, ih, and_assoc]

theorem context_congr (bounds : RegionBounds) (before left right after : List Stmt)
    (hs : ∀ s, Stmt.TraceSafeListR .triv bounds left s ↔ Stmt.TraceSafeListR .triv bounds right s)
    (he : ∀ s, stepStmts left s = stepStmts right s) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (before ++ left ++ after) s ↔
      Stmt.TraceSafeListR .triv bounds (before ++ right ++ after) s := by
  simp only [List.append_assoc, append_iff, hs, he]

theorem range_congr (bounds : RegionBounds) (idx : RegName) (stop stride : Nat)
    (left right : List Stmt)
    (hs : ∀ s, Stmt.TraceSafeListR .triv bounds left s ↔ Stmt.TraceSafeListR .triv bounds right s)
    (he : ∀ s, stepStmts left s = stepStmts right s) (cur : Nat) (s : BlockState) :
    Stmt.forRangeTraceSafeR .triv bounds idx cur stop stride left s ↔
      Stmt.forRangeTraceSafeR .triv bounds idx cur stop stride right s := by
  rw [Stmt.forRangeTraceSafeR, Stmt.forRangeTraceSafeR]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  simp only [hz, ↓reduceIte]
  by_cases hc : cur < stop
  · simp only [hc, ↓reduceIte, hs, stepStmtsR_triv, he]
    cases stepStmts right (s.setReg idx .nat [] (Tile.scalar cur)) with
    | none => rfl
    | some t => exact and_congr_right (fun _ => range_congr bounds idx stop stride left right hs he (cur + stride) t)
  · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega


def varianceAssignment (B : Nat) (native : Bool) : Stmt :=
  .assign .real [B] "var" (.add .real (.consSame .nil) (.ref .real [B] "var")
    (if native then .libdevicePowi .scalarR (.ref .real [B] "x") (.constInt 2)
     else .mul .real (.consSame .nil)
       (.castFloat .real .real (.ref .real [B] "x"))
       (.castFloat .real .real (.ref .real [B] "x"))))

theorem variance_assignment_exec (B : Nat) (s : BlockState) :
    stepStmts [varianceAssignment B Bool.true] s =
      stepStmts [varianceAssignment B Bool.false] s := by
  simp only [varianceAssignment, Bool.false_eq_true, ↓reduceIte, stepStmts, stepStmt,
    evalOp_add, powi_square_eval, evalOp_mul, real_cast_eval]

theorem variance_assignment_safe (bounds : RegionBounds) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds [varianceAssignment B Bool.true] s ↔
      Stmt.TraceSafeListR .triv bounds [varianceAssignment B Bool.false] s := by
  simp [varianceAssignment, Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  split <;> split <;> trivial

theorem variance_exec (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    stepStmts (powiVarBody X sxk N B) s = stepStmts (varBody X sxk N B) s := by
  have hb : varBody X sxk N B = (varBody X sxk N B).take 3 ++
      [varianceAssignment B Bool.false] := rfl
  have hp : powiVarBody X sxk N B = (varBody X sxk N B).take 3 ++
      [varianceAssignment B Bool.true] := rfl
  rw [hb, hp]
  simp only [ProgramRewrite.append, variance_assignment_exec]

theorem variance_safe (bounds : RegionBounds) (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (powiVarBody X sxk N B) s ↔
      Stmt.TraceSafeListR .triv bounds (varBody X sxk N B) s := by
  have hb : varBody X sxk N B = (varBody X sxk N B).take 3 ++
      [varianceAssignment B Bool.false] := rfl
  have hp : powiVarBody X sxk N B = (varBody X sxk N B).take 3 ++
      [varianceAssignment B Bool.true] := rfl
  conv_rhs => rw [hb]
  rw [hp]
  simp only [append_iff, variance_assignment_safe, variance_assignment_exec]

theorem prefix_exec (v : Bool) (sxb sxm : Nat) (s : BlockState) :
    stepStmts (indexPrefix v sxb sxm) s = stepStmts (indexPrefix Bool.false sxb sxm) s := by
  cases v
  · rfl
  · simp [indexPrefix, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem typed_safe_iff (bounds : RegionBounds) (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s ↔
      (rmsnorm_triton X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.TraceSafeR .triv bounds s := by
  rw [typed_projection]
  have hs : ∀ t, Stmt.TraceSafeListR .triv bounds
      [.forRange "block_n_start_idx" 0 N B (powiVarBody X sxk N B)] t ↔
      Stmt.TraceSafeListR .triv bounds
      [.forRange "block_n_start_idx" 0 N B (varBody X sxk N B)] t := by
    intro t
    simp only [Stmt.TraceSafeListR, Stmt.TraceSafeR]
    simp only [range_congr bounds "block_n_start_idx" N B _ _
      (variance_safe bounds X sxk N B) (variance_exec X sxk N B)]
    split <;> split <;> simp
  have he : ∀ t, stepStmts [.forRange "block_n_start_idx" 0 N B (powiVarBody X sxk N B)] t =
      stepStmts [.forRange "block_n_start_idx" 0 N B (varBody X sxk N B)] t := by
    intro t
    simp only [stepStmts, stepStmt]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (variance_exec X sxk N B)]
  have hprefix (variant : Bool) (t : BlockState) :
      Stmt.TraceSafeListR .triv bounds (indexPrefix variant sxb sxm) t := by
    cases variant <;> simp [indexPrefix, Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
      stepStmtR_triv, stepStmt, evalOp.eq_def]
  have hp : prefixStmts X sxb sxm B =
      indexPrefix Bool.false sxb sxm ++ (prefixStmts X sxb sxm B).drop 3 := rfl
  unfold Kernel.TraceSafeR
  rw [math_body, Full.Final.body_decomp]
  conv_rhs => rw [hp]
  simp only [List.append_assoc, append_iff, hprefix, prefix_exec, hs, he, true_and]

end Safety

/-- Actual access safety for both complete erased source programs. The only
conditions concern valid launch geometry, separation and active access bounds. -/
theorem typed_trace_safe (input weight output : Storage) (v : Bool) (bounds : RegionBounds)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hOX : O ≠ X) (hOW : O ≠ W) (hsok : 0 < sok) (s : BlockState)
    (hX : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sxb + s.pids 1 * sxm + (t.val * B + j.val) * sxk < bounds X)
    (hW : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      (t.val * B + j.val) * sw < bounds W)
    (hO : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sob + s.pids 1 * som + (t.val * B + j.val) * sok < bounds O) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s :=
  (Safety.typed_safe_iff bounds input weight output v X W O sxb sxm sxk sw sob som sok N B eps s).mpr
    (legacy_trace_safe .triv bounds X W O sxb sxm sxk sw sob som sok N B eps
      hB hOX hOW hsok s hX hW hO)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormTriton
