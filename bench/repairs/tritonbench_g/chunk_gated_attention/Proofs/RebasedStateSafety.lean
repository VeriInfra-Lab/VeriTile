import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RebasedAccessSafety

/-! Actual flattened initialization and finalization access safety, with the
original generic checked-access lemmas. All unspecified values are irrelevant. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedStateSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
open RealHAccessSafety (load2_safe store2_safe)
open RebasedAccess RebasedAccessSafety
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

open RebasedInit (decoded registers)
theorem registers_executes (p : Layout) (r : Regions) (base : Nat) (u : BlockState) :
    stepStmts ((RebasedProgram.initialization p r base Bool.false).take 5) u =
      some (registers p base u) := by
  simp only [RebasedProgram.initialization,List.take,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (RebasedProgram.plus (.programId 0) (.constNat base)) u =
      some (Tile.scalar (u.pids 0+base)) from by
        simp only [RebasedProgram.plus,evalOp_add,evalOp_programId,evalOp_constNat]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.mod .nat .nil (.ref .nat [] "linear") (.constNat (p.V/p.BV))) _ =
      some (Tile.scalar (decoded p base u).value) from by
        simp only [evalOp,BlockState.setReg_same,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.mod .nat .nil (.div .nat .nil (.ref .nat [] "linear") (.constNat (p.V/p.BV)))
      (.constNat (p.K/p.BK))) _ = some (Tile.scalar (decoded p base u).key) from by
        simp only [evalOp,BlockState.setReg_same,BlockState.setReg_ne_name,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show
    evalOp (.div .nat .nil (.ref .nat [] "linear") (.constNat ((p.V/p.BV)*(p.K/p.BK)))) _ =
      some (Tile.scalar (decoded p base u).head) from by
        simp only [evalOp,BlockState.setReg_same,BlockState.setReg_ne_name,Option.bind_eq_bind,Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  rw [stepStmts.nil]
  rfl

theorem init_safe (p : Layout) (r : Regions) (base : Nat) (hasSeed : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : hasSeed = Bool.true → StateBounds p (decoded p base u) r.seed bounds) :
    Stmt.TraceSafeList bounds (RebasedProgram.initialization p r base hasSeed) u := by
  have he := registers_executes p r base u
  change Stmt.TraceSafeList bounds
    (((RebasedProgram.initialization p r base Bool.false).take 5) ++
      [.ifThen (.constBool hasSeed)
        [.assign .blockPtr [p.BK,p.BV] "p_h" (RebasedProgram.statePtr p r.seed),
         .assign .real [p.BK,p.BV] "b_h"
           (.add .real (.consSame (.consSame .nil)) (carry p) (load2 p.BK p.BV "p_h"))]]) u
  apply TraceProofs.append_safe _ u (by
    apply TraceProofs.all_safe
    simp [RebasedProgram.initialization,Stmt.TraceSafe,RebasedProgram.plus,Op.SafeAt])
  intro v hv
  rw [he] at hv
  cases hv
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases hasSeed
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_eval p (decoded p base u) r.seed _
      (by simp [registers]) (by simp [registers]) (by simp [registers])))]
    refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.statePtr,RebasedProgram.local2,stateBase,times,Op.SafeAt],?_⟩
    apply TraceProofs.assign_cons
    · rw [Op.SafeAt.eq_def]
      refine ⟨by simp [carry,Op.SafeAt],?_⟩
      exact load2_safe r.seed ((decoded p base u).head*p.K*p.V+(decoded p base u).key*p.BK*p.V+(decoded p base u).value*p.BV)
        p.BK p.BV p.BK p.BV p.V 1 0 0 "p_h" bounds _ (hb rfl) (by simp [statePointer,matrix])
    · intro _
      exact Stmt.TraceSafeList.nil_intro
theorem final_safe (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : storeFinal = Bool.true → StateBounds p l r.final bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    Stmt.TraceSafeList bounds (RebasedProgram.finalization p r storeFinal) u := by
  unfold RebasedProgram.finalization
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases storeFinal
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_eval p l r.final u hih hik hiv))]
    refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.statePtr,RebasedProgram.local2,stateBase,times,Op.SafeAt],?_⟩
    apply Stmt.TraceSafeList.cons_intro
      (store2_safe r.final (l.head*p.K*p.V+l.key*p.BK*p.V+l.value*p.BV) p.BK p.BV p.BK p.BV p.V 1 0 0
        "p_h" "b_h" bounds _ (hb rfl) (by simp [statePointer,matrix]))
    intro _ _
    exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedStateSafety
