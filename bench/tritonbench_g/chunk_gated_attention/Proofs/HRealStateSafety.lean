import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealInit
import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealFinal
import bench.tritonbench_g.chunk_gated_attention.Proofs.HBounds
import VeriTile.Triton.Memory.Integer
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.StateSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RebasedAccess VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
open RealHAccessSafety (load2_safe store2_safe)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

theorem prelude_safe (p : Parameters) (r : RealHProgram.Regions) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (prelude p r) u := by
  rw [prelude_ast]
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt]

theorem seed_safe (p : Parameters) (l : Launch) (r : RealHProgram.Regions)
    (bounds : RegionBounds) (u : BlockState)
    (hb : p.hasSeed=Bool.true→StateBounds p.toLayout l r.seed bounds)
    (hik : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int))) :
    Stmt.TraceSafeList bounds (seedStage p r) u := by
  rw [seed_ast]
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro zero
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _;exact Stmt.TraceSafeList.nil_intro)
  by_cases h:p.hasSeed
  · simp only [h]
    rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_pointer_eval p l r.seed _
      (by simp [hih]) (by simp [hik]) (by simp [hiv])))]
    refine ⟨by simp [Stmt.TraceSafe,statePtr,Op.SafeAt],?_⟩
    apply TraceProofs.assign_cons
    · rw [Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt],?_⟩
      exact load2_safe r.seed (l.head*p.K*p.V+l.key*p.BK*p.V+l.value*p.BV)
        p.BK p.BV p.BK p.BV p.V 1 0 0 "p_h" bounds _ (hb h) (by simp [statePointer,matrix])
    · intro _;exact Stmt.TraceSafeList.nil_intro
  · simp only [h]
    rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩

theorem initialization_safe (p : Parameters) (r : RealHProgram.Regions)
    (bounds : RegionBounds) (u : BlockState)
    (hb : p.hasSeed=Bool.true→StateBounds p.toLayout (decoded p u) r.seed bounds) :
    Stmt.TraceSafeList bounds (prelude p r++seedStage p r) u := by
  apply TraceProofs.append_safe _ u (prelude_safe p r bounds u)
  intro v hv
  rw [(preludeExecution p r u).property] at hv
  cases hv
  exact seed_safe p (decoded p u) r bounds _ hb (prelude_key p r u) (prelude_value p r u) (prelude_head p r u)

theorem final_safe (p : Parameters) (l : Launch) (r : RealHProgram.Regions)
    (bounds : RegionBounds) (u : BlockState)
    (hb : p.storeFinal=Bool.true→StateBounds p.toLayout l r.final bounds)
    (hik : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int))) :
    Stmt.TraceSafeList bounds (finalStage p r) u := by
  rw [final_ast]
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _;exact Stmt.TraceSafeList.nil_intro)
  by_cases h:p.storeFinal
  · simp only [h]
    rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_pointer_eval p l r.final u hih hik hiv))]
    refine ⟨by simp [Stmt.TraceSafe,statePtr,Op.SafeAt],?_⟩
    apply Stmt.TraceSafeList.cons_intro
      (store2_safe r.final (l.head*p.K*p.V+l.key*p.BK*p.V+l.value*p.BV) p.BK p.BV p.BK p.BV p.V 1 0 0
        "p_h" "b_h" bounds _ (hb h) (by simp [statePointer,matrix]))
    intro _ _;exact Stmt.TraceSafeList.nil_intro
  · simp only [h]
    rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩

#axiomsClean prelude_safe
#axiomsClean seed_safe
#axiomsClean initialization_safe
#axiomsClean final_safe
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.StateSafety
