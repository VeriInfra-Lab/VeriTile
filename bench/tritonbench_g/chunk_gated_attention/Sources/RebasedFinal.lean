import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedInit
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHFinal
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedFinal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
abbrev base := RealHFinal.base
abbrev footprint := RealHFinal.footprint
def output (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState) : BlockState :=
  if storeFinal then RealHStore.scatter p l r.final (base p l) p.V 1 f
    (u.setReg "p_h" .blockPtr [p.BK,p.BV] (RebasedAccess.statePointer p l r.final))
  else u

theorem executes (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    stepStmts (RebasedProgram.finalization p r storeFinal) u = some (output p l r storeFinal f u) := by
  unfold RebasedProgram.finalization
  cases storeFinal
  · rw [stepStmts.cons_some (RealHInit.if_false _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [RealHInit.if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (RebasedAccess.state_eval p l r.final u hih hik hiv))]
      erw [stepStmts.cons_some (RebasedStore.store_eval p l r.final (base p l) p.V 1 f "p_h" _ hk hv
        (by simp [hh]) (by simp [RebasedAccess.statePointer,RebasedStore.shifted,base,RealHFinal.base])),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem readback (p : Layout) (l : Launch) (r : Regions) (f : StateValue p) (u : BlockState)
    (hinj : RealHStore.NoCollision p l (base p l) p.V 1)
    (e : Fin p.BK) (j : Fin p.BV) (ha : RealHStore.active p l e.val j.val) :
    (output p l r Bool.true f u).readMem r.final
      (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)) = f e j := by
  simpa [output,RealHStore.address,base,stateAddress] using
    RealHStore.readback p l r.final (base p l) p.V 1 f _ hinj e j ha

theorem frame (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l r storeFinal) u (output p l r storeFinal f u) := by
  cases storeFinal
  · intro _ _ _; rfl
  · intro region off hn
    apply RealHStore.frame p l r.final (base p l) p.V 1 f
      (u.setReg "p_h" .blockPtr [p.BK,p.BV] (RebasedAccess.statePointer p l r.final)) region off
    intro h
    exact hn ⟨rfl,h⟩

theorem unchanged (p : Layout) (l : Launch) (r : Regions) (f : StateValue p) (u : BlockState) :
    (output p l r Bool.false f u).mem = u.mem := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedFinal
