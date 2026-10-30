import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedFinal
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHExecution

/-! Complete repaired typed H source, all chunks and all source storage/gate/
seed/final variants. The unchanged independent closed form and output/frame
interfaces are imported. Full tiles follow from staging, not an input-shape
restriction; no oracle-value premise remains. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
abbrev footprint := RealHExecution.footprint
abbrev Outputs := RealHExecution.Outputs

theorem executes (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p (RebasedInit.decoded p base u) c)
    (hkey : (RebasedInit.decoded p base u).key*p.BK+p.BK ≤ p.K)
    (hvalue : (RebasedInit.decoded p base u).value*p.BV+p.BV ≤ p.V)
    (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (RebasedInit.decoded p base u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (RebasedInit.decoded p base u) (RealHFinal.base p (RebasedInit.decoded p base u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off) :
    ∃ v, exec (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel u = some v ∧
      Outputs p (RebasedInit.decoded p base u) r a gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint p (RebasedInit.decoded p base u) r storeFinal) u v := by
  let l := RebasedInit.decoded p base u
  let start := RebasedInit.initial p r a.seed base hasSeed u
  have hei := RebasedInit.executes p r a.seed base hasSeed u hkey hvalue hseed
  obtain ⟨mid,hel,hm⟩ := RebasedLoop.executes u p l r a gateK hasSeed swapped hfull hs hhi start
    (RebasedInit.invariant p r a base gateK hasSeed u hk hv hg)
  let f := expected p l a gateK hasSeed p.NT
  let v := RebasedFinal.output p l r storeFinal f mid
  have hef := RebasedFinal.executes p l r storeFinal f mid hkey hvalue hm.key hm.value hm.head hm.carry
  have hf := RebasedFinal.frame p l r storeFinal f mid
  refine ⟨v,?_,⟨?_,?_⟩,?_⟩
  · rw [exec,RebasedProgram.source_body,RebasedProgram.body,List.append_assoc,stepStmts.append_some hei]
    change stepStmts (RebasedLoop.statement p r gateK swapped :: RebasedProgram.finalization p r storeFinal) start = some v
    rw [stepStmts.cons_some hel]
    exact hef
  · intro c e j hc ha
    have hframe := hf r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))
      (fun h => hfsep h.1 h.2.1)
    unfold BlockState.readMem
    rw [← hframe]
    exact hm.history c e j hc ha
  · intro hon e j ha
    subst storeFinal
    exact RebasedFinal.readback p l r f mid (hfi rfl) e j ha
  · intro region off hn
    exact (hm.frame region off (fun h => hn (Or.inl h))).trans
      (hf region off (fun h => hn (Or.inr h)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedExecution
