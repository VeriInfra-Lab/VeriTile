import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealInit
import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealFinal
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHExecution
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

theorem executes (p : Parameters) (r : RealHProgram.Regions) (dot gate history seed final : FloatDType)
    (a : Inputs) (u : BlockState)
    (hfull : ∀ c,c<p.NT→RebasedAccess.Full p.toLayout (decoded p u) c)
    (hkey : (decoded p u).key*p.BK+p.BK≤p.K)
    (hvalue : (decoded p u).value*p.BV+p.BV≤p.V)
    (hs : RealHStep.Separate r)
    (hfsep : p.storeFinal=Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p.toLayout (decoded p u))
    (hfi : p.storeFinal=Bool.true→RealHStore.NoCollision p.toLayout (decoded p u)
      (RealHFinal.base p.toLayout (decoded p u)) p.V 1)
    (hk : ∀ off,u.readMem r.key off=a.key off)
    (hv : ∀ off,u.readMem r.value off=a.value off)
    (hg : ∀ off,u.readMem r.gate off=a.gate off)
    (hseed : ∀ off,u.readMem r.seed off=a.seed off) :
    ∃ v,exec (kernel p r dot gate history seed final).eraseDType.toAlgKernel u=some v ∧
      RealHExecution.Outputs p.toLayout (decoded p u) r a p.gateK p.hasSeed p.storeFinal v ∧
      BlockState.WriteWithin (RealHExecution.footprint p.toLayout (decoded p u) r p.storeFinal) u v := by
  let l:=decoded p u
  let start:=initial p r a.seed u
  have hei:=initialization_executes p r a.seed u hkey hvalue hseed
  obtain ⟨mid,hel,hm⟩:=Loop.executes u p l r a hfull hs hhi start (invariant p r a u hk hv hg)
  let f:=expected p.toLayout l a p.gateK p.hasSeed p.NT
  let v:=Final.output p l r p.storeFinal f mid
  have hef:=Final.executes p l r f mid hkey hvalue hm.key hm.value hm.head hm.carry
  have hf:=Final.frame p l r p.storeFinal f mid
  refine ⟨v,?_,⟨?_,?_⟩,?_⟩
  · rw [source_connection]
    have hd : (algorithm p r).body=(prelude p r++seedStage p r)++loopStatement p r::finalStage p r := by
      rw [decomposition]
      simp only [List.append_assoc,List.singleton_append]
    rw [exec,hd,stepStmts.append_some hei,stepStmts.cons_some hel]
    exact hef
  · intro c e j hc ha
    have hframe:=hf r.history
      (historyAddress p.toLayout l c (RealHMath.keyIndex p.toLayout l e.val) (RealHMath.valueIndex p.toLayout l j.val))
      (fun h=>hfsep h.1 h.2.1)
    unfold BlockState.readMem
    rw [←hframe]
    exact hm.history c e j hc ha
  · intro hon e j ha
    have hr:=Final.readback p l r f mid (hfi hon) e j ha
    simpa only [v,f,expected,hon] using hr
  · intro region off hn
    exact (hm.frame region off (fun h=>hn (Or.inl h))).trans
      (hf region off (fun h=>hn (Or.inr h)))

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
