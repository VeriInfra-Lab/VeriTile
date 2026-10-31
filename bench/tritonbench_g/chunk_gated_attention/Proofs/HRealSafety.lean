import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealExecution
import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealStateSafety
import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealLoopSafety
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

theorem source_safe (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) (u : BlockState)
    (hfull : ∀ c,c<p.NT→RebasedAccess.Full p.toLayout (decoded p u) c)
    (hkey : (decoded p u).key*p.BK+p.BK≤p.K)
    (hvalue : (decoded p u).value*p.BV+p.BV≤p.V)
    (hs : RealHStep.Separate r)
    (hinj : RealHLoop.HistoryNoCollision p.toLayout (decoded p u))
    (bounds : RegionBounds) (hb : Bounds p.toLayout (decoded p u) r p.gateK p.hasSeed p.storeFinal bounds) :
    (kernel p r dot gate history seed final).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l:=decoded p u
  let a : Inputs:=⟨u.readMem r.key,u.readMem r.value,u.readMem r.gate,u.readMem r.seed⟩
  let start:=initial p r a.seed u
  have hi:=invariant p r a u (fun _=>rfl) (fun _=>rfl) (fun _=>rfl)
  have hei:=initialization_executes p r a.seed u hkey hvalue (fun _=>rfl)
  have hloop:=LoopSafety.loop_safe u p l r a hfull hs hinj bounds hb.loop start hi
  obtain ⟨mid,hel,hm⟩:=Loop.executes u p l r a hfull hs hinj start hi
  rw [source_connection,Kernel.TraceSafe]
  have hd : (algorithm p r).body=(prelude p r++seedStage p r)++loopStatement p r::finalStage p r := by
    rw [decomposition]
    simp only [List.append_assoc,List.singleton_append]
  rw [hd]
  apply TraceProofs.append_safe _ u (StateSafety.initialization_safe p r bounds u hb.initial)
  intro v hv
  rw [hei] at hv
  cases hv
  rw [TraceProofs.cons_of_run_iff hel]
  exact ⟨hloop,StateSafety.final_safe p l r bounds mid hb.final hm.key hm.value hm.head⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
