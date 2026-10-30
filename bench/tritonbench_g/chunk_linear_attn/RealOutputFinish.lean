import bench.tritonbench_g.chunk_linear_attn.RealOutput
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem output_result_chunk (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool) (scale : ℝ)
    (hh : HistoryAt p pid.z pid.y m seed) (hk : p.K≤keyBlocks p*p.BK)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (i : TileIndex [p.BT,p.BV]) (hi : outputActive p pid i) :
    outputResult algebra dotPrimitive p pid m scale (outputStates algebra dotPrimitive p pid m (keyBlocks p)) i=
      chunkOutput scale (inputs p pid.z m seed) p.BT ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  let a := outputStates algebra dotPrimitive p pid m (keyBlocks p)
  have hs : (∑ u : Fin p.BT, (if u.val≤ i.1.val then a.score (i.1,u,()) else 0)*
      valueTile algebra p pid m "value" pid.x (u,i.2.1,()))=
      ∑ u : Fin p.BT, if u.val≤ i.1.val then
        (if pid.y*p.BT+u.val<p.T then
          (∑ e : Fin p.K, query p pid.z m (pid.y*p.BT+i.1.val) e.val*
            key p pid.z m (pid.y*p.BT+u.val) e.val)*
            value p pid.z m (pid.y*p.BT+u.val) (pid.x*p.BV+i.2.1.val) else 0) else 0 := by
    apply Finset.sum_congr rfl
    intro u _
    by_cases hu : u.val≤ i.1.val
    · have ht : pid.y*p.BT+u.val<p.T := by have := hi.1; omega
      have hscore := output_score_eq p pid m hk (i.1,u,()) hi.1 ht
      change a.score (i.1,u,())=_ at hscore
      simp only [if_pos hu,if_pos ht,hscore]
      simp [valueTile,load,rawLoad,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
        value,ht,hi.2]
    · simp [hu]
  change (a.output i+(∑ u : Fin p.BT, (if u.val≤ i.1.val then a.score (i.1,u,()) else 0)*
      valueTile algebra p pid m "value" pid.x (u,i.2.1,())))*scale=_
  rw [hs,output_acc_eq p pid m seed hh hk i hi]
  rw [causal_block p.T p.NT p.BT pid.y i.1.val hp hc hpid
    (fun u => (∑ e : Fin p.K, query p pid.z m (pid.y*p.BT+i.1.val) e.val*key p pid.z m u e.val)*
      value p pid.z m u (pid.x*p.BV+i.2.1.val))]
  simp [chunkOutput,inputs,before,prior,block_quotient pid.y p.BT i.1.val i.1.isLt,Fin.le_def]

theorem output_result_correct (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool) (scale : ℝ)
    (hh : HistoryAt p pid.z pid.y m seed) (hk : p.K≤keyBlocks p*p.BK)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (i : TileIndex [p.BT,p.BV]) (hi : outputActive p pid i) :
    outputResult algebra dotPrimitive p pid m scale (outputStates algebra dotPrimitive p pid m (keyBlocks p)) i=
      output scale (inputs p pid.z m seed) ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [output_result_chunk p pid m seed scale hh hk hp hc hpid i hi]
  exact chunkOutput_correct _ _ _ _ _

#axiomsClean output_result_chunk
#axiomsClean output_result_correct
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
