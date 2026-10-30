import bench.repairs.tritonbench_g.chunk_linear_attn.RealGradientAccum
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem grad_query_chunk (p : Parameters) (pid : PID) (m : Memory ℝ) (seed terminal : Bool) (scale : ℝ)
    (hh : HistoryAt p pid.z pid.y m seed) (hv : p.V≤valueBlocks p*p.BV)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradQueryResult algebra dotPrimitive p pid m scale (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)) i=
      (chunkGradients scale (inputs p pid.z m seed)
        (fun t j => incoming p pid.z m t.val j.val)
        (fun e j => endpoint p pid.z m Bool.true terminal e.val j.val)
        p.BT).q ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  let a := gradStates algebra dotPrimitive p pid m scale (valueBlocks p)
  have hs : (∑ u : Fin p.BT, (if u.val≤ i.1.val then a.score (i.1,u,())*scale else 0)*
      gradKey algebra p pid m (u,i.2.1,()))=
      ∑ u : Fin p.BT, if u.val≤ i.1.val then
        (if pid.y*p.BT+u.val<p.T then
          (∑ j : Fin p.V, incoming p pid.z m (pid.y*p.BT+i.1.val) j.val*
            value p pid.z m (pid.y*p.BT+u.val) j.val)*scale*
            key p pid.z m (pid.y*p.BT+u.val) (pid.x*p.BK+i.2.1.val) else 0) else 0 := by
    apply Finset.sum_congr rfl
    intro u _
    by_cases hu : u.val≤ i.1.val
    · have ht : pid.y*p.BT+u.val<p.T := by have := hi.1; omega
      have hscore := grad_score_acc p pid m scale hv (i.1,u,()) hi.1 ht
      change a.score (i.1,u,())=_ at hscore
      simp only [if_pos hu,if_pos ht,hscore]
      simp [gradKey,load,rawLoad,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,key,ht,hi.2]
      simp [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
    · simp [hu]
  change a.query i+(∑ u : Fin p.BT, (if u.val≤ i.1.val then a.score (i.1,u,())*scale else 0)*
    gradKey algebra p pid m (u,i.2.1,()))=_
  rw [hs,grad_query_acc p pid m seed scale hh hpid hv i hi]
  rw [causal_block p.T p.NT p.BT pid.y i.1.val hp hc hpid
    (fun u => (∑ j : Fin p.V, incoming p pid.z m (pid.y*p.BT+i.1.val) j.val*value p pid.z m u j.val)*scale*
      key p pid.z m u (pid.x*p.BK+i.2.1.val))]
  simp only [chunkGradients,inputs,before,prior,block_quotient pid.y p.BT i.1.val i.1.isLt,Fin.le_def]

theorem grad_key_chunk (p : Parameters) (pid : PID) (m : Memory ℝ) (seed terminal : Bool) (scale : ℝ)
    (hh : ReverseHistoryAt p pid.z pid.y m scale terminal) (hv : p.V≤valueBlocks p*p.BV)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradKeyResult algebra dotPrimitive p pid m scale (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)) i=
      (chunkGradients scale (inputs p pid.z m seed)
        (fun t j => incoming p pid.z m t.val j.val)
        (fun e j => endpoint p pid.z m Bool.true terminal e.val j.val)
        p.BT).k ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  let a := gradStates algebra dotPrimitive p pid m scale (valueBlocks p)
  have hs : (∑ t : Fin p.BT, gradQuery algebra p pid m (i.2.1,t,())*
      (if i.1.val≤t.val then a.score (t,i.1,())*scale else 0))=
      ∑ t : Fin p.BT, if i.1.val≤t.val then
        (if pid.y*p.BT+t.val<p.T then
          query p pid.z m (pid.y*p.BT+t.val) (pid.x*p.BK+i.2.1.val)*
          ((∑ j : Fin p.V, incoming p pid.z m (pid.y*p.BT+t.val) j.val*
            value p pid.z m (pid.y*p.BT+i.1.val) j.val)*scale) else 0) else 0 := by
    apply Finset.sum_congr rfl
    intro t _
    by_cases hu : i.1.val≤t.val
    · by_cases ht : pid.y*p.BT+t.val<p.T
      · have hscore := grad_score_acc p pid m scale hv (t,i.1,()) ht hi.1
        change a.score (t,i.1,())=_ at hscore
        simp only [if_pos hu,if_pos ht,hscore]
        simp [gradQuery,load,rawLoad,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,query,ht,hi.2]
      · simp [hu,ht,gradQuery,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra]
    · simp [hu]
  change a.key i+(∑ t : Fin p.BT, gradQuery algebra p pid m (i.2.1,t,())*
      (if i.1.val≤t.val then a.score (t,i.1,())*scale else 0))=_
  rw [hs,grad_key_acc p pid m terminal scale hh hpid hv i hi]
  rw [future_block p.T p.NT p.BT pid.y i.1.val hp hc hpid
    (fun t => query p pid.z m t (pid.x*p.BK+i.2.1.val)*
      ((∑ j : Fin p.V, incoming p pid.z m t j.val*value p pid.z m (pid.y*p.BT+i.1.val) j.val)*scale))]
  simp only [chunkGradients,inputs,after,posterior,block_quotient pid.y p.BT i.1.val i.1.isLt,Fin.le_def,eq_comm]

theorem grad_query_correct (p : Parameters) (pid : PID) (m : Memory ℝ) (seed terminal : Bool) (scale : ℝ)
    (hh : HistoryAt p pid.z pid.y m seed) (hv : p.V≤valueBlocks p*p.BV)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradQueryResult algebra dotPrimitive p pid m scale (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)) i=
      (gradients scale (inputs p pid.z m seed)
        (fun t j => incoming p pid.z m t.val j.val)
        (fun e j => endpoint p pid.z m Bool.true terminal e.val j.val)).q ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [grad_query_chunk p pid m seed terminal scale hh hv hp hc hpid i hi,chunkGradients_correct]

theorem grad_key_correct (p : Parameters) (pid : PID) (m : Memory ℝ) (seed terminal : Bool) (scale : ℝ)
    (hh : ReverseHistoryAt p pid.z pid.y m scale terminal) (hv : p.V≤valueBlocks p*p.BV)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradKeyResult algebra dotPrimitive p pid m scale (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)) i=
      (gradients scale (inputs p pid.z m seed)
        (fun t j => incoming p pid.z m t.val j.val)
        (fun e j => endpoint p pid.z m Bool.true terminal e.val j.val)).k ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [grad_key_chunk p pid m seed terminal scale hh hv hp hc hpid i hi,chunkGradients_correct]

#axiomsClean grad_query_chunk
#axiomsClean grad_key_chunk
#axiomsClean grad_query_correct
#axiomsClean grad_key_correct
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
