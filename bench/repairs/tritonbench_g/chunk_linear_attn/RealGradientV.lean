import bench.repairs.tritonbench_g.chunk_linear_attn.RealGradientAccum
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def keyQueryBlock (p : Parameters) (head : Nat) (m : Memory ℝ) (kb u t : Nat) : ℝ :=
  ∑ e : Fin p.BK, if kb*p.BK+e.val<p.K then
    key p head m u (kb*p.BK+e.val)*query p head m t (kb*p.BK+e.val) else 0
def valuePartial (p : Parameters) (head c : Nat) (m : Memory ℝ) (terminal : Bool) (scale : ℝ) (kb u j : Nat) : ℝ :=
  (∑ e : Fin p.BK, if kb*p.BK+e.val<p.K then
    key p head m u (kb*p.BK+e.val)*posterior p head m scale terminal c (kb*p.BK+e.val) j else 0)+
  ∑ t : Fin p.T, if t.val/p.BT=c ∧ u≤t.val then
    (keyQueryBlock p head m kb u t.val*scale)*incoming p head m t.val j else 0

theorem grad_keyquery_dot (p : Parameters) (pid : PID) (m : Memory ℝ)
    (i : TileIndex [p.BT,p.BT])
    (hu : pid.y*p.BT+i.1.val<p.T) (ht : pid.y*p.BT+i.2.1.val<p.T) :
    dot dotPrimitive (gradKey algebra p pid m) (gradQuery algebra p pid m) i=
      keyQueryBlock p pid.z m pid.x (pid.y*p.BT+i.1.val) (pid.y*p.BT+i.2.1.val) := by
  change (∑ e : Fin p.BK, gradKey algebra p pid m (i.1,e,())*
    gradQuery algebra p pid m (e,i.2.1,()))=_
  unfold keyQueryBlock
  apply Finset.sum_congr rfl
  intro e _
  by_cases he : pid.x*p.BK+e.val<p.K <;>
    simp [gradKey,gradQuery,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
      key,query,hu,ht,he] <;> simp [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]

theorem grad_value_history_dot (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (terminal : Bool) (hh : ReverseHistoryAt p pid.z pid.y m scale terminal)
    (hc : pid.y<p.NT) (vb : Nat) (i : TileIndex [p.BT,p.BV]) (hi : gradValueActive p pid vb i) :
    dot dotPrimitive (gradKey algebra p pid m) (gradHistoryGradient algebra p pid m vb) i=
      ∑ e : Fin p.BK, if pid.x*p.BK+e.val<p.K then
        key p pid.z m (pid.y*p.BT+i.1.val) (pid.x*p.BK+e.val)*
        posterior p pid.z m scale terminal pid.y (pid.x*p.BK+e.val) (vb*p.BV+i.2.1.val) else 0 := by
  change (∑ e : Fin p.BK, gradKey algebra p pid m (i.1,e,())*
    gradHistoryGradient algebra p pid m vb (e,i.2.1,()))=_
  apply Finset.sum_congr rfl
  intro e _
  by_cases he : pid.x*p.BK+e.val<p.K
  · rw [grad_reverse_read p pid m scale terminal hh hc vb (e,i.2.1,()) he hi.2]
    simp [gradKey,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,key,hi.1,he]
    simp [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  · simp [gradKey,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,he]

theorem grad_value_partial (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (terminal : Bool) (hh : ReverseHistoryAt p pid.z pid.y m scale terminal)
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hpid : pid.y<p.NT)
    (vb : Nat) (i : TileIndex [p.BT,p.BV]) (hi : gradValueActive p pid vb i) :
    gradValue algebra dotPrimitive p pid m scale vb i=
      valuePartial p pid.z pid.y m terminal scale pid.x
        (pid.y*p.BT+i.1.val) (vb*p.BV+i.2.1.val) := by
  have hs : (∑ t : Fin p.BT, gradScore algebra dotPrimitive p pid m scale (i.1,t,())*
      valueTile algebra p pid m "incoming" vb (t,i.2.1,()))=
      ∑ t : Fin p.BT, if i.1.val≤t.val then
        (if pid.y*p.BT+t.val<p.T then
          (keyQueryBlock p pid.z m pid.x (pid.y*p.BT+i.1.val) (pid.y*p.BT+t.val)*scale)*
            incoming p pid.z m (pid.y*p.BT+t.val) (vb*p.BV+i.2.1.val) else 0) else 0 := by
    apply Finset.sum_congr rfl
    intro t _
    by_cases hl : i.1.val≤t.val
    · by_cases ht : pid.y*p.BT+t.val<p.T
      · have hscore := grad_keyquery_dot p pid m (i.1,t,()) hi.1 ht
        simp only [gradScore,if_pos hl,hscore]
        simp [hl,mul,scalar,cast,valueTile,load,rawLoad,zero,algebra,
          VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,incoming,ht,hi.2]
      · simp [hl,ht,valueTile,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra]
    · simp [gradScore,hl,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra]
  change dot dotPrimitive (gradKey algebra p pid m) (gradHistoryGradient algebra p pid m vb) i+
    (∑ t : Fin p.BT, gradScore algebra dotPrimitive p pid m scale (i.1,t,())*
      valueTile algebra p pid m "incoming" vb (t,i.2.1,()))=_
  rw [grad_value_history_dot p pid m scale terminal hh hpid vb i hi,hs]
  rw [future_block p.T p.NT p.BT pid.y i.1.val hp hc hpid
    (fun t => (keyQueryBlock p pid.z m pid.x (pid.y*p.BT+i.1.val) t*scale)*
      incoming p pid.z m t (vb*p.BV+i.2.1.val))]
  rfl

theorem value_partials_complete (p : Parameters) (head c : Nat) (m : Memory ℝ)
    (seed terminal : Bool) (scale : ℝ) (u : Fin p.T) (j : Fin p.V)
    (hu : u.val/p.BT=c) (hk : p.K≤keyBlocks p*p.BK) :
    (∑ kb : Fin (keyBlocks p), valuePartial p head c m terminal scale kb.val u.val j.val)=
      (chunkGradients scale (inputs p head m seed)
        (fun t j => incoming p head m t.val j.val)
        (fun e j => endpoint p head m Bool.true terminal e.val j.val) p.BT).v u j := by
  unfold valuePartial
  rw [Finset.sum_add_distrib]
  have hfirst := blocks_complete p.K (keyBlocks p) p.BK hk
    (fun e => key p head m u.val e*posterior p head m scale terminal c e j.val)
  rw [hfirst]
  rw [Finset.sum_comm]
  have hscore : ∀ t : Fin p.T, (∑ kb : Fin (keyBlocks p), keyQueryBlock p head m kb.val u.val t.val)=
      ∑ e : Fin p.K, key p head m u.val e.val*query p head m t.val e.val := by
    intro t
    exact blocks_complete p.K (keyBlocks p) p.BK hk
      (fun e => key p head m u.val e*query p head m t.val e)
  unfold chunkGradients inputs after
  simp only [hu]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : t.val/p.BT=c ∧ u.val≤t.val
  · have ht' : c=t.val/p.BT ∧ u≤t := ⟨ht.1.symm,ht.2⟩
    simp only [if_pos ht,if_pos ht',← Finset.sum_mul,hscore]
  · have ht' : ¬(c=t.val/p.BT ∧ u≤t) := by simpa only [eq_comm,Fin.le_def] using ht
    simp only [if_neg ht,if_neg ht',Finset.sum_const_zero]

#axiomsClean grad_keyquery_dot
#axiomsClean grad_value_history_dot
#axiomsClean grad_value_partial
#axiomsClean value_partials_complete
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
