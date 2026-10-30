import bench.repairs.tritonbench_g.chunk_linear_attn.RealOutputFinish
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem grad_history_read (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool)
    (hh : HistoryAt p pid.z pid.y m seed) (hc : pid.y<p.NT) (vb : Nat)
    (i : TileIndex [p.BV,p.BK])
    (he : pid.x*p.BK+i.2.1.val<p.K) (hj : vb*p.BV+i.1.val<p.V) :
    gradHistory algebra p pid m vb i=
      prior p pid.z m seed pid.y (pid.x*p.BK+i.2.1.val) (vb*p.BV+i.1.val) := by
  have hb : pid.y*p.K+pid.x*p.BK+i.2.1.val<p.NT*p.K := by simpa only [Nat.add_assoc] using packed_lt hc he
  have hs := hh _ _ he hj
  simp only [gradHistory,load,rawLoad]
  simp only [hb,hj,and_self,ite_true]
  change (m "history" (pid.z*(p.NT*p.K*p.V)+(vb*p.BV+i.1.val)*1+
    (pid.y*p.K+pid.x*p.BK+i.2.1.val)*p.V)).read .real=_
  convert hs using 1 <;> congr 2 <;> ring

theorem grad_reverse_read (p : Parameters) (pid : PID) (m : Memory ℝ) (scale : ℝ) (terminal : Bool)
    (hh : ReverseHistoryAt p pid.z pid.y m scale terminal) (hc : pid.y<p.NT) (vb : Nat)
    (i : TileIndex [p.BK,p.BV])
    (he : pid.x*p.BK+i.1.val<p.K) (hj : vb*p.BV+i.2.1.val<p.V) :
    gradHistoryGradient algebra p pid m vb i=
      posterior p pid.z m scale terminal pid.y (pid.x*p.BK+i.1.val) (vb*p.BV+i.2.1.val) := by
  have hb : pid.y*p.K+pid.x*p.BK+i.1.val<p.NT*p.K := by simpa only [Nat.add_assoc] using packed_lt hc he
  have hs := hh _ _ he hj
  simp only [gradHistoryGradient,load,rawLoad]
  simp only [hb,hj,and_self,ite_true]
  change (m "historyGradient" (pid.z*(p.NT*p.K*p.V)+
    (pid.y*p.K+pid.x*p.BK+i.1.val)*p.V+(vb*p.BV+i.2.1.val)*1)).read .real=_
  convert hs using 1 <;> congr 2 <;> ring

theorem grad_query_dot (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool)
    (hh : HistoryAt p pid.z pid.y m seed) (hc : pid.y<p.NT) (vb : Nat)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    dot dotPrimitive (valueTile algebra p pid m "incoming" vb) (gradHistory algebra p pid m vb) i=
      ∑ j : Fin p.BV, if vb*p.BV+j.val<p.V then
        incoming p pid.z m (pid.y*p.BT+i.1.val) (vb*p.BV+j.val)*
        prior p pid.z m seed pid.y (pid.x*p.BK+i.2.1.val) (vb*p.BV+j.val) else 0 := by
  change (∑ j : Fin p.BV, valueTile algebra p pid m "incoming" vb (i.1,j,())*
    gradHistory algebra p pid m vb (j,i.2.1,()))=_
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : vb*p.BV+j.val<p.V
  · rw [grad_history_read p pid m seed hh hc vb (j,i.2.1,()) hi.2 hj]
    simp [valueTile,load,rawLoad,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,incoming,hi.1,hj]
  · simp [valueTile,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,hj]

theorem grad_key_dot (p : Parameters) (pid : PID) (m : Memory ℝ) (scale : ℝ) (terminal : Bool)
    (hh : ReverseHistoryAt p pid.z pid.y m scale terminal) (hc : pid.y<p.NT) (vb : Nat)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    dot dotPrimitive (valueTile algebra p pid m "value" vb) (transpose (gradHistoryGradient algebra p pid m vb)) i=
      ∑ j : Fin p.BV, if vb*p.BV+j.val<p.V then
        value p pid.z m (pid.y*p.BT+i.1.val) (vb*p.BV+j.val)*
        posterior p pid.z m scale terminal pid.y (pid.x*p.BK+i.2.1.val) (vb*p.BV+j.val) else 0 := by
  change (∑ j : Fin p.BV, valueTile algebra p pid m "value" vb (i.1,j,())*
    gradHistoryGradient algebra p pid m vb (i.2.1,j,()))=_
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : vb*p.BV+j.val<p.V
  · rw [grad_reverse_read p pid m scale terminal hh hc vb (i.2.1,j,()) hi.2 hj]
    simp [valueTile,load,rawLoad,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,value,hi.1,hj]
  · simp [valueTile,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,hj]

theorem grad_score_dot (p : Parameters) (pid : PID) (m : Memory ℝ) (vb : Nat)
    (i : TileIndex [p.BT,p.BT])
    (ht : pid.y*p.BT+i.1.val<p.T) (hu : pid.y*p.BT+i.2.1.val<p.T) :
    dot dotPrimitive (valueTile algebra p pid m "incoming" vb) (transpose (valueTile algebra p pid m "value" vb)) i=
      ∑ j : Fin p.BV, if vb*p.BV+j.val<p.V then
        incoming p pid.z m (pid.y*p.BT+i.1.val) (vb*p.BV+j.val)*
        value p pid.z m (pid.y*p.BT+i.2.1.val) (vb*p.BV+j.val) else 0 := by
  change (∑ j : Fin p.BV, valueTile algebra p pid m "incoming" vb (i.1,j,())*
    valueTile algebra p pid m "value" vb (i.2.1,j,()))=_
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : vb*p.BV+j.val<p.V <;>
    simp [valueTile,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,incoming,value,ht,hu,hj]

theorem grad_states_expanded (p : Parameters) (pid : PID) (m : Memory ℝ) (scale : ℝ) (n : Nat) :
    (∀ i, (gradStates algebra dotPrimitive p pid m scale n).query i=
      ∑ vb ∈ Finset.range n, dot dotPrimitive (valueTile algebra p pid m "incoming" vb) (gradHistory algebra p pid m vb) i*scale) ∧
    (∀ i, (gradStates algebra dotPrimitive p pid m scale n).key i=
      ∑ vb ∈ Finset.range n, dot dotPrimitive (valueTile algebra p pid m "value" vb) (transpose (gradHistoryGradient algebra p pid m vb)) i) ∧
    (∀ i, (gradStates algebra dotPrimitive p pid m scale n).score i=
      ∑ vb ∈ Finset.range n, dot dotPrimitive (valueTile algebra p pid m "incoming" vb) (transpose (valueTile algebra p pid m "value" vb)) i) := by
  induction n with
  | zero => simp [gradStates,gradInitial,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra]
  | succ n ih =>
    constructor
    · intro i
      change (gradStates algebra dotPrimitive p pid m scale n).query i+
        dot dotPrimitive (valueTile algebra p pid m "incoming" n) (gradHistory algebra p pid m n) i*scale=_
      rw [ih.1,Finset.sum_range_succ]
    · constructor
      · intro i
        change (gradStates algebra dotPrimitive p pid m scale n).key i+
          dot dotPrimitive (valueTile algebra p pid m "value" n) (transpose (gradHistoryGradient algebra p pid m n)) i=_
        rw [ih.2.1,Finset.sum_range_succ]
      · intro i
        change (gradStates algebra dotPrimitive p pid m scale n).score i+
          dot dotPrimitive (valueTile algebra p pid m "incoming" n) (transpose (valueTile algebra p pid m "value" n)) i=_
        rw [ih.2.2,Finset.sum_range_succ]

theorem grad_query_acc (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool) (scale : ℝ)
    (hh : HistoryAt p pid.z pid.y m seed) (hc : pid.y<p.NT) (hv : p.V≤valueBlocks p*p.BV)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)).query i=
      (∑ j : Fin p.V, incoming p pid.z m (pid.y*p.BT+i.1.val) j.val*
        prior p pid.z m seed pid.y (pid.x*p.BK+i.2.1.val) j.val)*scale := by
  rw [(grad_states_expanded p pid m scale _).1,← Finset.sum_mul]
  simp_rw [grad_query_dot p pid m seed hh hc _ i hi]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  exact blocks_complete p.V (valueBlocks p) p.BV hv
    (fun j => incoming p pid.z m (pid.y*p.BT+i.1.val) j*prior p pid.z m seed pid.y (pid.x*p.BK+i.2.1.val) j)

theorem grad_key_acc (p : Parameters) (pid : PID) (m : Memory ℝ) (terminal : Bool) (scale : ℝ)
    (hh : ReverseHistoryAt p pid.z pid.y m scale terminal) (hc : pid.y<p.NT) (hv : p.V≤valueBlocks p*p.BV)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)).key i=
      ∑ j : Fin p.V, value p pid.z m (pid.y*p.BT+i.1.val) j.val*
        posterior p pid.z m scale terminal pid.y (pid.x*p.BK+i.2.1.val) j.val := by
  rw [(grad_states_expanded p pid m scale _).2.1]
  simp_rw [grad_key_dot p pid m scale terminal hh hc _ i hi]
  rw [← Fin.sum_univ_eq_sum_range]
  exact blocks_complete p.V (valueBlocks p) p.BV hv
    (fun j => value p pid.z m (pid.y*p.BT+i.1.val) j*posterior p pid.z m scale terminal pid.y (pid.x*p.BK+i.2.1.val) j)

theorem grad_score_acc (p : Parameters) (pid : PID) (m : Memory ℝ) (scale : ℝ)
    (hv : p.V≤valueBlocks p*p.BV) (i : TileIndex [p.BT,p.BT])
    (ht : pid.y*p.BT+i.1.val<p.T) (hu : pid.y*p.BT+i.2.1.val<p.T) :
    (gradStates algebra dotPrimitive p pid m scale (valueBlocks p)).score i=
      ∑ j : Fin p.V, incoming p pid.z m (pid.y*p.BT+i.1.val) j.val*
        value p pid.z m (pid.y*p.BT+i.2.1.val) j.val := by
  rw [(grad_states_expanded p pid m scale _).2.2]
  simp_rw [grad_score_dot p pid m _ i ht hu]
  rw [← Fin.sum_univ_eq_sum_range]
  exact blocks_complete p.V (valueBlocks p) p.BV hv
    (fun j => incoming p pid.z m (pid.y*p.BT+i.1.val) j*value p pid.z m (pid.y*p.BT+i.2.1.val) j)

#axiomsClean grad_history_read
#axiomsClean grad_reverse_read
#axiomsClean grad_states_expanded
#axiomsClean grad_query_acc
#axiomsClean grad_key_acc
#axiomsClean grad_score_acc
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
