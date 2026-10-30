import bench.repairs.tritonbench_g.chunk_linear_attn.RealSums
import bench.repairs.tritonbench_g.chunk_linear_attn.GridReadback
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
abbrev algebra := VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra
abbrev dotPrimitive := VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.dot

def query (p : Parameters) (head : Nat) (m : Memory ℝ) (t e : Nat) : ℝ :=
  (m "query" (head*(p.T*p.K)+e+t*p.K)).read .real
def key (p : Parameters) (head : Nat) (m : Memory ℝ) (t e : Nat) : ℝ :=
  (m "key" (head*(p.T*p.K)+e+t*p.K)).read .real
def value (p : Parameters) (head : Nat) (m : Memory ℝ) (t j : Nat) : ℝ :=
  (m "value" (head*(p.T*p.V)+t*p.V+j)).read .real
def incoming (p : Parameters) (head : Nat) (m : Memory ℝ) (t j : Nat) : ℝ :=
  (m "incoming" (head*(p.T*p.V)+t*p.V+j)).read .real
def endpoint (p : Parameters) (head : Nat) (m : Memory ℝ) (rev seed : Bool) (e j : Nat) : ℝ :=
  if seed then (m (seedName rev) (head*p.K*p.V+e*p.V+j)).read .real else 0

def inputs (p : Parameters) (head : Nat) (m : Memory ℝ) (seed : Bool) : Inputs p.T p.K p.V where
  q := fun t e => query p head m t.val e.val
  k := fun t e => key p head m t.val e.val
  v := fun t j => value p head m t.val j.val
  initial := fun e j => endpoint p head m Bool.false seed e.val j.val

def stateTerm (p : Parameters) (head : Nat) (m : Memory ℝ) (scale : ℝ) (rev : Bool) (t e j : Nat) : ℝ :=
  if rev then (query p head m t e*scale)*incoming p head m t j
  else key p head m t e*value p head m t j
def stateBlock (p : Parameters) (head : Nat) (m : Memory ℝ) (scale : ℝ) (rev : Bool) (c e j : Nat) : ℝ :=
  ∑ t : Fin p.BT, if c*p.BT+t.val<p.T then stateTerm p head m scale rev (c*p.BT+t.val) e j else 0

theorem state_initial_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (rev seed : Bool) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    stateInitial algebra p pid m rev seed i=
      endpoint p pid.z m rev seed (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val) := by
  rcases hi with ⟨hk,hv⟩
  simp [stateInitial,endpoint,rawLoad,cast,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,hk,hv]

theorem state_dot_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (rev : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    dot dotPrimitive (stateLeft algebra p pid m scale rev n) (stateRight algebra p pid m rev n) i=
      stateBlock p pid.z m scale rev (chunk p rev n) (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val) := by
  rcases hi with ⟨hk,hv⟩
  change (∑ t : Fin p.BT, stateLeft algebra p pid m scale rev n (i.1,t,())*
    stateRight algebra p pid m rev n (t,i.2.1,()))=_
  unfold stateBlock
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : chunk p rev n*p.BT+t.val<p.T
  · cases rev <;> simp_all [stateLeft,stateRight,load,rawLoad,cast,mul,scalar,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
      stateTerm,query,key,value,incoming,chunk,Nat.add_assoc]
  · cases rev <;> simp_all [stateLeft,stateRight,load,rawLoad,cast,mul,scalar,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
      stateTerm,query,key,value,incoming,chunk]

theorem state_values_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (rev seed : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    stateValues algebra dotPrimitive p pid m scale rev seed n i=
      endpoint p pid.z m rev seed (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val)+
      ∑ c ∈ Finset.range n, stateBlock p pid.z m scale rev (chunk p rev c)
        (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val) := by
  induction n with
  | zero => simpa [stateValues] using state_initial_eq p pid m rev seed i hi
  | succ n ih =>
    change stateValues algebra dotPrimitive p pid m scale rev seed n i+
      dot dotPrimitive (stateLeft algebra p pid m scale rev n) (stateRight algebra p pid m rev n) i=_
    rw [ih,state_dot_eq p pid m scale rev n i hi,Finset.sum_range_succ,add_assoc]

theorem forward_state_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (seed : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV])
    (hp : 0<p.BT) (hi : hActive p pid i) :
    stateValues algebra dotPrimitive p pid m scale Bool.false seed n i=
      before (inputs p pid.z m seed) p.BT n ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [state_values_eq p pid m scale Bool.false seed n i hi]
  unfold before inputs
  congr 1
  simp only [chunk,Bool.false_eq_true,ite_false,stateBlock,stateTerm]
  rw [← Fin.sum_univ_eq_sum_range]
  rw [blocks_prefix p.T n p.BT (fun t => key p pid.z m t (pid.x*p.BK+i.1.val)*value p pid.z m t (pid.y*p.BV+i.2.1.val))]
  apply Finset.sum_congr rfl
  intro t _
  have he : t.val<n*p.BT ↔ t.val/p.BT<n := (Nat.div_lt_iff_lt_mul hp).symm
  simp only [he]

theorem forward_final_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (seed : Bool) (i : TileIndex [p.BK,p.BV])
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hi : hActive p pid i) :
    stateValues algebra dotPrimitive p pid m scale Bool.false seed p.NT i=
      finalState (inputs p pid.z m seed) ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [forward_state_eq p pid m scale seed p.NT i hp hi]
  unfold before finalState
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  have ht : t.val/p.BT<p.NT := (Nat.div_lt_iff_lt_mul hp).mpr (lt_of_lt_of_le t.isLt hc)
  simp only [if_pos ht]

theorem reverse_state_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (terminal : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV])
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hn : n≤p.NT) (hi : hActive p pid i) :
    stateValues algebra dotPrimitive p pid m scale Bool.true terminal n i=
      endpoint p pid.z m Bool.true terminal (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val)+
      scale*∑ t : Fin p.T, if p.NT-n≤t.val/p.BT then
        query p pid.z m t.val (pid.x*p.BK+i.1.val)*incoming p pid.z m t.val (pid.y*p.BV+i.2.1.val) else 0 := by
  rw [state_values_eq p pid m scale Bool.true terminal n i hi]
  congr 1
  simp only [chunk,ite_true]
  rw [reverse_prefix p.NT n hn (fun c => stateBlock p pid.z m scale Bool.true c (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val))]
  rw [← Fin.sum_univ_eq_sum_range]
  unfold stateBlock
  rw [blocks_select p.T p.NT p.BT hp hc (fun c => p.NT-n≤c)
    (fun t => stateTerm p pid.z m scale Bool.true t (pid.x*p.BK+i.1.val) (pid.y*p.BV+i.2.1.val))]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : p.NT-n≤t.val/p.BT <;> simp [ht,stateTerm,mul_comm,mul_left_comm,mul_assoc]

theorem backward_history_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (seed terminal : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV])
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hn : n<p.NT) (hi : hActive p pid i) :
    stateValues algebra dotPrimitive p pid m scale Bool.true terminal n i=
      after scale (inputs p pid.z m seed)
        (fun t j => incoming p pid.z m t.val j.val)
        (fun e j => endpoint p pid.z m Bool.true terminal e.val j.val)
        p.BT (chunk p Bool.true n) ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [reverse_state_eq p pid m scale terminal n i hp hc (Nat.le_of_lt hn) hi]
  unfold after inputs chunk
  congr 2
  apply Finset.sum_congr rfl
  intro t _
  have he : p.NT-n≤t.val/p.BT ↔ p.NT-1-n<t.val/p.BT := by omega
  simp only [he,ite_true]

theorem backward_final_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (scale : ℝ) (seed terminal : Bool) (i : TileIndex [p.BK,p.BV])
    (hp : 0<p.BT) (hc : p.T≤p.NT*p.BT) (hi : hActive p pid i) :
    stateValues algebra dotPrimitive p pid m scale Bool.true terminal p.NT i=
      (gradients scale (inputs p pid.z m seed)
        (fun t j => incoming p pid.z m t.val j.val)
        (fun e j => endpoint p pid.z m Bool.true terminal e.val j.val)).initial ⟨_,hi.1⟩ ⟨_,hi.2⟩ := by
  rw [reverse_state_eq p pid m scale terminal p.NT i hp hc le_rfl hi]
  simp [gradients,inputs]

#axiomsClean state_dot_eq
#axiomsClean state_values_eq
#axiomsClean forward_state_eq
#axiomsClean forward_final_eq
#axiomsClean reverse_state_eq
#axiomsClean backward_history_eq
#axiomsClean backward_final_eq
#specNonCircular inputs avoiding [run,FP.Structural.exec,stateValues]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
