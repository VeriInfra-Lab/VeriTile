import bench.repairs.tritonbench_g.chunk_linear_attn.RealState
import bench.repairs.tritonbench_g.chunk_linear_attn.RealWindow
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def prior (p : Parameters) (head : Nat) (m : Memory ℝ) (seed : Bool) (c e j : Nat) : ℝ :=
  endpoint p head m Bool.false seed e j+
    ∑ t : Fin p.T, if t.val/p.BT<c then key p head m t.val e*value p head m t.val j else 0
def posterior (p : Parameters) (head : Nat) (m : Memory ℝ) (scale : ℝ) (terminal : Bool) (c e j : Nat) : ℝ :=
  endpoint p head m Bool.true terminal e j+
    scale*∑ t : Fin p.T, if c<t.val/p.BT then query p head m t.val e*incoming p head m t.val j else 0

def HistoryAt (p : Parameters) (head c : Nat) (m : Memory ℝ) (seed : Bool) : Prop :=
  ∀ e j, e<p.K → j<p.V →
    (m "history" (head*(p.NT*p.K*p.V)+c*p.K*p.V+e*p.V+j)).read .real=prior p head m seed c e j

def ReverseHistoryAt (p : Parameters) (head c : Nat) (m : Memory ℝ) (scale : ℝ) (terminal : Bool) : Prop :=
  ∀ e j, e<p.K → j<p.V →
    (m "historyGradient" (head*(p.NT*p.K*p.V)+c*p.K*p.V+e*p.V+j)).read .real=posterior p head m scale terminal c e j

theorem output_states_expanded (p : Parameters) (pid : PID) (m : Memory ℝ) (n : Nat) :
    (∀ i, (outputStates algebra dotPrimitive p pid m n).output i=
      ∑ kb ∈ Finset.range n, dot dotPrimitive (queryTile algebra p pid m kb) (forwardHistoryTile algebra p pid m kb) i) ∧
    (∀ i, (outputStates algebra dotPrimitive p pid m n).score i=
      ∑ kb ∈ Finset.range n, dot dotPrimitive (queryTile algebra p pid m kb) (keyTile algebra p pid m kb) i) := by
  induction n with
  | zero => simp [outputStates,outputInitial,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra]
  | succ n ih =>
    constructor
    · intro i
      change (outputStates algebra dotPrimitive p pid m n).output i+dot dotPrimitive (queryTile algebra p pid m n) (forwardHistoryTile algebra p pid m n) i=_
      rw [ih.1,Finset.sum_range_succ]
    · intro i
      change (outputStates algebra dotPrimitive p pid m n).score i+dot dotPrimitive (queryTile algebra p pid m n) (keyTile algebra p pid m n) i=_
      rw [ih.2,Finset.sum_range_succ]

theorem output_history_dot (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool)
    (hh : HistoryAt p pid.z pid.y m seed) (kb : Nat)
    (i : TileIndex [p.BT,p.BV]) (hi : outputActive p pid i) :
    dot dotPrimitive (queryTile algebra p pid m kb) (forwardHistoryTile algebra p pid m kb) i=
      ∑ e : Fin p.BK, if kb*p.BK+e.val<p.K then
        query p pid.z m (pid.y*p.BT+i.1.val) (kb*p.BK+e.val)*
        prior p pid.z m seed pid.y (kb*p.BK+e.val) (pid.x*p.BV+i.2.1.val) else 0 := by
  rcases hi with ⟨ht,hv⟩
  change (∑ e : Fin p.BK, queryTile algebra p pid m kb (i.1,e,())*
    forwardHistoryTile algebra p pid m kb (e,i.2.1,()))=_
  apply Finset.sum_congr rfl
  intro e _
  by_cases hk : kb*p.BK+e.val<p.K
  · have hs := hh (kb*p.BK+e.val) (pid.x*p.BV+i.2.1.val) hk hv
    simp only [Nat.add_assoc] at hs
    simp [queryTile,forwardHistoryTile,load,rawLoad,zero,algebra,
      VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,query,ht,hv,hk,hs,Nat.add_assoc]
    simp [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  · simp [queryTile,forwardHistoryTile,load,rawLoad,zero,algebra,
      VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,hk]

theorem output_score_dot (p : Parameters) (pid : PID) (m : Memory ℝ) (kb : Nat)
    (i : TileIndex [p.BT,p.BT])
    (ht : pid.y*p.BT+i.1.val<p.T) (hu : pid.y*p.BT+i.2.1.val<p.T) :
    dot dotPrimitive (queryTile algebra p pid m kb) (keyTile algebra p pid m kb) i=
      ∑ e : Fin p.BK, if kb*p.BK+e.val<p.K then
        query p pid.z m (pid.y*p.BT+i.1.val) (kb*p.BK+e.val)*
        key p pid.z m (pid.y*p.BT+i.2.1.val) (kb*p.BK+e.val) else 0 := by
  change (∑ e : Fin p.BK, queryTile algebra p pid m kb (i.1,e,())*
    keyTile algebra p pid m kb (e,i.2.1,()))=_
  apply Finset.sum_congr rfl
  intro e _
  by_cases hk : kb*p.BK+e.val<p.K <;>
    simp [queryTile,keyTile,load,rawLoad,zero,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,
      query,key,ht,hu,hk,Nat.add_assoc] <;>
    simp [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]

theorem output_acc_eq (p : Parameters) (pid : PID) (m : Memory ℝ) (seed : Bool)
    (hh : HistoryAt p pid.z pid.y m seed) (hk : p.K≤keyBlocks p*p.BK)
    (i : TileIndex [p.BT,p.BV]) (hi : outputActive p pid i) :
    (outputStates algebra dotPrimitive p pid m (keyBlocks p)).output i=
      ∑ e : Fin p.K, query p pid.z m (pid.y*p.BT+i.1.val) e.val*
        prior p pid.z m seed pid.y e.val (pid.x*p.BV+i.2.1.val) := by
  rw [(output_states_expanded p pid m _).1]
  simp_rw [output_history_dot p pid m seed hh _ i hi]
  rw [← Fin.sum_univ_eq_sum_range]
  exact blocks_complete p.K (keyBlocks p) p.BK hk (fun e => query p pid.z m (pid.y*p.BT+i.1.val) e*prior p pid.z m seed pid.y e (pid.x*p.BV+i.2.1.val))

theorem output_score_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (hk : p.K≤keyBlocks p*p.BK) (i : TileIndex [p.BT,p.BT])
    (ht : pid.y*p.BT+i.1.val<p.T) (hu : pid.y*p.BT+i.2.1.val<p.T) :
    (outputStates algebra dotPrimitive p pid m (keyBlocks p)).score i=
      ∑ e : Fin p.K, query p pid.z m (pid.y*p.BT+i.1.val) e.val*
        key p pid.z m (pid.y*p.BT+i.2.1.val) e.val := by
  rw [(output_states_expanded p pid m _).2]
  simp_rw [output_score_dot p pid m _ i ht hu]
  rw [← Fin.sum_univ_eq_sum_range]
  exact blocks_complete p.K (keyBlocks p) p.BK hk (fun e => query p pid.z m (pid.y*p.BT+i.1.val) e*key p pid.z m (pid.y*p.BT+i.2.1.val) e)

#axiomsClean output_states_expanded
#axiomsClean output_history_dot
#axiomsClean output_score_dot
#axiomsClean output_acc_eq
#axiomsClean output_score_eq
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
