import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Forward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Geometry
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.TensorVJP
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Real
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 8000000
def ops : Ops ℝ := ⟨0,1,(·+·),(·-·),(·*·),id⟩
theorem tree_leaves (t : Tree) (x : Nat → ℝ) : t.eval ops x=(t.leaves.map x).sum := by
  induction t with
  | leaf i => simp [Tree.eval,Tree.leaves]
  | zero => rfl
  | add a b ha hb =>
    change a.eval ops x+b.eval ops x=_
    rw [ha,hb]
    simp [Tree.leaves]
theorem list_range_sum (n : Nat) (x : Nat → ℝ) :
    ((List.range n).map x).sum=∑ i∈Finset.range n,x i := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih]
theorem tree_sum (n : Nat) (t : Tree) (ht : t.Legal n) (x : Nat → ℝ) :
    t.eval ops x=∑ i∈Finset.range n,x i := by
  rw [tree_leaves,(ht.map x).sum_eq,list_range_sum]
theorem padded_sum (K BK : Nat) (hk : K≤BK) (x : Nat → ℝ)
    (hz : ∀ (i : Nat), K ≤ i → x i = 0) :
    (∑ i∈Finset.range BK,x i)=∑ i∈Finset.range K,x i := by
  symm
  apply Finset.sum_subset (Finset.range_mono hk)
  intro i _ hn
  exact hz i (by simpa using hn)
theorem key_sum_tree (s : Shape) (tr : Tree) (hg : tr.Legal s.BK)
    (m : Memory ℝ) (r : Region) (bh t : Nat) (a : Nat → ℝ) :
    tr.eval ops (fun i => a i*keyLoad s ops m r bh t i)=
      ∑ i : Fin s.K,a i.val*m ⟨r,bh,t,i.val⟩ := by
  rw [tree_sum _ _ hg,padded_sum s.K s.BK (Geometry.nextPower_ge s.K)]
  · rw [←Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro i hi
    simp [keyLoad,load,i.isLt]
  · intro i hi
    simp [keyLoad,load,show ¬i<s.K by omega,ops]

theorem key_sum (s : Shape) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (m : Memory ℝ) (r : Region) (bh t : Nat) (a : Nat → ℝ) :
    (g.key s.BK).eval ops (fun i => a i*keyLoad s ops m r bh t i)=
      ∑ i : Fin s.K,a i.val*m ⟨r,bh,t,i.val⟩ := by
  rw [tree_sum _ _ hg,padded_sum s.K s.BK (Geometry.nextPower_ge s.K)]
  · rw [←Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro i hi
    simp [keyLoad,load,i.isLt]
  · intro i hi
    simp [keyLoad,load,show ¬i<s.K by omega,ops]

def column (s : Shape) (f : Flags) (m : Memory ℝ) (bh v : Nat) : Math.Input s.K :=
  {q:=fun t i => m ⟨.q,bh,t,i.val⟩,k:=fun t i => m ⟨.k,bh,t,i.val⟩,
   x:=fun t => m ⟨.v,bh,t,v⟩,
   beta:=fun t => if f.vectorBeta then m ⟨.beta,bh,t,v⟩ else m ⟨.beta,bh,t,0⟩,
   h0:=fun i => if f.initial then m ⟨.h0,bh,i.val,v⟩ else 0}
theorem h_math (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (m : Memory ℝ) (bh iv j : Nat) (hj : iv*s.FBV+j<s.V) (n : Nat) (i : Fin s.K) :
    Forward.h s f ops g m bh iv n i.val j=Math.state (column s f m bh (iv*s.FBV+j)) n i := by
  induction n generalizing i with
  | zero => simp [Forward.h,seed,load,ops,Math.state,column,i.isLt,hj]
  | succ n ih =>
    have hr : (g.key s.BK).eval ops (fun k =>
        Forward.h s f ops g m bh iv n k j*keyLoad s ops m .k bh n k)=
        Math.dot (Math.state (column s f m bh (iv*s.FBV+j)) n) ((column s f m bh (iv*s.FBV+j)).k n) := by
      rw [key_sum s g hg]
      apply Finset.sum_congr rfl
      intro k _
      rw [ih k]
      rfl
    change Forward.h s f ops g m bh iv n i.val j+
      keyLoad s ops m .k bh n i.val*
        ((valLoad s ops m .v bh n iv s.FBV j-
          (g.key s.BK).eval ops (fun k => Forward.h s f ops g m bh iv n k j*keyLoad s ops m .k bh n k))*
            betaLoad s f ops m bh n iv s.FBV j)=_
    rw [hr,ih i]
    simp only [Math.state,Math.next,Math.residual]
    simp [keyLoad,valLoad,betaLoad,load,column,i.isLt,hj]
    ring
theorem residual_math (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK)
    (m : Memory ℝ) (bh iv j t : Nat) (hj : iv*s.FBV+j<s.V) :
    Forward.residual s f ops g m bh iv t j=
      Math.residual (Math.state (column s f m bh (iv*s.FBV+j)) t)
        ((column s f m bh (iv*s.FBV+j)).k t) ((column s f m bh (iv*s.FBV+j)).x t) := by
  unfold Forward.residual
  change valLoad s ops m .v bh t iv s.FBV j-
    (g.key s.BK).eval ops (fun i => Forward.h s f ops g m bh iv t i j*keyLoad s ops m .k bh t i)=_
  rw [key_sum s g hg]
  simp only [Math.residual,Math.dot,column,valLoad,load,hj,if_true]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [h_math s f g hg m bh iv j hj t i]
  rfl
theorem output_math (s : Shape) (f : Flags) (g : Trees) (hg : (g.key s.BK).Legal s.BK) (ho : (g.readout s.BK).Legal s.BK)
    (scale : ℝ) (m : Memory ℝ) (bh iv j t : Nat) (hj : iv*s.FBV+j<s.V) :
    Forward.output s f ops g scale m bh iv t j=
      Math.output scale (column s f m bh (iv*s.FBV+j)) t := by
  unfold Forward.output
  change (g.readout s.BK).eval ops (fun i => Forward.h s f ops g m bh iv (t+1) i j*(keyLoad s ops m .q bh t i*scale))=_
  have he : (fun i => Forward.h s f ops g m bh iv (t+1) i j*(keyLoad s ops m .q bh t i*scale))=
      (fun i => (scale*Forward.h s f ops g m bh iv (t+1) i j)*keyLoad s ops m .q bh t i) := by
    funext i;ring
  rw [he,key_sum_tree s (g.readout s.BK) ho]
  simp only [Math.output,Math.dot,column,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [h_math s f g hg m bh iv j hj (t+1) i]
  dsimp only [column]
  ring

#axiomsClean tree_sum
#axiomsClean h_math
#axiomsClean residual_math
#axiomsClean output_math
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Real
