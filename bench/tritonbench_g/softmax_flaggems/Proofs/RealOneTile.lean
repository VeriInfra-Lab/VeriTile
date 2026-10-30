import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardValue

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems RealMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 5000000

theorem sum_padding (N B : Nat) (hB : 0<B) (covered : N≤B) (v : Nat → ℝ) :
    (∑ i : Fin B, if i.val<N then v i.val else 0) = ∑ n : Fin N, v n.val := by
  have part := lane_tile_partition N B hB 1 (by simpa using covered) v
  simpa [Fin.sum_univ_eq_sum_range] using part

def padded (N B : Nat) (q : Nat → WithBot ℝ) (i : Fin B) : WithBot ℝ :=
  if i.val<N then q i.val else ⊥

def oneMax (N B : Nat) (q : Nat → WithBot ℝ) := Ideal.maximum (padded N B q)
def oneSum (N B : Nat) (q : Nat → WithBot ℝ) :=
  Ideal.sum fun i : Fin B => FPReference.exp Ideal.algebra
    (FPReference.sub Ideal.algebra (padded N B q i) (oneMax N B q))

theorem one_denominator (N B : Nat) (hB : 0<B) (covered : N≤B) (q : Nat → WithBot ℝ)
    (defined : Target.Defined (fun n : Fin N => q n.val)) :
    ∃ m : ℝ, oneMax N B q=(m : WithBot ℝ) ∧
      oneSum N B q=((Real.exp (-m)*Target.denominator (fun n : Fin N => q n.val) : ℝ) : WithBot ℝ) := by
  obtain ⟨n,hn⟩ := defined
  obtain ⟨m,hm⟩ := Ideal.maximum_finite (padded N B q)
    ⟨⟨n.val,lt_of_lt_of_le n.isLt covered⟩,by simpa [padded,n.isLt] using hn⟩
  refine ⟨m,hm,?_⟩
  have value (i : Fin B) : Target.mass (padded N B q i) =
      if i.val<N then Target.mass (q i.val) else 0 := by
    by_cases hi : i.val<N <;> simp [padded,hi]
  unfold oneSum
  change oneMax N B q=(m : WithBot ℝ) at hm
  simp_rw [hm,Ideal.exp_sub]
  rw [Ideal.sum_coe,←Finset.mul_sum]
  simp_rw [value]
  rw [sum_padding N B hB covered (fun n => Target.mass (q n))]
  rfl

theorem one_inner_value (s : State (WithBot ℝ)) (X : RegionName) (row N B : Nat)
    (hB : 0<B) (covered : N≤B)
    (defined : Target.Defined (fun n : Fin N => innerInput s X row N n.val))
    (i : TileIndex [B]) (hn : i.1.val<N) :
    FPReference.forwardInnerValue Ideal.algebra s X row N B i =
      ((Target.mass (innerInput s X row N i.1.val) /
        Target.denominator (fun n : Fin N => innerInput s X row N n.val) : ℝ) : WithBot ℝ) := by
  let q := innerInput s X row N
  have loadEq (j : TileIndex [B]) : FPReference.forwardInnerLoad Ideal.algebra s X row N B j = padded N B q j.1 := by
    simp [FPReference.forwardInnerLoad,FPReference.innerActive,FPReference.innerOffset,
      FPReference.cast,Ideal.algebra,padded,q,innerInput]
  have maxEq : Ideal.algebra.reduceMax (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
      (FPReference.forwardInnerLoad Ideal.algebra s X row N B) PUnit.unit = oneMax N B q := by
    change Ideal.maximum (fun j : Fin B => FPReference.forwardInnerLoad Ideal.algebra s X row N B (j,PUnit.unit)) = _
    simp_rw [loadEq]
    rfl
  have expEq (j : TileIndex [B]) : FPReference.forwardInnerExp Ideal.algebra s X row N B j =
      FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra (padded N B q j.1) (oneMax N B q)) := by
    simp only [FPReference.forwardInnerExp,loadEq,maxEq]
  have sumEq : Ideal.algebra.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
      (FPReference.forwardInnerExp Ideal.algebra s X row N B) PUnit.unit = oneSum N B q := by
    change Ideal.sum (fun j : Fin B => FPReference.forwardInnerExp Ideal.algebra s X row N B (j,PUnit.unit)) = _
    simp_rw [expEq]
    rfl
  obtain ⟨m,hm,hz⟩ := one_denominator N B hB covered q defined
  unfold FPReference.forwardInnerValue
  rw [expEq,sumEq,hm,hz]
  simpa only [padded,if_pos hn] using normalized_value (q i.1.val)
    (Target.denominator (fun n : Fin N => q n.val)) m

theorem one_noninner_value (s : State (WithBot ℝ)) (X : RegionName) (row col N K B C : Nat)
    (hB : 0<B) (covered : N≤B) (i : TileIndex [B,C]) (hn : i.1.val<N) (hk : col*C+i.2.1.val<K)
    (defined : Target.Defined (fun n : Fin N => noninnerInput s X row N K (col*C+i.2.1.val) n.val)) :
    FPReference.forwardNoninnerValue Ideal.algebra s X row col N K B C i =
      ((Target.mass (noninnerInput s X row N K (col*C+i.2.1.val) i.1.val) /
        Target.denominator (fun n : Fin N => noninnerInput s X row N K (col*C+i.2.1.val) n.val) : ℝ) : WithBot ℝ) := by
  rcases i with ⟨lane,k,u⟩
  cases u
  let q := noninnerInput s X row N K (col*C+k.val)
  have loadEq (j : Fin B) : FPReference.forwardNoninnerLoad Ideal.algebra s X row col N K B C (j,k,PUnit.unit) = padded N B q j := by
    by_cases hj : j.val<N <;>
      simp [FPReference.forwardNoninnerLoad,FPReference.noninnerActive,FPReference.noninnerOffset,
        FPReference.cast,Ideal.algebra,padded,q,noninnerInput,hk,hj]
  have maxEq : Ideal.algebra.reduceMax (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false
      (FPReference.forwardNoninnerLoad Ideal.algebra s X row col N K B C) (k,PUnit.unit) = oneMax N B q := by
    change Ideal.maximum (fun j : Fin B => FPReference.forwardNoninnerLoad Ideal.algebra s X row col N K B C (j,k,PUnit.unit)) = _
    simp_rw [loadEq]
    rfl
  have expEq (j : Fin B) : FPReference.forwardNoninnerExp Ideal.algebra s X row col N K B C (j,k,PUnit.unit) =
      FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra (padded N B q j) (oneMax N B q)) := by
    simp only [FPReference.forwardNoninnerExp,loadEq,maxEq]
  have sumEq : Ideal.algebra.reduceSum (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false
      (FPReference.forwardNoninnerExp Ideal.algebra s X row col N K B C) (k,PUnit.unit) = oneSum N B q := by
    change Ideal.sum (fun j : Fin B => FPReference.forwardNoninnerExp Ideal.algebra s X row col N K B C (j,k,PUnit.unit)) = _
    simp_rw [expEq]
    rfl
  obtain ⟨m,hm,hz⟩ := one_denominator N B hB covered q defined
  unfold FPReference.forwardNoninnerValue
  rw [expEq,sumEq,hm,hz]
  simpa only [padded,if_pos hn] using normalized_value (q lane.val)
    (Target.denominator (fun n : Fin N => q n.val)) m

#axiomsClean one_inner_value
#axiomsClean one_noninner_value
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
