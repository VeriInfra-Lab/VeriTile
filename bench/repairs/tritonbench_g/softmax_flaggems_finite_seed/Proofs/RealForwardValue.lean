import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealForwardLoop

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 5000000

theorem normalized_value (x : WithBot ℝ) (z m : ℝ) :
    FPReference.cast Ideal.algebra (FPReference.div Ideal.algebra
      (FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra x (m : WithBot ℝ)))
      ((Real.exp (-m)*z : ℝ) : WithBot ℝ)) = ((Target.mass x/z : ℝ) : WithBot ℝ) := by
  rw [Ideal.exp_sub,Ideal.div_coe,Ideal.cast_value]
  congr 1
  exact mul_div_mul_left _ _ (Real.exp_ne_zero _)

theorem inner_reduced (s : State (WithBot ℝ)) (X : RegionName) (row N B : Nat) (hB : 0<B) :
    ∃ m : ℝ,
      FPForwardInner.maxReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B)) = (m : WithBot ℝ) ∧
      FPForwardInner.sumReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B)) =
        ((Real.exp (-m)*Target.denominator (fun n : Fin N => innerInput s X row N n.val) : ℝ) : WithBot ℝ) :=
  collapse N B hB (innerInput s X row N)

theorem noninner_reduced (s : State (WithBot ℝ)) (X : RegionName)
    (row col N K B C : Nat) (hB : 0<B) (k : Fin C) :
    ∃ m : ℝ,
      FPForwardNoninner.maxReduced Ideal.algebra B C
        (noninnerModel s X row col N K B C (RealAddress.rounds N B)) (k,PUnit.unit) = (m : WithBot ℝ) ∧
      FPForwardNoninner.sumReduced Ideal.algebra B C
        (noninnerModel s X row col N K B C (RealAddress.rounds N B)) (k,PUnit.unit) =
        ((Real.exp (-m)*Target.denominator (fun n : Fin N => noninnerInput s X row N K (col*C+k.val) n.val) : ℝ) : WithBot ℝ) :=
  collapse N B hB (noninnerInput s X row N K (col*C+k.val))

theorem online_inner_value (s t : State (WithBot ℝ)) (X : RegionName)
    (same : ∀ o, FPReference.read t X o=FPReference.read s X o)
    (masked : Bool) (row N B cur : Nat) (hB : 0<B) (i : TileIndex [B])
    (hn : FPForwardInner.previous N B-cur+i.1.val<N) :
    FPForwardInner.storedValue Ideal.algebra t X masked row N B cur
      (FPForwardInner.maxReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B)))
      (FPForwardInner.sumReduced Ideal.algebra B (innerModel s X row N B (RealAddress.rounds N B))) i =
    ((Target.mass (innerInput s X row N (FPForwardInner.previous N B-cur+i.1.val)) /
      Target.denominator (fun n : Fin N => innerInput s X row N n.val) : ℝ) : WithBot ℝ) := by
  obtain ⟨m,hm,hz⟩ := inner_reduced s X row N B hB
  have loaded : FPForwardInner.load Ideal.algebra t X masked row N B (FPForwardInner.previous N B-cur) i =
      innerInput s X row N (FPForwardInner.previous N B-cur+i.1.val) := by
    simp [FPForwardInner.load,FPForwardInner.active,FPForwardInner.offset,Ideal.algebra,hn,innerInput,same]
  unfold FPForwardInner.storedValue
  rw [loaded,hm,hz]
  exact normalized_value _ _ _

theorem online_noninner_value (s t : State (WithBot ℝ)) (X : RegionName)
    (same : ∀ o, FPReference.read t X o=FPReference.read s X o)
    (row col N K B C cur : Nat) (hB : 0<B) (i : TileIndex [B,C])
    (hn : FPForwardNoninner.previous N B-cur+i.1.val<N) (hk : col*C+i.2.1.val<K) :
    FPForwardNoninner.storedValue Ideal.algebra t X row col N K B C cur
      (FPForwardNoninner.maxReduced Ideal.algebra B C (noninnerModel s X row col N K B C (RealAddress.rounds N B)))
      (FPForwardNoninner.sumReduced Ideal.algebra B C (noninnerModel s X row col N K B C (RealAddress.rounds N B))) i =
    ((Target.mass (noninnerInput s X row N K (col*C+i.2.1.val) (FPForwardNoninner.previous N B-cur+i.1.val)) /
      Target.denominator (fun n : Fin N => noninnerInput s X row N K (col*C+i.2.1.val) n.val) : ℝ) : WithBot ℝ) := by
  obtain ⟨m,hm,hz⟩ := noninner_reduced s X row col N K B C hB i.2.1
  have loaded : FPForwardNoninner.load Ideal.algebra t X row col N K B C (FPForwardNoninner.previous N B-cur) i =
      noninnerInput s X row N K (col*C+i.2.1.val) (FPForwardNoninner.previous N B-cur+i.1.val) := by
    simp [FPForwardNoninner.load,FPForwardNoninner.active,FPForwardNoninner.offset,Ideal.algebra,hn,hk,noninnerInput,same]
  unfold FPForwardNoninner.storedValue
  have eta : i.2=(i.2.1,PUnit.unit) := by rcases i with ⟨a,b,u⟩; cases u; rfl
  rw [eta,loaded,hm,hz]
  exact normalized_value _ _ _

#axiomsClean online_inner_value
#axiomsClean online_noninner_value
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
