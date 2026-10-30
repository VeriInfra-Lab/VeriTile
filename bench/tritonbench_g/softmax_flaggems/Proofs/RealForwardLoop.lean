import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardStep

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems RealMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 5000000

@[simp] theorem inner_seed (s : State (WithBot ℝ)) (X : RegionName) (row N B : Nat) :
    innerModel s X row N B 0 =
      ((fun _ => FPReference.literal Ideal.algebra (-340282346638528859811704183484516925440.0)),
       fun _ => FPReference.literal Ideal.algebra 0) := by
  ext i <;> simp [innerModel,ExtendedModel.state_zero]

@[simp] theorem noninner_seed (s : State (WithBot ℝ)) (X : RegionName) (row col N K B C : Nat) :
    noninnerModel s X row col N K B C 0 =
      ((fun _ => FPReference.literal Ideal.algebra (-340282346638528859811704183484516925440.0)),
       fun _ => FPReference.literal Ideal.algebra 0) := by
  ext i <;> simp [noninnerModel,ExtendedModel.state_zero]

theorem inner_walk (s : State (WithBot ℝ)) (X : RegionName) (row N B : Nat)
    (hN : 0<N) (hB : 0<B) :
    let first := (FPLoop.walk (FPForwardInner.accumulate Ideal.algebra s X Bool.false row N B)
      (FPForwardInner.previous N B) B 0 (innerModel s X row N B 0)).2
    (FPLoop.walk (FPForwardInner.accumulate Ideal.algebra s X Bool.true row N B)
      N B (FPForwardInner.previous N B) first).2 = innerModel s X row N B (RealAddress.rounds N B) := by
  let r := RealAddress.rounds N B
  have hp : FPForwardInner.previous N B=(r-1)*B := RealAddress.previous_eq N B
  have hc : RealAddress.rounds (FPForwardInner.previous N B) B=r-1 := by rw [hp]; exact MathLoop.rounds_mul _ B hB
  have hprev : FPForwardInner.previous N B<N := RealAddress.previous_lt hN hB
  have first := MathLoop.walk_exact
    (FPForwardInner.accumulate Ideal.algebra s X Bool.false row N B)
    (innerModel s X row N B) (FPForwardInner.previous N B) B 0 hB (Nat.zero_le _)
    (by
      intro c bound
      apply inner_step
      intro _
      have cb : c+1≤r-1 := by omega
      have le := Nat.mul_le_mul_right B cb
      rw [←hp] at le
      omega)
  simp only [Nat.zero_mul,hc] at first
  dsimp only
  rw [first,hp]
  exact MathLoop.walk_exact
    (FPForwardInner.accumulate Ideal.algebra s X Bool.true row N B)
    (innerModel s X row N B) N B (r-1) hB (Nat.sub_le _ _)
    (fun c _ => inner_step s X Bool.true row N B c (by simp))

theorem noninner_walk (s : State (WithBot ℝ)) (X : RegionName) (row col N K B C : Nat)
    (hB : 0<B) :
    (FPLoop.walk (FPForwardNoninner.accumulate Ideal.algebra s X row col N K B C)
      N B 0 (noninnerModel s X row col N K B C 0)).2 =
      noninnerModel s X row col N K B C (RealAddress.rounds N B) := by
  simpa only [Nat.zero_mul] using MathLoop.walk_exact
    (FPForwardNoninner.accumulate Ideal.algebra s X row col N K B C)
    (noninnerModel s X row col N K B C) N B 0 hB (Nat.zero_le _)
    (fun c _ => noninner_step s X row col N K B C c)

theorem exp_sub_unpack (a b : WithBot ℝ) :
    FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra a b) =
      (((WithBot.realExp (WithBot.realSub a b)).unbotD 0 : ℝ) : WithBot ℝ) := by
  cases a <;> cases b <;> rfl

def modelMax (N B : Nat) (q : Nat → WithBot ℝ) : WithBot ℝ :=
  Ideal.maximum fun i : Fin B => (ExtendedModel.state N B i.val (RealAddress.rounds N B) q).1

def modelSum (N B : Nat) (q : Nat → WithBot ℝ) : WithBot ℝ :=
  Ideal.sum fun i : Fin B =>
    FPReference.mul Ideal.algebra
      ((ExtendedModel.state N B i.val (RealAddress.rounds N B) q).2 : WithBot ℝ)
      (FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra
        (ExtendedModel.state N B i.val (RealAddress.rounds N B) q).1 (modelMax N B q)))

theorem collapse (N B : Nat) (hB : 0<B) (q : Nat → WithBot ℝ) :
    ∃ m : ℝ, modelMax N B q=(m : WithBot ℝ) ∧
      modelSum N B q=((Real.exp (-m)*Target.denominator (fun n : Fin N => q n.val) : ℝ) : WithBot ℝ) := by
  obtain ⟨m,hm⟩ := Ideal.maximum_finite
    (fun i : Fin B => (ExtendedModel.state N B i.val (RealAddress.rounds N B) q).1)
    (by
      refine ⟨⟨0,hB⟩,?_⟩
      obtain ⟨a,ha⟩ := ExtendedModel.state_finite N B 0 (RealAddress.rounds N B) q
      simp [ha])
  refine ⟨m,hm,?_⟩
  have point (i : Fin B) :
      FPReference.mul Ideal.algebra
        ((ExtendedModel.state N B i.val (RealAddress.rounds N B) q).2 : WithBot ℝ)
        (FPReference.exp Ideal.algebra (FPReference.sub Ideal.algebra
          (ExtendedModel.state N B i.val (RealAddress.rounds N B) q).1 (m : WithBot ℝ))) =
      ((Real.exp (-m)*ExtendedModel.laneMass N B i.val (RealAddress.rounds N B) q : ℝ) : WithBot ℝ) := by
    rw [exp_sub_unpack,Ideal.mul_coe]
    congr 1
    apply online_lane_rescale
    · exact ExtendedModel.state_consistent N B i.val (RealAddress.rounds N B) q
    · obtain ⟨a,ha⟩ := ExtendedModel.state_finite N B i.val (RealAddress.rounds N B) q
      simp [ha]
  unfold modelSum
  change Ideal.sum _ = _
  change modelMax N B q=(m : WithBot ℝ) at hm
  simp_rw [hm,point]
  rw [Ideal.sum_coe,←Finset.mul_sum,
    ExtendedModel.mass_partition N B (RealAddress.rounds N B) hB (RealAddress.coverage hB)]

#axiomsClean inner_walk
#axiomsClean noninner_walk
#axiomsClean collapse
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
