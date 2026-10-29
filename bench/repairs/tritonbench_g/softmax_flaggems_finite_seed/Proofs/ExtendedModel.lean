import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Target
import bench.tritonbench_g.softmax_flaggems.Proofs.RealAddress

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.ExtendedModel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems RealMath
open scoped BigOperators
noncomputable section

abbrev seed : ℝ := -340282346638528859811704183484516925440.0

def values (N B lane c : Nat) (q : Nat → WithBot ℝ) : List ℝ :=
  (List.range c).filterMap fun t => if t*B+lane<N then q (t*B+lane) else none

def state (N B lane c : Nat) (q : Nat → WithBot ℝ) : WithBot ℝ × ℝ :=
  (values N B lane c q).foldl onlineStep ((seed : WithBot ℝ),0)

def laneMass (N B lane c : Nat) (q : Nat → WithBot ℝ) : ℝ :=
  ∑ t ∈ Finset.range c, if t*B+lane<N then Target.mass (q (t*B+lane)) else 0

theorem values_mass (N B lane c : Nat) (q : Nat → WithBot ℝ) :
    ((values N B lane c q).map Real.exp).sum = laneMass N B lane c q := by
  induction c with
  | zero => simp [values,laneMass]
  | succ c ih =>
    simp only [values,List.range_succ,List.filterMap_append,List.filterMap_cons,
      List.filterMap_nil,List.map_append,List.sum_append,laneMass,Finset.sum_range_succ] at *
    rw [ih]
    by_cases h : c*B+lane<N
    · simp only [if_pos h]
      cases hq : q (c*B+lane) with
      | bot => simp
      | coe x => simp
    · simp [h]

theorem state_consistent (N B lane c : Nat) (q : Nat → WithBot ℝ) :
    (state N B lane c q).2 =
      (state N B lane c q).1.elim 0 (fun m => Real.exp (-m)) * laneMass N B lane c q := by
  have h := online_denominator_invariant (values N B lane c q) (seed : WithBot ℝ) 0 0
    (by simp) (by simp)
  simpa only [state,zero_add,values_mass] using h.1

theorem state_finite (N B lane c : Nat) (q : Nat → WithBot ℝ) :
    ∃ m : ℝ, (state N B lane c q).1 = (m : WithBot ℝ) := by
  have grow : ∀ (xs : List ℝ) (m : WithBot ℝ), m ≤ xs.foldl (fun (a : WithBot ℝ) (x : ℝ) => a ⊔ (x : WithBot ℝ)) m := by
    intro xs
    induction xs with
    | nil => intro m; exact le_rfl
    | cons x xs ih => intro m; exact le_trans le_sup_left (ih _)
  have ge := grow (values N B lane c q) (seed : WithBot ℝ)
  rw [←online_maximum (values N B lane c q) (seed : WithBot ℝ) 0] at ge
  change (seed : WithBot ℝ) ≤ (state N B lane c q).1 at ge
  cases hm : (state N B lane c q).1 with
  | bot => simp [hm] at ge
  | coe m => exact ⟨m,rfl⟩

@[simp] theorem state_zero (N B lane : Nat) (q : Nat → WithBot ℝ) :
    state N B lane 0 q = ((seed : WithBot ℝ),0) := by simp [state,values]

theorem state_succ (N B lane c : Nat) (q : Nat → WithBot ℝ) :
    state N B lane (c+1) q = maskedOnlineStep (state N B lane c q)
      (if c*B+lane<N then q (c*B+lane) else ⊥) := by
  obtain ⟨m,hm⟩ := state_finite N B lane c q
  have finite : (state N B lane c q).1=⊥ → (state N B lane c q).2=0 := by simp [hm]
  have step_none : maskedOnlineStep (state N B lane c q) ⊥ = state N B lane c q :=
    maskedOnlineStep_none _ _ finite
  have step : state N B lane (c+1) q =
      match (if c*B+lane<N then q (c*B+lane) else none) with
      | none => state N B lane c q
      | some x => onlineStep (state N B lane c q) x := by
    unfold state values
    rw [List.range_succ,List.filterMap_append,List.foldl_append]
    by_cases h : c*B+lane<N
    · cases hq : q (c*B+lane) <;> simp [List.filterMap_cons,List.filterMap_nil,List.foldl_cons,List.foldl_nil,h,hq]
    · simp [h]
  rw [step]
  by_cases h : c*B+lane<N
  · simp only [if_pos h]
    cases hq : q (c*B+lane) with
    | bot => exact step_none.symm
    | coe x => exact (maskedOnlineStep_some _ _ _).symm
  · simpa only [if_neg h] using step_none.symm

theorem mass_partition (N B c : Nat) (hB : 0<B) (covered : N ≤ c*B)
    (q : Nat → WithBot ℝ) :
    ∑ lane : Fin B, laneMass N B lane.val c q =
      Target.denominator (fun n : Fin N => q n.val) := by
  unfold laneMass Target.denominator
  rw [lane_tile_partition N B hB c covered (fun n => Target.mass (q n))]
  exact (Fin.sum_univ_eq_sum_range (fun n => Target.mass (q n)) N).symm

#axiomsClean state_consistent
#axiomsClean state_finite
#axiomsClean state_succ
#axiomsClean mass_partition
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.ExtendedModel
