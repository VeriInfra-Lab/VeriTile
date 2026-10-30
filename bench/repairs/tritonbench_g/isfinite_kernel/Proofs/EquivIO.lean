import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.IO

namespace VeriTile.Bench.Repairs.IsfiniteKernel.IOProof
open VeriTile.Triton FP.Structural
noncomputable section

/-- The output frame does not require injective addresses: it only excludes
actual active stores. Both candidates use exactly those same addresses. -/
theorem stores_frame {A : Type} [Inhabited A] (classify : A → Bool)
    (input output : View) (n B pid : Nat) (snapshot current : State A)
    (r : RegionName) (address : Nat)
    (outside : r≠"y" ∨ ∀ j : Fin B,pid*B+j.val<n → address≠output.address (pid*B+j.val)) :
    (Source.stores classify input output n B pid snapshot current).mem r address=current.mem r address := by
  apply State.masked_scatter_frame
  exact outside.elim Or.inl (fun h => Or.inr (fun j _ => h j.1))

theorem candidate_safe {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat)
    (xs : (classificationIO precision classify input output n B pid).Inputs)
    (bounds : RegionName → Nat)
    (bounded : (classificationIO precision classify input output n B pid).InBounds bounds xs)
    (initial : State A) :
    ScheduleSafe bounds ((indexCandidate precision classify input output n B pid).schedule xs) initial := by
  simp only [indexCandidate,scheduleSafe_singleton]
  apply Source.candidate_access_safe
  · intro j hj
    exact bounded.2.1 (0 : Fin 1) (j,PUnit.unit) hj
  · intro j hj
    exact bounded.2.2.1 (0 : Fin 1) (j,PUnit.unit) hj

/-- Integer-index commutation preserves the two complete source executions;
no classifier law, finite-input restriction, or output injectivity is needed. -/
theorem equivalent {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B pid : Nat) :
    (classificationIO precision classify input output n B pid).Equivalent
      (indexCandidate precision classify input output n B pid) := by
  intro xs _ bounds bounded initial inputs
  refine ⟨safe precision classify input output n B pid xs bounds bounded initial,
    candidate_safe precision classify input output n B pid xs bounds bounded initial,?_⟩
  let positioned : State A := {initial with pids := fun ax => if ax=0 then pid else 0}
  obtain ⟨left,leftRun,leftMemory⟩ := Source.executes precision classify input output n B positioned
  obtain ⟨right,rightRun,rightMemory⟩ := Source.commuted_executes precision classify input output n B positioned
  have frame : ∀ t : State A,t.mem=(Source.stores classify input output n B pid positioned positioned).mem →
      (classificationIO precision classify input output n B pid).Frame xs initial t := by
    intro t memory r address outside
    rw [memory]
    apply stores_frame
    by_cases hr : r="y"
    · right
      intro j hj
      exact (outside.1 (0 : Fin 1) (j,PUnit.unit) hj).resolve_left (not_not.mpr hr)
    · exact Or.inl hr
  refine ⟨left,right,?_,?_,?_,frame left leftMemory,frame right rightMemory⟩
  · simpa only [classificationIO,runSchedule_singleton,Invocation.run,Invocation.initial,
      bindRegisters,List.foldl,Option.getD,Bool.false_eq_true,↓reduceIte] using leftRun
  · simpa only [indexCandidate,runSchedule_singleton,Invocation.run,Invocation.initial,
      bindRegisters,List.foldl,Option.getD,Bool.false_eq_true,↓reduceIte] using rightRun
  · intro o j hj
    change mapCell id (left.mem _ _) = mapCell id (right.mem _ _)
    rw [leftMemory,rightMemory]

end
end VeriTile.Bench.Repairs.IsfiniteKernel.IOProof
