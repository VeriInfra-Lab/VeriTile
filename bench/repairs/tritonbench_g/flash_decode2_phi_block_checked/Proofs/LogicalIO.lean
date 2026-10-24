import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.RealIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.LogicalIO
open VeriTile.Triton FP.Structural FlashDecode2
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def weighted (g : PhiIO.Geometry) (xs : Inputs g) (row : Fin (g.B*g.H)) (d : Fin g.D) : ℝ :=
  (∑ j ∈ Finset.range (xs.steps row),Real.exp (xs.log row j)*xs.mid row j d) /
    (∑ j ∈ Finset.range (xs.steps row),Real.exp (xs.log row j))

/-- All old source safety, observations and frame obligations are retained. -/
theorem judgment_iff (g : PhiIO.Geometry) (order : List Nat) :
    (view g order RealIO.observe ⊨{(fun xs => Fits g xs ∧ Positive g xs)} weighted g) ↔
      (PhiIO.decodeIO Reference.ideal g order some RealIO.observe).ImplementsOn
        (fun xs => PhiIO.Fits g xs ∧ PhiIO.Positive g xs) (PhiIO.mathematical g) := by
  have fits : (fun raw => Fits g (inputEquiv g raw)) = PhiIO.Fits g := by
    funext raw; apply propext
    constructor
    · intro h row hr; exact h ⟨row,hr⟩
    · intro h row; exact h row.val row.isLt
  have positive : (fun raw => Positive g (inputEquiv g raw)) = PhiIO.Positive g := by
    funext raw; apply propext
    constructor
    · intro h row hr; exact h ⟨row,hr⟩
    · intro h row; exact h row.val row.isLt
  have domain : (fun raw => Fits g (inputEquiv g raw) ∧ Positive g (inputEquiv g raw)) =
      (fun raw => PhiIO.Fits g raw ∧ PhiIO.Positive g raw) := by rw [← fits,← positive]
  have formula : (view g order RealIO.observe).lift (weighted g) = PhiIO.mathematical g := by
    funext raw port j
    fin_cases port
    rcases j with ⟨row,d,u⟩
    cases u
    change weighted g (inputEquiv g raw) row d = Reference.target
      (PhiIO.logs raw row.val) (PhiIO.mids raw row.val) (PhiIO.steps raw row.val) d.val
    simp [weighted,Inputs.log,Inputs.mid,inputEquiv,Inputs.length,
      Inputs.steps,Reference.target,Reference.numerator,Reference.denominator,
      PhiIO.logs,PhiIO.mids,PhiIO.steps,PhiIO.lengths,row.isLt,d.isLt]
    all_goals rfl
  change (view g order RealIO.observe).core.ImplementsOn _ _ ↔ _
  rw [formula]
  change (PhiIO.decodeIO Reference.ideal g order some RealIO.observe).ImplementsOn
    (fun raw => Fits g (inputEquiv g raw) ∧ Positive g (inputEquiv g raw)) _ ↔ _
  rw [domain]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.LogicalIO
