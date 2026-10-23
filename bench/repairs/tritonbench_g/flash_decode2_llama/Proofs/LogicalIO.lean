import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.RealIO

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.LogicalIO
open VeriTile.Triton FP.Structural FlashDecode2
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def weighted (g : KernelInterface.Geometry) (xs : Inputs g) (row : Fin (g.B*g.H)) (d : Fin g.D) : ℝ :=
  (∑ j ∈ Finset.range ((xs.length row+g.Q-1)/g.Q),Real.exp (xs.log row j)*xs.mid row j d) /
    (∑ j ∈ Finset.range ((xs.length row+g.Q-1)/g.Q),Real.exp (xs.log row j))

/-- All old source safety, observations and frame obligations are retained. -/
theorem judgment_iff (g : KernelInterface.Geometry) (order : List Nat) :
    (view g order RealIO.observe ⊨{Lengths g} weighted g) ↔
      (IO.decodeIO Reference.ideal g order some RealIO.observe).ImplementsOn
        (IO.Lengths g) (IO.mathematical g) := by
  have domain : (fun raw => Lengths g (inputEquiv g raw)) = IO.Lengths g := by
    funext raw; apply propext
    constructor
    · intro h row hr; exact h ⟨row,hr⟩
    · intro h row; exact h row.val row.isLt
  have formula : (view g order RealIO.observe).lift (weighted g) = IO.mathematical g := by
    funext raw port j
    fin_cases port
    rcases j with ⟨row,d,u⟩
    cases u
    change weighted g (inputEquiv g raw) row d = Reference.target
      (IO.logValues raw row.val) (IO.midValues raw row.val) ((IO.lengths raw (row.val/g.H)+g.Q-1)/g.Q) d.val
    simp [weighted,Inputs.log,Inputs.mid,inputEquiv,Inputs.length,
      Reference.target,Reference.numerator,Reference.denominator,
      IO.logValues,IO.midValues,IO.lengths,row.isLt,d.isLt]
    all_goals rfl
  change (view g order RealIO.observe).core.ImplementsOn _ _ ↔ _
  rw [formula]
  change (IO.decodeIO Reference.ideal g order some RealIO.observe).ImplementsOn
    (fun raw => Lengths g (inputEquiv g raw)) _ ↔ _
  rw [domain]
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.LogicalIO
