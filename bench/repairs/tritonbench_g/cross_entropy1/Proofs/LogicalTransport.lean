import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.CrossEntropy1.LogicalProof
open VeriTile.Triton FP.Structural

/-- General bijective restatement of the entire existing typed IO judgment. -/
theorem reindex_iff {α β I O : Type} [Inhabited β] (io : LogicalProgramIO α β I O)
    (domain : io.core.Inputs → Prop) (f : io.core.Inputs → io.core.Outputs) :
    io.ImplementsOn (fun xs => domain (io.inputs.symm xs))
      (fun xs => io.outputs (f (io.inputs.symm xs))) ↔
        io.core.ImplementsOn domain f := by
  have hd : (fun raw => domain (io.inputs.symm (io.inputs raw))) = domain := by
    funext raw
    rw [Equiv.symm_apply_apply]
  have hf : (fun raw => io.outputs.symm (io.outputs (f (io.inputs.symm (io.inputs raw))))) = f := by
    funext raw
    rw [Equiv.symm_apply_apply, Equiv.symm_apply_apply]
  exact Iff.of_eq (congrArg₂ (fun d g => io.core.ImplementsOn d g) hd hf)

end VeriTile.Bench.Repairs.CrossEntropy1.LogicalProof
