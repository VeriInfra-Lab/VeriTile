import bench.repairs.tritonbench_g.fast_layernorm.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FastLayernorm.ReaderConnections
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO BigOperators
noncomputable section

theorem forward (N B pid : Nat) (covered : N ≤ B) (eps : ℝ) (bias : Bool)
    (F : LogicalIO.ForwardInputs B → LogicalIO.ForwardOutputs B)
    (definition : F = fun v =>
      let x := LogicalIO.entry (v.x)
      let mu := (∑ j ∈ Finset.range N, x j)/(N : ℝ)
      let r := 1/Real.sqrt ((∑ j ∈ Finset.range N, (x j-mu)^2)/(N : ℝ)+eps)
      LogicalIO.ForwardOutputs.mk (fun j => ((x j.val-mu)*r)*v.weight j +
      (if bias then v.bias j else 0)) r mu) :
    LogicalIO.forwardView KernelMath.algebra N B pid eps bias ⊨ F := by
  apply (LogicalIO.forward_iff N B pid eps bias _).mpr
  have formula : Interface.forwardMath N B eps bias = (fun raw =>
      let y := F ⟨Interface.forwardX raw,Interface.forwardW raw,Interface.forwardB raw⟩
      Interface.forwardResult y.values y.inverse y.mean) := by
    funext raw port j
    fin_cases port <;>
      simp [Interface.forwardMath, definition, RealProof.outputSpec,
        RealProof.invSpec, RealProof.varianceSpec, RealProof.meanSpec,
        LogicalIO.entry,Interface.forwardResult,Interface.extend_fin]
  exact formula ▸ Interface.forward_real N B pid covered eps bias

end
end VeriTile.Bench.Repairs.FastLayernorm.ReaderConnections
