import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IOReal

namespace VeriTile.Bench.Repairs.FastLayernorm.LogicalIO
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem forward_iff (N B pid : Nat) (eps : ℝ) (bias : Bool) (f : ForwardInputs B → ForwardOutputs B) :
    (forwardView KernelMath.algebra N B pid eps bias ⊨ f) ↔
      (Interface.forwardIO KernelMath.algebra N B pid eps bias).Implements (fun raw =>
        let y := f ⟨Interface.forwardX raw,Interface.forwardW raw,Interface.forwardB raw⟩
        Interface.forwardResult y.values y.inverse y.mean) := Iff.rfl

theorem backward_iff (N B pid : Nat) (f : BackwardInputs B → Fin B → ℝ) :
    (backwardView KernelMath.algebra N B pid ⊨ f) ↔
      (Interface.backwardIO KernelMath.algebra N B pid).Implements (fun raw =>
        Interface.backwardResult (f ⟨Interface.backwardX raw,Interface.backwardDY raw,Interface.backwardW raw,
          Interface.backwardMean raw,Interface.backwardInverse raw⟩)) := Iff.rfl

theorem affine_iff (rows N wc bc pid : Nat) (bias : Bool) (f : AffineInputs rows → AffineOutputs) :
    (affineView KernelMath.algebra rows N wc bc pid bias ⊨ f) ↔
      (Interface.affineIO KernelMath.algebra rows N wc bc pid bias).Implements (fun raw =>
        let y := f ⟨Interface.affineX raw,Interface.affineDY raw,Interface.affineMean raw,Interface.affineInverse raw⟩
        Interface.affineResult y.weight y.bias) := Iff.rfl
end
end VeriTile.Bench.Repairs.FastLayernorm.LogicalIO
