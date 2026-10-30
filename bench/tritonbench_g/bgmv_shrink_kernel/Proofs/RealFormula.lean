import bench.tritonbench_g.bgmv_shrink_kernel.Mathematical
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.CanonicalTrace
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.Real

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RealFormula
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open scoped BigOperators
noncomputable section

/-- Compatibility witness for the existing Real theorem; not a public input. -/
def block (h : Call) (indices : Nat → Int) (vals : ChannelValues ℝ (IO.inputs h indices)) : BlockState where
  mem := fun r i => .real (if r="X" then IOValues.input h indices vals i else
    if r="W" then IOValues.weights h indices vals i else
    if r="Y" then IOValues.output h indices vals i else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0

theorem data_same (h : Call) (indices : Nat → Int) (vals : ChannelValues ℝ (IO.inputs h indices)) :
    Real.data h (block h indices vals) indices = IOValues.data h indices h.scaling vals := by
  rfl

theorem real_formula (h : Call) (indices : Nat → Int) (orders : Orders h)
    (vals : ChannelValues ℝ (IO.inputs h indices)) (b : Fin h.batches) (n : Fin h.N) :
    ofFloat h.outputType (FPValues.target Real.algebra Real.primitives h orders
      (IOValues.data h indices h.scaling vals) b n) = Mathematical.real h indices vals 0 (b,n,PUnit.unit) := by
  have result := Real.readback h orders (block h indices vals) indices
    (CanonicalTrace.run Real.algebra Real.primitives h orders (Real.data h (block h indices vals) indices)) b n
  change FPValues.target Real.algebra Real.primitives h orders
    (Real.data h (block h indices vals) indices) b n = Target.real h (block h indices vals) indices b n at result
  rw [data_same] at result
  rw [result]
  change ofFloat h.outputType (if indices b.val = -1 then IOValues.output h indices vals (outputOffset h b.val n.val) else
    IOValues.output h indices vals (outputOffset h b.val n.val)+h.scaling*∑ k : Fin h.K,
      IOValues.input h indices vals (inputOffset h b.val k.val)*
        IOValues.weights h indices vals (weightOffset h (indices b.val).toNat n.val k.val)) = _
  have summation : (∑ k : Fin h.K, IOValues.input h indices vals (inputOffset h b.val k.val)*
      IOValues.weights h indices vals (weightOffset h (indices b.val).toNat n.val k.val)) =
      ∑ k : Fin h.K, toFloat (dtype h.storage) (vals 1 (b,k,PUnit.unit))*
        toFloat (dtype h.storage) (vals 2 (⟨(indices b.val).toNat,IO.selected_inside h indices b⟩,n,k,PUnit.unit)) := by
    apply Finset.sum_congr rfl
    intro k _
    rw [IOValues.input_value,IOValues.weight_value h indices vals
      ⟨(indices b.val).toNat,IO.selected_inside h indices b⟩ n k]
  rw [IOValues.output_value,summation]
  rfl

#axiomsClean real_formula
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RealFormula
