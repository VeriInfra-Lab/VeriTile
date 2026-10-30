import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.RuntimeScalar

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Native
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

theorem scalar_ops (plan : ReductionPlan) (binding : ScalarBinding) :
    Source.ops (scalarAlgebra plan binding)=Source.ops (algebra plan) := by
  simp [Source.ops,scalarAlgebra,algebra]

/-- The actual integer call and its source conversion yield the same designated
ABI operand used by the independent formula. Integer 1 uses its literal case. -/
theorem scalar_value (plan : ReductionPlan) (binding : ScalarBinding) (raw : PythonNumber) :
    (Runtime.nativeScalar binding raw).value (scalarAlgebra plan binding)=scalar binding raw := by
  cases raw with
  | «float» bits => rfl
  | integer value =>
    simp only [Runtime.nativeScalar,Runtime.classify]
    split_ifs with one small signed
    · subst value
      simp [Runtime.Scalar.value,scalarAlgebra]
    · rfl
    · rfl
    · have hn : 0≤value := by omega
      simp [Runtime.Scalar.value,scalarAlgebra,Int.toNat_of_nonneg hn]

theorem scalar_rows (plan : ReductionPlan) (binding : ScalarBinding) (c : Shape)
    (s : State Term) (scale : Term) (n : Nat) (i : TileIndex [c.BV]) :
    Source.rowValues (scalarAlgebra plan binding) c s scale n i=
      (Source.ops (scalarAlgebra plan binding)).storeCast
        (scalarPartial (Source.ops (scalarAlgebra plan binding)) c (plans c plan)
          (FusedRwkv6Kernel.inputs c (Source.memory s)) scale (Source.program s).z (Source.program s).ik n
          (valueIndex c (Source.program s) i.1.val)) := by
  have row : Source.rowValues (scalarAlgebra plan binding) c s scale n i=
      Source.rowValues (algebra plan) c s scale n i := by
    unfold Source.rowValues Source.terms
    simp only [scalar_ops]
    rfl
  rw [row,scalar_ops]
  exact Source.native_rows plan c s scale n i

#axiomsClean scalar_ops
#axiomsClean scalar_value
#axiomsClean scalar_rows
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Native
