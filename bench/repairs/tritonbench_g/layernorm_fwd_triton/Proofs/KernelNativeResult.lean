import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelGrid
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelNativeResult
open VeriTile.Triton FP.Structural Address
noncomputable section

/-- Complete native kernel grid over raw terms, preserving the selected local
instructions, profile, reduction trees and scalar operand. -/
theorem layernorm_fwd_triton_kernel_native (I : NativeKernel.Instructions)
    (P : NativeKernel.Profile) (R : NativeKernel.Reductions) (shape : Shape) (st : Strides)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val))
    (epsilon : NativeKernel.Term) (initial : State NativeKernel.Term) :
    ∃ final, Source.launch (NativeKernel.algebra R) shape st epsilon initial=some final ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns,
        NativeKernel.observeCell I P (final.mem "Y" (yOffset st row col.val))=.mk .fp32
          (NativeKernel.reference I P R shape.columns
            (KernelGrid.xValues initial st row) (KernelGrid.wValues initial st row) epsilon col.val)) ∧
      (∀ region off, KernelGrid.Outside st (Source.programs shape) region off →
        final.mem region off=initial.mem region off) := by
  obtain ⟨final,run,effect⟩ := KernelGrid.complete (NativeKernel.algebra R) injective epsilon initial
  refine ⟨final,run,?_,effect.frame⟩
  intro row col
  rw [effect.output row (KernelGrid.mem_programs row) col]
  rfl

#axiomsClean layernorm_fwd_triton_kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelNativeResult
