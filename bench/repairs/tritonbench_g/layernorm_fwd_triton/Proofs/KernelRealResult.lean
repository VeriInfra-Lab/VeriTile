import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelGrid
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelRealResult
open VeriTile.Triton FP.Structural Address
noncomputable section

/-- Complete Real kernel grid over raw typed input memory and output geometry. -/
theorem layernorm_fwd_triton_kernel_real (shape : Shape) (st : Strides)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val))
    (epsilon : ℝ) (initial : State ℝ) :
    ∃ final, Source.launch Mathematics.ideal shape st epsilon initial=some final ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns,
        final.mem "Y" (yOffset st row col.val)=.mk .fp32
          (Contract.Math.output shape.columns (KernelGrid.xValues initial st row)
            (KernelGrid.wValues initial st row) epsilon col.val)) ∧
      (∀ region off, KernelGrid.Outside st (Source.programs shape) region off →
        final.mem region off=initial.mem region off) := by
  obtain ⟨final,run,effect⟩ := KernelGrid.complete Mathematics.ideal injective epsilon initial
  refine ⟨final,run,?_,effect.frame⟩
  intro row col
  rw [effect.output row (KernelGrid.mem_programs row) col,
    Mathematics.output_real _ _ _ _ _ col.isLt]

#axiomsClean layernorm_fwd_triton_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelRealResult
