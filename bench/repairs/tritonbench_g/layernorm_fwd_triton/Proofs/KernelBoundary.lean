import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelGrid
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelBoundary
open VeriTile.Triton FP.Structural Address InputBinding
noncomputable section
variable {α : Type} [Inhabited α] {shape : Shape} {st : Strides} {eps : Contract.Epsilon}

theorem original_values (a : Arguments shape st eps) (read : RegionName → Nat → α)
    (scratch : State α) (row : Row shape) (col : Fin shape.columns) :
    KernelGrid.xValues (state read a scratch) st row col.val=logicalX read a row col.val ∧
    KernelGrid.wValues (state read a scratch) st row col.val=logicalW read a row col.val := by
  have hx := (x_pointer a (fun _ => FP.Value.zero .fp32) row col).1
  have hw := (w_pointer a (fun _ => FP.Value.zero .fp32) row col).1
  constructor
  · simp [KernelGrid.xValues,state,logicalX,hx]
  · simp [KernelGrid.wValues,state,logicalW,hw]

/-- Complete canonical DSL success from the original Tensor pointer reads.
All live outputs use original logical values, and every other regional cell
retains both its original payload and its dtype tag. -/
theorem complete (M : Algebra α) (a : Arguments shape st eps)
    (read : RegionName → Nat → α) (scratch : State α) (epsilon : α) :
    ∃ final, Source.launch M shape st epsilon (state read a scratch)=some final ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns,
        final.mem "Y" (yOffset st row col.val)=.mk .fp32
          (Graph.output M shape.columns (logicalX read a row) (logicalW read a row) epsilon col.val)) ∧
      (∀ region off, KernelGrid.Outside st (Source.programs shape) region off →
        final.mem region off=(state read a scratch).mem region off) := by
  obtain ⟨final,run,effect⟩ := KernelGrid.complete M a.outputInjective epsilon (state read a scratch)
  refine ⟨final,run,?_,effect.frame⟩
  intro row col
  rw [effect.output row (KernelGrid.mem_programs row) col]
  congr 1
  exact Graph.output_congr M shape.columns _ _ _ _ epsilon
    (fun j hj => (original_values a read scratch row ⟨j,hj⟩).1)
    (fun j hj => (original_values a read scratch row ⟨j,hj⟩).2) col.val col.isLt

#axiomsClean original_values
#axiomsClean complete
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelBoundary
