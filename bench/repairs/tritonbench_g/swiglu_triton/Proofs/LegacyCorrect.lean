import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectHost

namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton
open Memory DirectPhysical DirectHost
noncomputable section

/-- Complete checked forward/backward host paths over the direct contiguous
Tensor spans, with original source interpretation, returned views and frame. -/
theorem swiglu_direct_correct {M β : Type} (ops : Ops M ℝ β)
    (rt : Runtime backward) (entry : M) :
    VeriTile.Spec.Real (HostSpec ops Math.formula
      (fun p h plan s => SourceSteps ops Math.algebra backward (layout p h) plan.order s)
      rt entry (execute ops (DirectTile.scalar Math.algebra) rt entry)) := by
  apply host_spec
  · intro ch a b d
    exact Math.scalar_real ch a b d
  · intro p h plan s
    apply source_steps ops Math.algebra backward (layout p h) (valid p h) plan.order
    intro r hr
    exact List.mem_range.mp (plan.schedule.mem_iff.mp hr)

#axiomsClean swiglu_direct_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton
