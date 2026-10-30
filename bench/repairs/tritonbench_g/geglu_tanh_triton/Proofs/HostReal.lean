import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.DirectHost

namespace VeriTile.Bench.Repairs.GegluTanhTriton
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical DirectHost
noncomputable section

/-- Every flat output, including columns beyond 128, and both original
return-view branches. Real values remain independent of FP32 byte storage. -/
specification geglu_direct_correct {M β : Type} (ops : Ops M ℝ β)
    (rt : Runtime backward) (entry : M) :
    VeriTile.Spec.Real (HostSpec ops RealExecution.formula
      (fun p h plan s => Source.Certificates ops RealExecution.algebra backward (layout p h) plan.order s)
      rt entry (execute ops (Tile.scalar RealExecution.algebra) rt entry)) := by
  apply host_spec
  · exact RealExecution.scalar_real
  · intro p h plan s
    exact Source.certificates ops RealExecution.algebra backward (layout p h) (valid p h) plan.order s

#axiomsClean geglu_direct_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton
