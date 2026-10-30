import bench.repairs.tritonbench_g.geglu_tanh_triton.Kernels
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.GegluTanhTriton.RawCorrect
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section

/-- Complete forward/backward source grids over raw allocation descriptors,
physical memory, capacity/separation and a full legal program schedule. -/
theorem geglu_kernel_correctness {M β : Type} (ops : Ops M ℝ β)
    (candidate backward : Bool) (p : Layout) (valid : Valid p)
    (plan : RawKernel.Plan p) (entry : M) :
    VeriTile.Spec.Real (∃ final,
      RawKernel.Executes ops RealExecution.algebra candidate backward p valid plan entry final ∧
      RawKernel.RunSpec ops RealExecution.formula backward p valid entry final) := by
  refine ⟨_,RawKernel.executes ops RealExecution.algebra candidate backward p valid plan entry,?_⟩
  exact RawKernel.run_spec ops (Tile.scalar RealExecution.algebra) RealExecution.formula
    RealExecution.scalar_real backward p valid plan entry

#specNonCircular RealExecution.formula avoiding [source,indexCommutedSource,selectedSource,Tile.scalar]
#axiomsClean geglu_kernel_correctness
#axiomsClean Math.backward_is_gradient
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.RawCorrect
