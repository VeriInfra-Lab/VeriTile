import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Kernel
import bench.tritonbench_g.swiglu_fwd.Proofs.WholePublic

namespace VeriTile.Bench.Repairs.SwigluFwd.KernelRealResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- Complete selected kernel execution, independent activation formula and
exterior regional frame. Both program axes and all source row strides remain
explicit. This is mathematical semantics, not exact native arithmetic. -/
theorem swiglu_kernel_correct (c : Layout) (B row tile : Nat) (entry : State ℝ) :
    VeriTile.Spec.Real (∃ final,
      exec realAlgebra (kernel c.X c.Y c.OUT c.sx c.sy c.so c.cols B)
        (tileState realAlgebra c B row tile entry) = some final ∧
      KernelProof.Result formula c B row tile entry final) := by
  simpa only [KernelProof.Result, scalar_real, formula,
    VeriTile.Bench.TritonBenchG.SwigluFwd.Math.forward] using
    KernelProof.run realAlgebra c B row tile entry

#specNonCircular formula avoiding [kernel]
#axiomsClean swiglu_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd.KernelRealResult
