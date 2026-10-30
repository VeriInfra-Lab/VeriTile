import bench.tritonbench_g.iv_dependent_matmul.Proofs.Interface

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.KernelRealResult
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete quotation grid, independent full-K product and exterior frame on
raw geometry, typed allocation windows and arbitrary initial Real memory. -/
theorem kernel_real (rest : Algebra ℝ) (g : Kernel.Geometry) (layout : State Word)
    (a : Allocation g.request layout) (d : Index64.Addressable g.request layout a)
    (l : Index64.Launchable g.request) (initial : State ℝ) :
    ∃ final,Kernel.grid (RealExecution.backend rest) .source g.request initial=some final ∧
      Ideal.Result g.request initial final := by
  obtain ⟨final,run,memory⟩:=Kernel.real_grid rest g.request layout a d l initial
  have result:=(Ideal.host_correct .source g.request layout a d l initial).2
  refine ⟨final,run,?_⟩
  simpa only [Ideal.Result,Ideal.Good,memory] using result

#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.KernelRealResult
