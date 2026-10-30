import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Programs

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.NativeCorrect
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue
open IvDependentMatmul FPEquiv
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete native source observation. Exact active FP32 input cells preserve
the original allocation premise; source safety, all outputs and the raw frame
are conclusions of the shared judgment, for all five pointer schedules. -/
theorem kernel_native (rest : Algebra Word) (profile : Target.NativeProfile)
    (g : Kernel.Geometry) (layout : IO.Layout g) (launchable : Index64.Launchable g.request) :
    (kernelIO rest profile g).ImplementsCells (formula profile g) :=
  IOResults.native rest profile g layout launchable

#specNonCircular formula avoiding [Kernel.grid,FP.Structural.exec]
#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [FP.Structural.Algebra,Word,Target.NativeProfile,Kernel.Geometry,IO.Layout,Index64.Launchable,
   Kernel.Geometry.request,kernelIO,ProgramIO.ImplementsCells,formula,instInhabitedWord]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.NativeCorrect
