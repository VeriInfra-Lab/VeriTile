import bench.tritonbench_g.iv_dependent_matmul.Proofs.ScheduleEquivalence

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.FPEquiv
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue
open IvDependentMatmul
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Compare two distinct existing pointer schedules over their complete native
TF32 K8 grids. Both keep the same live FP32 carry, tail lanes, FP16 narrowing,
FP32 store and exact input-cell contract. No numerical rewrite is assumed. -/
specification kernel_native (rest : Algebra Word) (profile : Target.NativeProfile)
    (g : Kernel.Geometry) (schedule : Kernels.Schedule) (different : schedule ≠ g.schedule)
    (layout : IO.Layout g) (launchable : Index64.Launchable g.request) :
    original rest profile g ≡[[]] pointerScheduled rest profile g schedule :=
  ScheduleEquivalence.equivalent rest profile g schedule layout launchable

#print_fp_assumptions kernel_native
#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.FPEquiv
