import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPWorkset
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.KernelInterface
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section

/-- Complete source success and frame for every raw batch/split program,
including the sentinel branch. This is a conclusion, never a run premise. -/
def Programs [Inhabited α] (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α) : Prop :=
  ∀ p : Program h, ∃ t,
    exec (machine M P) (Surface.source h) (entry h p.batch.val p.split.val data)=some t ∧
    (data.indices p.batch.val=-1 → t.mem=(entry h p.batch.val p.split.val data).mem) ∧
    (∀ (r : RegionName) off,¬(data.indices p.batch.val≠-1 ∧ r="Y" ∧ ∃ n : Fin h.N,
      off=outputOffset h p.batch.val n.val) → t.mem r off=(entry h p.batch.val p.split.val data).mem r off)

theorem programs [Inhabited α] (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0≤data.indices b) :
    Programs M P h data := by
  intro p
  obtain ⟨t,run,skip,_,frame⟩ := FPReadback.every_program M P h p data (valid p.batch.val p.batch.isLt)
  exact ⟨t,run,skip,frame⟩

#axiomsClean programs
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.KernelInterface
