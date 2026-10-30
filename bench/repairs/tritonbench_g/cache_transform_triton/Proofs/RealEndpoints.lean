import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.CacheTransform.RealEndpoints
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete byte-copy source on raw regions. The arbitrary Real observation
interprets copied input symbols; no floating encoding is imposed on Real data. -/
theorem kernel_real (M : Algebra ℝ) (p : Parameters)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit)
    (s : State ℝ) (n capacity : Nat) (rows cache : Nat → Nat) (observe : Nat → ℝ)
    (input : 0<p.count → GatherInput p s n capacity rows cache) :
    Spec.Real (∃ t,KernelSource.launch M p s (blocks p.count)=some t ∧
      (∀ o,o<p.count → t.mem p.output o=.mk .nat (byteTarget p rows cache o)) ∧
      (∀ o,o<p.count → observe ((t.mem p.output o).read .nat)=
        observe (cache (rows (o/p.rowBytes)*p.rowBytes+o%p.rowBytes))) ∧
      (∀ (r : RegionName) o,r≠p.output ∨ p.count≤o → t.mem r o=s.mem r o)) := by
  obtain ⟨t,run,values,frame⟩ := KernelInterface.complete M p start block grid s n capacity rows cache input
  refine ⟨t,run,values,?_,frame⟩
  intro o ho
  rw [values o ho,Cell.read_same]
  rfl

#axiomsClean kernel_real
#specNonCircular byteTarget avoiding [source,KernelSource.launch,gather]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.RealEndpoints
