import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelRealLoop
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelRealResults
open VeriTile.Triton
noncomputable section

/-- Complete static-loop source on raw parameters and initial memory. Distinct
active output addresses give every K/V snapshot value; all other cells are
preserved. Length/table values are read directly from the raw state. -/
theorem kv_cache_copy_real (p : Source.Parameters) (s : BlockState)
    (unique : ∀ key u v,u<p.dims/p.width → v<p.dims/p.width →
      ∀ i j : TileIndex [p.width],
        (KernelReal.targetInt p s key u i).toNat=(KernelReal.targetInt p s key v j).toNat →
          u=v ∧ i=j) :
    Spec.Real (∃ final,exec (Source.originalKernel p).toAlgKernel s=some final ∧
      final.mem=(KernelReal.copied p s (p.dims/p.width)).mem ∧
      (∀ key q,q<p.dims/p.width → ∀ i : TileIndex [p.width],
        final.readMem (KernelReal.outputRegion key) (KernelReal.targetInt p s key q i).toNat=
          s.readMem (KernelReal.inputRegion key) (KernelReal.sourceOffset p s.pids key q i)) ∧
      (∀ (r : RegionName) o,(∀ key q,q<p.dims/p.width → r≠KernelReal.outputRegion key ∨
        ∀ i : TileIndex [p.width],o≠(KernelReal.targetInt p s key q i).toNat) →
        final.mem r o=s.mem r o)) := by
  obtain ⟨final,run,memory⟩:=KernelReal.execute p s
  refine ⟨final,run,memory,?_,?_⟩
  · intro key q hq i
    simpa only [BlockState.readMem,memory] using
      KernelReal.copied_readback p s (p.dims/p.width) key q hq i (unique key)
  · intro r o outside
    rw [memory]
    exact KernelReal.copied_frame p s _ r o outside

#axiomsClean kv_cache_copy_real
#specNonCircular KernelReal.copied avoiding [Source.originalKernel,Source.kernel,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelRealResults
