import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelReal
import VeriTile.Meta.Specification
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.KernelRealResult
open VeriTile.Triton
noncomputable section

/-- Full source execution over raw kernel geometry and memory. Signed metadata
lookups select the destination; every output copies the independent source
address value and all cells outside the destination slice are unchanged. -/
theorem kcache_copy_triton_real (p : Source.Parameters) (s : BlockState)
    (injective : Function.Injective (KernelReal.targetOffset p s)) :
    Spec.Real (∃ final, exec (Source.originalKernel p).toAlgKernel s = some final ∧
      (∀ i : TileIndex [p.width], final.readMem p.KCache (KernelReal.targetOffset p s i) =
        s.readMem p.K (KernelReal.sourceOffset p s.pids i)) ∧
      (∀ (r : RegionName) o, (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width],
        o ≠ KernelReal.targetOffset p s i) → final.mem r o = s.mem r o)) :=
  ⟨(KernelReal.execution p s).val,(KernelReal.execution p s).property,
    KernelReal.readback p s injective,KernelReal.frame p s⟩

#axiomsClean kcache_copy_triton_real
#specNonCircular KernelReal.sourceOffset avoiding [Source.originalKernel,exec,KernelReal.execution]
#specNonCircular KernelReal.targetOffset avoiding [Source.originalKernel,exec,KernelReal.execution]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.KernelRealResult
