import bench.tritonbench_g.cache_transform_triton.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.CacheTransform.Correct
open VeriTile.Triton FP.Structural CacheIO
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Whole-grid row gather of the complete byte stream, using signed row
metadata and named cache row/byte axes. Exact input tags, the empty-launch
domain, actual source safety and complete exterior frame are retained. -/
specification kernel_real (M : Algebra ℝ) (p : Parameters) (n capacity : Nat)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count ≤ gridLimit) :
    Spec.Real (logicalIO M p n capacity ⊨{logicalDomain M p n capacity} fun xs byte =>
      xs.byteAt (xs.rowAt (byte.val/p.rowBytes)).toNat (byte.val%p.rowBytes)) :=
  logical_implements M p n capacity start block grid

#axiomsClean kernel_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.Correct
