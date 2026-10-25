import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.LegacyPastedSurface
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.KvCacheCopy.Source
open VeriTile.Triton

/-- The actual full body, including every split, is the retained complete body.
Call ports state actual roles in parameter order, independent of load order. -/
theorem whole_body_preserved (version : Bool) (p : Parameters) :
    kernel version p = ComputeKernel.mk ["key","value","table","lengths"] ["keyCache","valueCache"]
      (LegacyPastedSurface.kernel version p).surfaceBody := by
  cases version <;> rfl

theorem unused_stride_preserved (p : Parameters) (stride_kcd : Nat) :
    quotedOriginal p stride_kcd = originalKernel p ∧
      quotedCommuted p stride_kcd = commutedKernel p := ⟨rfl,rfl⟩

namespace SourcePreservation
/-- The historical generated source retains the entire old body as well.
The actual source ABI now declares its ports independently of first-load order. -/
theorem kernel (version : Bool) (p : Parameters) :
    (Source.kernel version p).surfaceBody = (LegacySurface.kernel version p).surfaceBody := by
  cases version <;> rfl
#axiomsClean kernel
end SourcePreservation

#axiomsClean whole_body_preserved
#axiomsClean unused_stride_preserved
end VeriTile.Bench.Repairs.KvCacheCopy.Source
