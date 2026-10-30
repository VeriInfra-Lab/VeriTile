import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.LegacyPastedSurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.Source
open VeriTile.Triton

theorem whole_source_preserved (commuted : Bool) (p : Parameters) :
    kernel commuted p = LegacyPastedSurface.kernel commuted p := by
  cases commuted <;> rfl

theorem unused_arguments_preserved (p : Parameters) (stride_kcx HEAD_DIM : Nat) :
    quotedOriginal p stride_kcx HEAD_DIM = originalKernel p ∧
      quotedCommuted p stride_kcx HEAD_DIM = commutedKernel p := ⟨rfl,rfl⟩

#axiomsClean whole_source_preserved
#axiomsClean unused_arguments_preserved
end VeriTile.Bench.Repairs.KcacheCopyTriton.Source
