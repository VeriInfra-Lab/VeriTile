import bench.tritonbench_g.destindex_copy.IO
import bench.tritonbench_g.destindex_copy_kv1.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.DestindexCopy.ReaderProof
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
open DestindexCopyKv1
noncomputable section

theorem index_commuted_equiv (p : Layout) :
    (⟨kernel p, []⟩ : Program) ≡[[]] (⟨indexCommutedKernel p, []⟩ : Program) :=by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM s hd
  let t := DestindexCopyKv1.Rewrite.execution (FP.ProfiledRewrite.engine M) p s
  exact ⟨t.val, t.val, t.property,
    (DestindexCopyKv1.Rewrite.exec_eq _ p s).symm.trans t.property, rfl⟩

end
end VeriTile.Bench.Repairs.DestindexCopy.ReaderProof
