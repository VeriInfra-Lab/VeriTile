import bench.tritonbench_g.triton_softmax.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax
open VeriTile.Triton

/-- A successful complete source run preserves every output column beyond the
selected tile width, even if n_cols is larger. The wrapper caps that width at
1024, so its wide-row behavior cannot be called a full-row softmax. -/
theorem columns_beyond_tile_unchanged (out inp : RegionName)
    (si so n B : Nat) (hB : 0 < B) (s t : BlockState)
    (he : exec (softmax_kernel out inp si so n B).toAlgKernel s = some t)
    (column : Nat) (hc : B ≤ column) :
    t.mem out (s.pid * so + column) = s.mem out (s.pid * so + column) := by
  obtain ⟨u, hu, _, hf⟩ := softmax_kernel_region_run out inp si so n B hB s
    (fun j => s.readMem inp (s.pid * si + j.val)) (by intro j hj; rfl)
  have hut : u = t := Option.some.inj (hu.symm.trans he)
  subst u
  apply hf out (s.pid * so + column)
  right
  intro j _
  have hj := j.isLt
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax
