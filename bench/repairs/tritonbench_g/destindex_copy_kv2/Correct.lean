import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.LogicalReader
import bench.repairs.tritonbench_g.destindex_copy_kv2.IO
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option autoImplicit false

/-- FP32 source/metadata/output channels and the complete explicit launch
schedule. IO.lean reuses kv1's typed finite windows and binds the actual kv2 source. -/
def copyIO (rest : Algebra ℝ) (c : Config) (ps : List Nat) := CopyIO.copyIO rest c ps

/-- Independent row-copy formula: every source[row,head,dim] is stored at
output[destination[row],head,dim]. The source omits a feature mask, so its exact
width condition is an explicit premise. All complete program orders are covered. -/
specification destindex_copy_kv2_real (rest : Algebra ℝ) (c : Config)
    (width : block c.dims = c.dims) (ps : List Nat)
    (schedule : ∀ p, p∈ps ↔ p<c.count) :
    Spec.Real ((CopyIO.logicalIO rest c ps) ⊨{CopyIO.LogicalAdmissible c}
      (fun xs row head dim => xs.payload row head dim)) :=
  CopyIO.logical_correct rest c width ps schedule

#axiomsClean destindex_copy_kv2_real
#stmtSurfaceSubset destindex_copy_kv2_real ⊆
  [VeriTile.Bench.Repairs.DestindexCopyKv1.Config.count,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.dims,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.heads,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalInputs.payload,
   VeriTile.Bench.Repairs.DestindexCopyKv2.Config,
   VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO.LogicalAdmissible,
   VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO.LogicalInputs,
   VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO.LogicalOutputs,
   VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO.logicalIO,
   VeriTile.Bench.Repairs.DestindexCopyKv2.block,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.Algebra,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn]

end
end VeriTile.Bench.Repairs.DestindexCopyKv2.Correct
