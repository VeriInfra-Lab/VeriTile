import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.IO
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOResults

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.LogicalTensors
open VeriTile Triton FP.Structural KernelPacket
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original judgment, in both directions, for the original
canonical formula after bijective input/output tensor coordinates. -/
theorem full_canonical_iff (a : Call) :
    (io a ⊨ fun values => outputs a.shape ((PublicIO.mathematical a) ((inputs a.shape).symm values))) ↔
      (PublicIO.realIO a).Implements (PublicIO.mathematical a) := by
  change (PublicIO.realIO a).Implements (fun raw => (outputs a.shape).symm (outputs a.shape ((PublicIO.mathematical a) ((inputs a.shape).symm (inputs a.shape raw))))) ↔ (PublicIO.realIO a).Implements (PublicIO.mathematical a)
  simp only [Equiv.symm_apply_apply]


theorem correct (a : Call) (chunk : a.stage=.backwardGradient → a.pid.y<a.shape.chunks) :
    io a ⊨ fun values => outputs a.shape ((PublicIO.mathematical a) ((inputs a.shape).symm values)) :=
  (full_canonical_iff a).mpr (PublicIO.real_implements a chunk)

#axiomsClean full_canonical_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.LogicalTensors
