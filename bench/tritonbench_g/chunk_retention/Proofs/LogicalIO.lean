import bench.tritonbench_g.chunk_retention.IO
import bench.tritonbench_g.chunk_retention.Proofs.PublicCorrectness

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.LogicalTensors
open VeriTile Triton FP.Structural KernelPacket
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original judgment, in both directions, for the original
canonical formula after bijective input/output tensor coordinates. -/
theorem full_canonical_iff (a : Call) (p grid : Nat→Nat) :
    (io a p grid ⊨ fun values => outputs a.shape ((PublicIO.realFormula a p) ((inputs a.shape).symm values))) ↔
      (PublicIO.realIO a p grid).Implements (PublicIO.realFormula a p) := by
  change (PublicIO.realIO a p grid).Implements (fun raw => (outputs a.shape).symm (outputs a.shape ((PublicIO.realFormula a p) ((inputs a.shape).symm (inputs a.shape raw))))) ↔ (PublicIO.realIO a p grid).Implements (PublicIO.realFormula a p)
  simp only [Equiv.symm_apply_apply]


theorem correct (a : Call) (p grid : Nat→Nat) (legal : PublicIO.Legal a p grid) :
    io a p grid ⊨ fun values => outputs a.shape ((PublicIO.realFormula a p) ((inputs a.shape).symm values)) :=
  (full_canonical_iff a p grid).mpr (PublicIO.real_implements a p grid legal)

#axiomsClean full_canonical_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.LogicalTensors
