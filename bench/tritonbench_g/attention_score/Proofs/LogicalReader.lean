import bench.tritonbench_g.attention_score.IO
import bench.tritonbench_g.attention_score.Proofs.RealIOProof

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.LogicalReader
open VeriTile Triton Reused Geometry Physical
noncomputable section

theorem statement_iff (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (connection : (LogicalIO.io c offset size scale window complement).lift f = RealInterface.formula c offset size scale window complement) :
    (LogicalIO.io c offset size scale window complement).ImplementsOn (LogicalIO.initialOutput c) f ↔
      (RealInterface.programIO c offset size scale window complement).ImplementsOn (RealInterface.initialOutput c) (RealInterface.formula c offset size scale window complement) := by
  change (RealInterface.programIO c offset size scale window complement).ImplementsOn (RealInterface.initialOutput c) ((LogicalIO.io c offset size scale window complement).lift f) ↔ _
  rw [connection]

theorem correct (c : Shape) (keyRange : c.N≠0 → c.outputSize≠0 → c.programs*c.B<2147483648) (offset size : Int) (scale : ℝ) (window complement : Bool) (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (connection : (LogicalIO.io c offset size scale window complement).lift f = RealInterface.formula c offset size scale window complement) :
    (LogicalIO.io c offset size scale window complement).ImplementsOn (LogicalIO.initialOutput c) f :=
  (statement_iff c offset size scale window complement f connection).mpr (RealInterface.Proof.implements c keyRange offset size scale window complement)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.LogicalReader
