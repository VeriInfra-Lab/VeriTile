import bench.tritonbench_g.attn_fwd_causal.IO
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealPublicIO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.LogicalReader
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Exact equivalence of the entire previous IO judgment, not only values. -/
theorem statement_iff (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = RealIO.formula p work) :
    (LogicalIO.io p work).Implements f ↔
      (RealIO.program p work).Implements (RealIO.formula p work) := by
  change (RealIO.program p work).Implements ((LogicalIO.io p work).lift f) ↔ _
  rw [connection]

/-- Source correctness reused after a checked complete-formula equality. -/
theorem correct (p : Parameters) (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS) (work : List Program) (legal : (programs (grid p)).Perm work)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = RealIO.formula p work) :
    (LogicalIO.io p work).Implements f :=
  (statement_iff p work f connection).mpr (RealPublicIO.implements p apart work legal)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttnFwdCausal.LogicalReader
