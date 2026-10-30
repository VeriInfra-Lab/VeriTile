import bench.tritonbench_g.attention_forward_triton.IO
import bench.tritonbench_g.attention_forward_triton.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.LogicalReader
open VeriTile Triton FP.Structural Source Whole AttentionIO
noncomputable section

/-- Exact equivalence of the entire previous IO judgment, not only values. -/
theorem statement_iff (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = AttentionIO.mathematical p) :
    (LogicalIO.io p work).ImplementsOn (LogicalIO.finiteInputs p) f ↔
      (AttentionIO.realIO p work).ImplementsOn ((fun raw => AttentionIO.RealInputs p raw)) (AttentionIO.mathematical p) := by
  change (AttentionIO.realIO p work).ImplementsOn ((fun raw => AttentionIO.RealInputs p raw)) ((LogicalIO.io p work).lift f) ↔ _
  rw [connection]

/-- Source correctness reused after a checked complete-formula equality. -/
theorem correct (p : Parameters) (sep : Separated p) (injective : RawKernel.OutputInjective p) (work : List Program) (legal : (programs (grid p)).Perm work)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = AttentionIO.mathematical p) :
    (LogicalIO.io p work).ImplementsOn (LogicalIO.finiteInputs p) f :=
  (statement_iff p work f connection).mpr (AttentionIO.real_implements p sep injective work legal)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.LogicalReader
