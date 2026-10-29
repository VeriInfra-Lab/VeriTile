import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16IOSafety
import bench.repairs.tritonbench_g.attention_kernel_aligned.IO
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.LogicalReader
open VeriTile Triton FP.Structural Full Whole Interface
noncomputable section

/-- Exact equivalence of the entire previous IO judgment, not only values. -/
theorem statement_iff (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = Interface.realFormula p work) :
    (LogicalIO.io p work).Implements f ↔
      (Interface.realIO p work).Implements (Interface.realFormula p work) := by
  change (Interface.realIO p work).Implements ((LogicalIO.io p work).lift f) ↔ _
  rw [connection]

/-- Source correctness reused after a checked complete-formula equality. -/
theorem correct (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p) (work : List Program) (order : (programs (grid p)).Perm work)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = Interface.realFormula p work) :
    (LogicalIO.io p work).Implements f :=
  (statement_iff p work f connection).mpr (Interface.real_implements p g separate work order)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.LogicalReader

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.LogicalReader
open VeriTile Triton FP.Structural Full Whole Interface
noncomputable section

/-- Exact equivalence of the entire previous IO judgment, not only values. -/
theorem statement_iff (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = Interface.realFormula p work) :
    (LogicalIO.io p work).Implements f ↔
      (Interface.realIO p work).Implements (Interface.realFormula p work) := by
  change (Interface.realIO p work).Implements ((LogicalIO.io p work).lift f) ↔ _
  rw [connection]

/-- Source correctness reused after a checked complete-formula equality. -/
theorem correct (p : Parameters) (g : Guard p) (separate : Whole.Nonempty p → Separated p) (work : List Program) (order : (programs (grid p)).Perm work)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = Interface.realFormula p work) :
    (LogicalIO.io p work).Implements f :=
  (statement_iff p work f connection).mpr (Interface.real_implements p g separate work order)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.LogicalReader
