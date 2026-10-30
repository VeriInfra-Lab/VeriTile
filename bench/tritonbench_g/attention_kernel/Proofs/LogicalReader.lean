import bench.tritonbench_g.attention_kernel.IO
import bench.tritonbench_g.attention_kernel.Proofs.IOResults

namespace VeriTile.Bench.Repairs.AttentionKernel.LogicalReader
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section

/-- Exact equivalence of the entire previous IO judgment, not only values. -/
theorem statement_iff (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = Target.mathematical p) :
    (LogicalIO.io p work).Implements f ↔
      (IO.realIO p work).Implements (Target.mathematical p) := by
  change (IO.realIO p work).Implements ((LogicalIO.io p work).lift f) ↔ _
  rw [connection]

theorem formula_connection (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (pointwise : ∀ values b h row col,
      (LogicalIO.storageEquiv p.storage).symm
        (f ((LogicalIO.inputsEquiv p work) values) b h row col) =
          Target.mathematical p values (0 : Fin 1) (b,h,row,col,())) :
    (LogicalIO.io p work).lift f = Target.mathematical p := by
  funext values o j
  have zero : o = (0 : Fin 1) := Fin.eq_zero o
  subst o
  rcases j with ⟨b,h,row,col,⟨⟩⟩
  exact pointwise values b h row col

/-- Source correctness reused after a checked complete-formula equality. -/
theorem correct (p : Parameters) (bases : Bases) (domain : Domain p bases) (hbm : 0<p.bm) (separate : Separated p) (work : List Program) (order : (programs (grid p)).Perm work)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (pointwise : ∀ values b h row col,
      (LogicalIO.storageEquiv p.storage).symm
        (f ((LogicalIO.inputsEquiv p work) values) b h row col) =
          Target.mathematical p values (0 : Fin 1) (b,h,row,col,())) :
    (LogicalIO.io p work).Implements f :=
  (statement_iff p work f (formula_connection p work f pointwise)).mpr (IOResults.real_implements p bases domain hbm separate work order)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionKernel.LogicalReader
