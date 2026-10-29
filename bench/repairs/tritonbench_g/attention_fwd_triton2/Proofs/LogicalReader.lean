import bench.repairs.tritonbench_g.attention_fwd_triton2.IO
import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.MathematicalIO

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.LogicalReader
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section

/-- Exact equivalence of the entire previous IO judgment, not only values. -/
theorem statement_iff (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (connection : (LogicalIO.io p work).lift f = Interface.realFormula p) :
    (LogicalIO.io p work).Implements f ↔
      (Interface.io Real.algebra Real.dotModel p some (fun _ v => v.unbotD 0) work).Implements (Interface.realFormula p) := by
  change (Interface.io Real.algebra Real.dotModel p some (fun _ v => v.unbotD 0) work).Implements ((LogicalIO.io p work).lift f) ↔ _
  rw [connection]

theorem formula_connection (p : Parameters) (work : List Program)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (pointwise : ∀ values b h row col,
      f ((LogicalIO.inputsEquiv p work) values) b h row col =
        Interface.realFormula p values (0 : Fin 1) (b,h,row,col,())) :
    (LogicalIO.io p work).lift f = Interface.realFormula p := by
  funext values o j
  have zero : o = (0 : Fin 1) := Fin.eq_zero o
  subst o
  rcases j with ⟨b,h,row,col,⟨⟩⟩
  exact pointwise values b h row col

/-- Source correctness reused after a checked complete-formula equality. -/
theorem correct (p : Parameters) (bases : Bases) (dom : Domain p bases) (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS) (work : List Program) (order : (programs (grid p)).Perm work)
    (f : LogicalIO.Inputs p → LogicalIO.Outputs p)
    (pointwise : ∀ values b h row col,
      f ((LogicalIO.inputsEquiv p work) values) b h row col =
        Interface.realFormula p values (0 : Fin 1) (b,h,row,col,())) :
    (LogicalIO.io p work).Implements f :=
  (statement_iff p work f (formula_connection p work f pointwise)).mpr (MathematicalIO.implements p bases dom sep work order)

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.LogicalReader
