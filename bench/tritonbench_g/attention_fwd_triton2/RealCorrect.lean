import bench.tritonbench_g.attention_fwd_triton2.Proofs.LogicalReader
import bench.tritonbench_g.attention_fwd_triton2.Proofs.MathematicalIO
import VeriTile.Meta.Specification

/-! Complete-grid normalized base-2 attention. IO.lean exposes finite input
windows, all output coordinates, source precision and actual launch metadata.
The original raw/Tensor memory consequences remain in TensorBridge. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Correct
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def attentionScore (q k : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m j : Nat) : ℝ :=
  ((Finset.range 96).sum (fun d => q m d*k j d))*qs (m/128)*ks (j/64)

def attention (N : Nat) (q k v : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m d : Nat) : ℝ :=
  ((List.range N).map (fun j => pow2 (attentionScore q k qs ks m j)*v j d)).sum /
    ((List.range N).map (fun j => pow2 (attentionScore q k qs ks m j))).sum

/-- Every logical BF16-tagged output, in any complete grid permutation,
implements the independent normalized attention formula. The shared IO
judgment includes actual source access safety and the complete exterior frame. -/
specification kernel_raw_correct (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    Spec.Real (LogicalIO.io p work ⊨ fun xs b h row col =>
      attention p.length (xs.query b h) (xs.key b h) (xs.value b h)
        (xs.queryScale b h) (xs.keyScale b h) row.val col.val) :=
  LogicalReader.correct p bases dom sep work order _ (by intros; rfl)

#specNonCircular attention avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update,Source.kernel]
#axiomsClean kernel_raw_correct
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Correct
