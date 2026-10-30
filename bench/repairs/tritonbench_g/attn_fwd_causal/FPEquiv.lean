import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.PairIO

/-! Complete repaired-source comparison with an integer-CSE candidate.
Both programs retain TF32 QK, FP16 P/V/PV and output, FP32 state, explicit
causal masks, both stages and the same local instruction interpretation. -/
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Equivalence
open VeriTile Triton FP.Structural Source Whole
open scoped VeriTile.Spec
noncomputable section

abbrev original (instructions : Native.Instructions) (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.original (NativeIO.program instructions p work) (fun _ => True)

abbrev quotientRemainder (instructions : Native.Instructions) (p : Parameters) (work : List Program) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate instructions p work) (fun _ => True)

/-- Exact whole-kernel equality. All native input words and any complete CTA
order are retained, with both successful safe runs, tagged outputs and frames.
No numerical rewrite or experimental rule is required. -/
specification quotient_remainder_equiv (instructions : Native.Instructions) (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    original instructions p work ≡[[]] quotientRemainder instructions p work := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.candidate instructions p work)
    (PairIO.equivalent instructions p apart work legal)

#axiomsClean quotient_remainder_equiv
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Equivalence
