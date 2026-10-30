import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.LogicalReader
import bench.repairs.tritonbench_g.attn_fwd_causal.RealIO
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.RealPublicIO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Correct
open VeriTile Triton FP.Structural Source Whole
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def attentionScore (q k : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m j : Nat) : ℝ :=
  ((Finset.range 128).sum (fun d => q m d*k j d))*qs (m/128)*ks (j/64)

def attention (N : Nat) (q k v : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m d : Nat) : ℝ :=
  ((List.range (min N (m+1))).map (fun j => pow2 (attentionScore q k qs ks m j)*v j d)).sum /
    ((List.range (min N (m+1))).map (fun j => pow2 (attentionScore q k qs ks m j))).sum

/-- Independent causal softmax-weighted sum at every batch/head/row/feature.
The shared IO judgment includes actual source access safety, successful full
grid execution, all typed outputs and the complete exterior cell frame. Empty
dimensions and read-only aliases retain their original kernel domain. -/
specification kernel_real (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (LogicalIO.io p work ⊨ fun xs b h row col =>
      attention p.length (xs.query b h) (xs.key b h) (xs.value b h)
        (xs.queryScale b h) (xs.keyScale b h) row.val col.val) :=
  LogicalReader.correct p apart work legal _ rfl

#axiomsClean kernel_real
#specNonCircular attention avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Correct
