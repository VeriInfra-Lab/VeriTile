import bench.tritonbench_g.attention_forward_triton.Proofs.LogicalReader
import bench.tritonbench_g.attention_forward_triton.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Correct
open VeriTile Triton FP.Structural Source Whole AttentionIO
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def attentionScore (q k : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m j : Nat) : ℝ :=
  ((Finset.range 96).sum (fun d => q m d*k j d))*qs (m/128)*ks (j/64)

def attention (N : Nat) (q k v : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m d : Nat) : ℝ :=
  ((List.range N).map (fun j => pow2 (attentionScore q k qs ks m j)*v j d)).sum /
    ((List.range N).map (fun j => pow2 (attentionScore q k qs ks m j))).sum

/-- Complete noncausal attention on the explicit finite Q/K/V/scale channels,
for any legal full-grid order. Actual source safety, successful execution,
every FP16-tagged finite Real result and the exact exterior frame are proved.
The independent formula is the scaled base-two softmax weighted value sum. -/
specification attention_kernel_real (p : Parameters) (sep : Separated p)
    (injective : RawKernel.OutputInjective p) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    Spec.Real (LogicalIO.io p work ⊨{LogicalIO.finiteInputs p} fun xs b h row col =>
      some (attention p.length (xs.query b h) (xs.key b h) (xs.value b h)
        (xs.queryScale b h) (xs.keyScale b h) row.val col.val)) :=
  LogicalReader.correct p sep injective work legal _ rfl

#specNonCircular attention avoiding
  [Source.kernel,runPrograms,FP.Structural.exec,Source.accumulated]
#axiomsClean attention_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Correct
