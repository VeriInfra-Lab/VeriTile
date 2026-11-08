import bench.tritonbench_g.token_attn_mistral.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile Triton FP.Structural TokenAttnMistral
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Every submitted job implements its gathered PV sum. All seven input
channels, arbitrary strides, distinct masks, attention metadata equality and
all output lanes are retained. The sliding-window guard remains visible. -/
specification kernel_correct (p : Parameters) (work : List Grid.PID)
    (counts : Fin work.length → Nat) (positive : 0<p.block)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    Spec.Real (Logical.io p work counts ⊨{Logical.domain p work counts}
      fun xs job d => ∑ n ∈ Finset.range (xs.attention job),
        if n + max (xs.sequence job-p.window) 0 < xs.sequence job then
          Reference.probAt 0 xs job n * Reference.valueAt 0 xs job n d else 0) :=
  (Logical.judgment_iff p work counts).2 (PublicIO.real p work counts positive separate unique)

#specNonCircular Logical.domain avoiding [kernel,TokenAttnMistral.kernel,FP.Structural.exec,Grid.run,outputValues]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
