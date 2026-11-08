import bench.tritonbench_g.token_attn_llama2.Proofs.LogicalIO
import bench.tritonbench_g.token_attn_llama2.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Correct
open VeriTile Triton FP.Structural Whole
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Complete quoted score grid on query/key tensors and all signed metadata
snapshots; the original addressed adapter, output mask and frame are retained. -/
abbrev kernelIO := Logical.io

/-- Every active gathered score is its grouped-head dot product divided by
sqrt D, on the original broad Valid stride and metadata domain. -/
specification token_attention_kernel_correct (c : Call) (m : Metadata c)
    (domain : Valid c m) (work : List Program) (order : (programs c.grid).Perm work) :
    Spec.Real (kernelIO c m work ⊨{Logical.domain c m} fun xs b h t =>
      (∑ d : Fin c.dim, xs.query b h d *
        xs.keyAt (m.page b.val t.val) (h.val/(c.heads/c.kvHeads)) d) *
      (1 / Real.sqrt (c.dim : ℝ))) :=
  (Logical.judgment_iff c m work).2 (IOImplements.real c m domain work order)

#specNonCircular Logical.domain avoiding [source,FP.Structural.exec,NativeGrid.runPrograms]
#axiomsClean token_attention_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Correct
