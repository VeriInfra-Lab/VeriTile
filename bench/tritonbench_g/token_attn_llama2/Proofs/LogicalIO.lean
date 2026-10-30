import bench.tritonbench_g.token_attn_llama2.IO
import bench.tritonbench_g.token_attn_llama2.Proofs.IOImplements
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Logical
open VeriTile Triton FP.Structural Whole
noncomputable section
theorem judgment_iff (c : Call) (m : Metadata c) (work : List Program) :
    (io c m work).ImplementsOn (domain c m)
      (fun xs b h t => (∑ d : Fin c.dim, xs.query b h d *
        xs.keyAt (m.page b.val t.val) (h.val/(c.heads/c.kvHeads)) d) * (1 / Real.sqrt (c.dim : ℝ))) ↔
    (Interface.io (fun _ => realAlgebra) c m (idealScale c.dim) work (fun _ _ x => x) id).ImplementsOn
      (Interface.MetadataValues c m) (Reference.real c m) := by
  have same : (io c m work).lift
      (fun xs b h t => (∑ d : Fin c.dim, xs.query b h d *
        xs.keyAt (m.page b.val t.val) (h.val/(c.heads/c.kvHeads)) d) * (1 / Real.sqrt (c.dim : ℝ))) =
      Reference.real c m := by
    funext xs port
    fin_cases port
    rfl
  change (Interface.io (fun _ => realAlgebra) c m (idealScale c.dim) work (fun _ _ x => x) id).ImplementsOn _ _ ↔ _
  rw [same]
  rfl
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Logical
