import bench.tritonbench_g.token_attn_llama2.Proofs.IOBinding
import bench.tritonbench_g.token_attn_llama2.Proofs.SourceFormula

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.FormulaBinding
open VeriTile Triton FP.Structural Whole IOBinding
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited α] [Inhabited β]

theorem products_eq (M : Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (encode : RegionName → Nat → α → β) (xs : Interface.Inputs α c)
    (s : State β) (v : Valid c m) (bound : Reads c m xs encode s)
    (b : Fin c.batches) (h : Fin c.heads) (block : Nat) :
    products M (c.parameters scale) (start c.grid (b.val,h.val,block) s) =
      Reference.products M encode c m xs b h block := by
  have hq (d : Fin c.dim) :
      queryValue M (c.parameters scale) (start c.grid (b.val,h.val,block) s) d =
      M.cast (some .fp32) .fp32 .real
        (encode "Q" (Reference.queryAddress c b.val h.val d.val) (Interface.query xs b h d)) := by
    exact congrArg (M.cast (some .fp32) .fp32 .real) (bound.query b h d)
  have hk (i : Fin 32) (d : Fin c.dim) :
      keyValue M (c.parameters scale) (start c.grid (b.val,h.val,block) s) (i,PUnit.unit) d =
      M.cast (some .fp32) .fp32 .real
        (if block*32+i.val < m.len b.val then
          encode "K" (Reference.keyAddress c (m.page b.val (block*32+i.val))
            (h.val/(c.heads/c.kvHeads)) d.val)
            (Reference.keyAt c xs (m.page b.val (block*32+i.val)) (h.val/(c.heads/c.kvHeads)) d)
        else M.cast (some .fp32) .real .fp32 (M.literal (some .fp32) .real 0)) := by
    unfold keyValue
    have hi := active_eq bound v scale (b.val,h.val,block) b.isLt (i,PUnit.unit)
    by_cases hn : block*32+i.val<m.len b.val
    · rw [if_pos (hi.mpr hn),if_pos hn]
      have hpage := (v.pages b _ hn).2
      have hgroup := grouped_head c m v h.val h.isLt
      rw [keyOffset,page_eq bound v scale (b.val,h.val,block) b.isLt (i,PUnit.unit) hn]
      have read := bound.key ⟨_,hpage⟩ ⟨_,hgroup⟩ d
      simpa [Reference.keyAt,hpage,hgroup,Reference.keyAddress,Call.parameters,start,ids] using
        congrArg (M.cast (some .fp32) .fp32 .real) read
    · rw [if_neg (fun ha => hn (hi.mp ha)),if_neg hn]
      norm_num
  funext j
  simp only [products,Reference.products,hq,hk]

theorem score_eq (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (encode : RegionName → Nat → α → β) (xs : Interface.Inputs α c)
    (s : State β) (v : Valid c m) (bound : Reads c m xs encode s)
    (b : Fin c.batches) (h : Fin c.heads) (t : Fin c.maxLength) :
    score (M (b.val,h.val,t.val/32)) (c.parameters scale)
      (start c.grid (b.val,h.val,t.val/32) s) (lane c scale t.val) =
      Reference.score M encode c m scale xs b h t := by
  unfold score
  rw [products_eq _ c m scale encode xs s v bound b h (t.val/32)]
  rfl

theorem real_score (c : Call) (m : Metadata c) (xs : Interface.Inputs ℝ c)
    (b : Fin c.batches) (h : Fin c.heads) (t : Fin c.maxLength)
    (ht : t.val<m.len b.val) :
    Reference.score (fun _ => realAlgebra) (fun _ _ x => x) c m (idealScale c.dim) xs b h t =
      (Reference.real c m xs) ⟨0,by simp [Interface.outputs]⟩ (b,h,t,PUnit.unit) := by
  have split : t.val/32*32+t.val%32=t.val := by omega
  simp [Reference.score,Reference.products,Reference.real,Interface.result,
    realAlgebra,realSum,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    split,ht]
  left
  rfl

#axiomsClean score_eq
#axiomsClean real_score
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.FormulaBinding
