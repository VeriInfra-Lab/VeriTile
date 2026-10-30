import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.IO
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Real
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.NativeKernel

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Reference
open VeriTile Triton FP.Structural Whole KernelNative
noncomputable section
open scoped BigOperators

/-- Independent Real score formula: grouped-head dot product divided by sqrt D. -/
def real (c : Call) (m : Metadata c) (xs : Interface.Inputs ℝ c) : Interface.Outputs ℝ c :=
  Interface.result fun b h t =>
    (∑ d, Interface.query xs b h d *
      keyAt c xs (m.page b.val t.val) (h.val/(c.heads/c.kvHeads)) d) * idealScale c.dim

/-- The logical padded dot tile. Unconverted FP32 inputs, inactive K zeros and the
source's exact multiply precision remain explicit, without reading any State. -/
def products {α β : Type} [Inhabited α] (M : Algebra β)
    (encode : RegionName → Nat → α → β) (c : Call) (m : Metadata c)
    (xs : Interface.Inputs α c) (b : Fin c.batches) (h : Fin c.heads) (block : Nat) :
    Values β .real [32,c.dim] := fun j =>
  let q := encode "Q" (queryAddress c b.val h.val j.2.1.val)
    (Interface.query xs b h j.2.1)
  let t := block*32+j.1.val
  let k := if t < m.len b.val then
      encode "K" (keyAddress c (m.page b.val t) (h.val/(c.heads/c.kvHeads)) j.2.1.val)
        (keyAt c xs (m.page b.val t) (h.val/(c.heads/c.kvHeads)) j.2.1)
    else M.cast none .real .fp32 (M.literal none .real 0)
  M.binary none .fp32 .mul q k

/-- Reduce the logical dot tile, multiply the exact selected scalar and cast
to FP32 storage. For native graphs the reduction plan sees shapes, not words. -/
def score {α β : Type} [Inhabited α] (M : Program → Algebra β)
    (encode : RegionName → Nat → α → β) (c : Call) (m : Metadata c) (scale : ℝ)
    (xs : Interface.Inputs α c) (b : Fin c.batches) (h : Fin c.heads)
    (t : Fin c.maxLength) : β :=
  let A := M (b.val,h.val,t.val/32)
  A.cast none .real .fp32
    (A.binary (some .fp32) .real .mul
      (A.cast (some .fp32) .fp32 .real (A.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false
        (products A encode c m xs b h (t.val/32))
        (⟨t.val%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit)))
      (A.literal (some .fp32) .real scale))

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Reference
