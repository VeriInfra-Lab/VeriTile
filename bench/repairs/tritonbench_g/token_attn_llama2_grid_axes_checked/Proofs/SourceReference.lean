import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Kernels
import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.FPReference
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
/-- All leaves come from the initial memory; the two input channels are fp32.
The original loads have no explicit conversion. -/
def queryValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (d : Fin p.dimensions) : α :=
  (s.mem p.q (queryOffset p s d)).read .fp32

def keyValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) (d : Fin p.dimensions) : α :=
  if active p s i then (s.mem p.k (keyOffset p s i d)).read .fp32
  else M.cast none .real .fp32 (M.literal none .real 0.0)

def products {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.tokens,p.dimensions] := fun i =>
  M.binary none .fp32 .mul (queryValue M p s i.2.1)
    (keyValue M p s (i.1,PUnit.unit) i.2.1)

/-- Whole row reduction followed by the original scalar multiplication and
FP32 storage. The scale is a strong runtime FP32 operand. -/
def score {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) : α :=
  M.cast none .real .fp32
    (M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) .fp32 .real
        (M.reduceSum (some .fp32) ⟨1, by simp⟩ Bool.false (products M p s) i))
      (M.literal (some .fp32) .real p.scale))

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  if blockActive p s then
    (TileShape.allIndices [p.tokens]).foldl (fun t i =>
      if active p s i then t.write p.output (outputOffset p s i) (.mk .fp32 (score M p s i))
      else t) s
  else s

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
