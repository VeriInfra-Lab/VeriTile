import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.FPValues

/-! Ordered value formula for the actual pasted source. The shared operand
loads, FP16 products and FP16 reduction stay unchanged. The optional FP32
addition already has output precision; only the real half-to-FP32 widening
remains. The legacy Direct formula is kept for its explicit-cast program. -/
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.SourceValues
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
variable {α : Type} [Inhabited α]

def result (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch slot row : Nat) (data : Data α) (n : Fin 256) : α :=
  let total := M.cast (some .fp32) .fp16 .real
    (sums M reduction h batch slot row data (n,PUnit.unit))
  if h.add then M.binary (some .fp32) .real .add total (old M h batch row data n)
  else total

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.SourceValues
