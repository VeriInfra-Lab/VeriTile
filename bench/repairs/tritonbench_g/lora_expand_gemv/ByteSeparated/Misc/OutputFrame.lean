import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Physical

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

theorem sentinel_frame (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (t : Inputs) (s : Scalars) (spans : Backings)
    (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (b n : Nat) (hb : b<t.indices.numel)
    (hn : n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w)
    (sentinel : indices t before b=-1) (byte : Fin 4) :
    sourceStores rest ops schedule plans t s before t.y.device.index
      ((tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).address+byte.val)=
    before t.y.device.index
      ((tensorKey t.y (yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)).address+byte.val) := by
  apply source_frame
  intro q hq active covered
  have hbyte:=byte.isLt
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,tensorKey] at lo hi
  have equal : offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q=
      yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n := by omega
  have coords:=coordinates (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q hq
  have row:=y_injective (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s)
    (indices t before) (valid t s spans (indices t before) ok) _ _ b n coords.2 coords.1 hb hn equal
  exact active (by rw [row.1];exact sentinel)

theorem output_padding_frame (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (t : Inputs) (s : Scalars) (before : Raw.State)
    (off : Nat) (outside : ∀ b n,b<t.indices.numel →
      n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w →
      off≠yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n)
    (byte : Fin 4) :
    sourceStores rest ops schedule plans t s before t.y.device.index
      ((tensorKey t.y off).address+byte.val)=before t.y.device.index ((tensorKey t.y off).address+byte.val) := by
  apply source_frame
  intro q hq _ covered
  have coords:=coordinates (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q hq
  have hbyte:=byte.isLt
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,tensorKey] at lo hi
  have equal : off=offset (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) q := by omega
  exact outside _ _ coords.2 coords.1 equal

#axiomsClean sentinel_frame
#axiomsClean output_padding_frame
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
