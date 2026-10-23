import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Direct
namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Native
open VeriTile.Triton

/-- Every evaluated source offset is exact, including inactive padded lanes. -/
theorem padded_offset_exact (p : Parameters) (b : Bounds p xb sb ob scaleCount)
    (pid lane : Nat) (hr : pid<p.rows) (hl : lane<p.width) :
    offset p.cols pid lane=(pid*p.cols+lane : Nat) ∧ pid*p.cols+lane<2^51 := by
  have rp : pid<2^31 := hr.trans b.grid
  have lp : lane<2^20 := lt_of_lt_of_le hl b.width
  have cp : p.cols≤2^20 := le_trans p.covers b.width
  have product : pid*p.cols≤(2^31-1)*2^20 := Nat.mul_le_mul (by omega) cp
  have sum : pid*p.cols+lane<2^51 := by omega
  have rowEq : signed64 (int32 pid)=(pid : Int) := by
    rw [int32_exact _ (by omega) (by omega),signed64_exact _ (by omega) (by omega)]
  have colEq : signed64 (int32 lane)=(lane : Int) := by
    rw [int32_exact _ (by omega) (by omega),signed64_exact _ (by omega) (by omega)]
  have productEq : signed64 ((pid : Int)*(p.cols : Int))=((pid*p.cols : Nat) : Int) := by
    rw [←Nat.cast_mul]
    apply signed64_exact <;> omega
  have sumEq : signed64 (((pid*p.cols : Nat) : Int)+(lane : Int))=((pid*p.cols+lane : Nat) : Int) := by
    rw [←Nat.cast_add]
    apply signed64_exact <;> omega
  exact ⟨by simp only [offset,rowEq,colEq,productEq,sumEq],sum⟩

theorem checked_alignment (c : Direct.Request) (positive : 0<c.x.metadata.numel) :
    (Direct.xView c).base%4=0 ∧ (Direct.scaleView c).base%4=0 := by
  constructor
  · apply VeriTile.Torch.FP32.dataView_aligned c.x c.xLaws c.contract.contiguous.1
    intro h
    exact (c.contract.alignment h).1
  · apply VeriTile.Torch.FP32.dataView_aligned c.scale c.scaleLaws c.contract.contiguous.2
    intro _
    exact (c.contract.alignment positive).2

#axiomsClean padded_offset_exact
#axiomsClean checked_alignment
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Native
