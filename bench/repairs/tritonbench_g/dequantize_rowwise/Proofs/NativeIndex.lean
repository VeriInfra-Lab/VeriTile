import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.InputBinding
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.NativeAddress

namespace VeriTile.Bench.Repairs.DequantizeRowwise.NativeIndex
open NativeAddress InputBinding

/-- Bounds supplied by backed inputs and an existing FP16 output buffer.
INT8 input addresses scale by one byte; scales by four; outputs by two. -/
structure Bounds (p : Parameters) (xb sb ob scaleCount : Nat) : Prop where
  grid : p.rows < 2^31
  width : p.width ≤ 2^20
  scaleRows : p.rows ≤ scaleCount
  xCapacity : xb + p.rows * p.cols ≤ 2^64
  scaleCapacity : sb + 4 * scaleCount ≤ 2^64
  outputCapacity : ob + 2 * (p.rows * p.cols) ≤ 2^64

def offset (cols pid lane : Nat) : Int :=
  signed64 (signed64 (signed64 (int32 pid) * (cols : Int)) + signed64 (int32 lane))

def lane (p : Parameters) (xb sb ob pid column : Nat) : Option (Nat × Nat × Nat) :=
  if signed64 (int32 column) < (p.cols : Int) then
    some (pointer 1 xb (offset p.cols pid column),
      pointer 4 sb (signed64 (int32 pid)), pointer 2 ob (offset p.cols pid column))
  else none

/-- Fixed-width native mask/index/pointer projection for each launched lane.
No arithmetic or address condition is added beyond the public contract and
ordinary buffer capacities. -/
theorem lane_exact (p : Parameters) {xb sb ob scaleCount : Nat}
    (bounds : Bounds p xb sb ob scaleCount) (pid column : Nat)
    (hr : pid < p.rows) (hd : column < p.width) :
    lane p xb sb ob pid column =
      if column < p.cols then
        some (xb + (pid * p.cols + column), sb + 4 * pid, ob + 2 * (pid * p.cols + column))
      else none := by
  have pc : p.cols ≤ 2^20 := le_trans p.covers bounds.width
  have rp : pid < 2^31 := lt_trans hr bounds.grid
  have lp : column < 2^20 := lt_of_lt_of_le hd bounds.width
  have rowEq : signed64 (int32 pid) = (pid : Int) := by
    rw [int32_exact _ (by omega) (by omega), signed64_exact _ (by omega) (by omega)]
  have colEq : signed64 (int32 column) = (column : Int) := by
    rw [int32_exact _ (by omega) (by omega), signed64_exact _ (by omega) (by omega)]
  by_cases active : column < p.cols
  · have indexed : pid * p.cols + column < p.rows * p.cols := by
      have step : (pid + 1) * p.cols ≤ p.rows * p.cols :=
        Nat.mul_le_mul_right _ (by omega)
      simp only [Nat.add_mul, Nat.one_mul] at step
      omega
    have productBound : pid * p.cols ≤ (2^31 - 1) * 2^20 :=
      Nat.mul_le_mul (by omega) pc
    have product : signed64 ((pid : Int) * (p.cols : Int)) = ((pid * p.cols : Nat) : Int) := by
      rw [← Nat.cast_mul]
      exact signed64_exact _ (by omega) (by omega)
    have sum : signed64 (((pid * p.cols : Nat) : Int) + (column : Int)) =
        ((pid * p.cols + column : Nat) : Int) := by
      rw [← Nat.cast_add]
      exact signed64_exact _ (by omega) (by omega)
    have mask : (column : Int) < (p.cols : Int) := by exact_mod_cast active
    simp only [lane, colEq, mask, ↓reduceIte, active, offset, rowEq, product, sum]
    rw [pointer_exact 1 xb (p.rows * p.cols) _ (by decide) (by simpa using bounds.xCapacity) indexed,
      pointer_exact 4 sb scaleCount pid (by decide) bounds.scaleCapacity (lt_of_lt_of_le hr bounds.scaleRows),
      pointer_exact 2 ob (p.rows * p.cols) _ (by decide) bounds.outputCapacity indexed]
    simp
  · have mask : ¬(column : Int) < (p.cols : Int) := by exact_mod_cast active
    simp only [lane, colEq, mask, ↓reduceIte, active]

theorem bounds_from_inputs (c : Request) (ob : Nat)
    (nonempty : 0 < c.x.metadata.numel)
    (outputCapacity : ob + 2 * c.x.metadata.numel ≤ 2^64) :
    Bounds (parameters c.x.metadata) c.x.metadata.dataPtr.toNat (scaleView c).base
      ob c.scale.metadata.numel := by
  refine ⟨c.contract.grid, padded_bound _ _ c.contract, c.contract.scales nonempty, ?_, ?_, ?_⟩
  · simpa [parameters, numel_eq _ _ c.contract] using
      Int8.capacity c.x c.xLaws c.contract.contiguous.1 nonempty
  · exact (scaleView c).spanFits
  · simpa [parameters, numel_eq _ _ c.contract] using outputCapacity

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DequantizeRowwise.NativeIndex
