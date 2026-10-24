import bench.repairs.tritonbench_g.square_matrix.Proofs.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.SquareMatrix.Indices
open VeriTile.Triton

/-- Separate source pointer additions, with signed i32 row multiplication. -/
def pointer (base : BitVec 64) (n r c : Nat) : BitVec 64 :=
  base + (BitVec.ofNat 32 r * BitVec.ofNat 32 n).signExtend 64 * 4 +
    (BitVec.ofNat 32 c).signExtend 64 * 4

def mask (n c : Nat) : Bool := (BitVec.ofNat 32 c).slt (BitVec.ofNat 32 n)

theorem signed32 (n : Nat) (bound : n < 2^31) :
    (BitVec.ofNat 32 n).toInt = (n : Int) := by
  exact BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem sign_extend (n : Nat) (bound : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),signed32 n bound]
  symm
  exact BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem checked_bounds (x : TensorMetadata) (h : InputContract x) (r c : Nat)
    (hr : r < rows x) (hc : c < block (cols x)) (nonempty : 0 < x.numel) :
    r < 2^31 ∧ r*cols x < 2^31 ∧ cols x < 2^31 ∧ c < 2^31 := by
  have size := numel_eq x h
  have ext := h.extent
  have front := h.frontend nonempty
  have cover := block_covers (cols x)
  have positive : 0 < cols x := by nlinarith
  have rowBound := Nat.mul_le_mul_right (cols x) (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at rowBound
  have rr : r ≤ r*cols x := by
    simpa using Nat.mul_le_mul_left r positive
  norm_num at front ext ⊢
  omega

theorem mask_exact (x : TensorMetadata) (h : InputContract x) (r c : Nat)
    (hr : r < rows x) (hc : c < block (cols x)) (nonempty : 0 < x.numel) :
    mask (cols x) c = decide (c < cols x) := by
  have hb := checked_bounds x h r c hr hc nonempty
  rw [mask,BitVec.slt_eq_decide,signed32 c hb.2.2.2,signed32 (cols x) hb.2.2.1]
  simp only [Int.ofNat_lt]

theorem pointer_exact (x : TensorMetadata) (h : InputContract x)
    (base : Nat) (span : base+4*x.numel ≤ 2^64) (r c : Nat)
    (hr : r < rows x) (hc : c < cols x) :
    (pointer (BitVec.ofNat 64 base) (cols x) r c).toNat = base+4*(r*cols x+c) ∧
    base+4*(r*cols x+c)+4 ≤ base+4*x.numel := by
  have active : r*cols x+c < x.numel := by
    rw [numel_eq x h]; exact flat_lt _ _ _ _ hr hc
  have hb := checked_bounds x h r c hr (hc.trans_le (block_covers _)) (by omega)
  have addr : base+4*(r*cols x+c) < 2^64 := by omega
  constructor
  · simp only [pointer,BitVec.ofNat_mul_ofNat,sign_extend _ hb.2.1,sign_extend _ hb.2.2.2]
    change ((BitVec.ofNat 64 base + BitVec.ofNat 64 (r*cols x) * BitVec.ofNat 64 4) +
      BitVec.ofNat 64 c * BitVec.ofNat 64 4).toNat = _
    simp only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
    have eq : base+r*cols x*4+c*4=base+4*(r*cols x+c) := by ring
    rw [eq,Nat.mod_eq_of_lt addr]
  · omega

theorem tensor_pointer_read (x : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws x) (h : InputContract x.metadata)
    (read : Nat → VeriTile.Torch.FP32.Word) (r c : Nat)
    (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    read ((pointer x.dataPtr (cols x.metadata) r c).toNat) =
      VeriTile.Torch.FP32.logicalValue read x laws (r*cols x.metadata+c) := by
  let v := VeriTile.Torch.FP32.dataView x laws h.contiguous
  have hi : r*cols x.metadata+c < x.metadata.numel := by
    rw [numel_eq x.metadata h]; exact flat_lt _ _ _ _ hr hc
  have ptr := VeriTile.Torch.FP32.actual_pointer x laws h.contiguous (by omega)
  have pe := pointer_exact x.metadata h v.base v.spanFits r c hr hc
  rw [←ptr]
  change read ((pointer (BitVec.ofNat 64 v.base) (cols x.metadata) r c).toNat) = _
  rw [pe.1]
  exact VeriTile.Torch.FP32.direct_read_eq_logical read x laws h.contiguous h.noNeg _ hi

/-- Concrete original-source failure: a valid row id times stride two wraps
negative before sign extension, even when its allocation is otherwise large. -/
theorem original_overflow :
    (BitVec.ofNat 32 (2^30) * BitVec.ofNat 32 2).toInt = -(2^31) := by decide

#axiomsClean checked_bounds
#axiomsClean mask_exact
#axiomsClean pointer_exact
#axiomsClean tensor_pointer_read
#axiomsClean original_overflow
end VeriTile.Bench.Repairs.SquareMatrix.Indices
