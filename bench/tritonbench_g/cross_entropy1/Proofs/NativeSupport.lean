import bench.tritonbench_g.cross_entropy1.Proofs.TensorModel
namespace VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native
open VeriTile.Triton
theorem nat64_signed (n : Nat) (h : n < 2^63) :
    (BitVec.ofNat 64 n).toInt = (n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem widen32_exact (n : Nat) (h : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have h32 : (BitVec.ofNat 32 n).toInt = (n : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int))
      (by decide) (by omega) (by norm_num; omega)
  rw [h32, nat64_signed n (by omega)]


structure Buffer where
  base : Nat
  capacity : Nat
  device : Nat

def Buffer.Valid (b : Buffer) (n : Nat) : Prop :=
  4*n ≤ b.capacity ∧ b.base+b.capacity ≤ 2^64 ∧ b.base % 4 = 0

def Buffer.Disjoint (a b : Buffer) : Prop :=
  a.base+a.capacity ≤ b.base ∨ b.base+b.capacity ≤ a.base


end VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native
