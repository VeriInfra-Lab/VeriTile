import bench.repairs.tritonbench_g.fused_activation.Proofs.Contract

namespace VeriTile.Bench.Repairs.FusedActivation.Pointers

def biasIndex (weights n pid lane : Nat) : BitVec 32 :=
  BitVec.ofInt 32 ((TritonMul2Finite.offset32 (block n) pid lane).toInt %
    (BitVec.ofNat 32 weights).toInt)

theorem bias_exact (weights n pid lane : Nat) (hw : 0 < weights) (hwb : weights < 2^31)
    (hn : n < 2^31) (hp : pid < (n+block n-1)/block n) (hl : lane < block n) :
    (biasIndex weights n pid lane).toInt = ((pid*block n+lane)%weights : Nat) := by
  have off := (native_indices n pid lane hn hp hl).1
  have bounded := (Nat.mod_lt (pid*block n+lane) hw).trans hwb
  unfold biasIndex
  rw [off,TritonMul2.nat32_signed weights hwb,Int.ofNat_mod_ofNat]
  exact TritonMul2.nat32_signed _ bounded

def pointer (base : Nat) (offset : BitVec 32) : BitVec 64 :=
  BitVec.ofInt 64 ((base : Int)+4*offset.toInt)

theorem pointer_exact (base extent i : Nat) (span : base+4*extent ≤ 2^64)
    (inside : i < extent) (offset : BitVec 32) (index : offset.toInt = (i : Int)) :
    (pointer base offset).toNat = base+4*i ∧ (pointer base offset).toNat+4 ≤ base+4*extent := by
  have value : (pointer base offset).toNat = base+4*i := by
    unfold pointer
    rw [index]
    have cast : (base : Int)+4*(i : Int) = ((base+4*i : Nat) : Int) := by push_cast; ring
    rw [cast]
    change (base+4*i)%2^64 = base+4*i
    exact Nat.mod_eq_of_lt (by omega)
  exact ⟨value,by rw [value]; omega⟩

/-- Natural alignment and distinct live input/output spans are checked by
Python. Valid Tensor storage supplies the capacity bound used above. -/
theorem aligned (base i : Nat) (h : base%4=0) : (base+4*i)%4=0 := by omega

theorem separated_bytes (x nx input ni j k u v : Nat)
    (h : separate x nx input ni) (hj : j < nx) (hk : k < ni) (hu : u < 4) (hv : v < 4) :
    x+4*j+u ≠ input+4*k+v := by rcases h with h|h <;> omega

#axiomsClean bias_exact
#axiomsClean pointer_exact
#axiomsClean aligned
#axiomsClean separated_bytes
end VeriTile.Bench.Repairs.FusedActivation.Pointers
