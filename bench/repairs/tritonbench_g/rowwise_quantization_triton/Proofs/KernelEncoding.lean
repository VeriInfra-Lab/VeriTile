import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Encoding

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelEncoding
open VeriTile.Triton

/-- Storing the low byte after signed64 interpretation is independent of the
signed representative. This is an integer identity, never an experiment. -/
theorem narrow64 (z : Int) :
    narrow8 (BitVec.ofInt 64 z).toInt = narrow8 z := by
  have hd : (256 : Int) ∣ (2 ^ 64 : Int) := by norm_num
  have hm : (BitVec.ofInt 64 z).toInt % (2 ^ 64 : Int) = z % (2 ^ 64 : Int) := by
    simpa only [BitVec.toInt_ofInt] using (Int.bmod_emod (x := z) (m := 2 ^ 64))
  have he : (BitVec.ofInt 64 z).toInt % 256 = z % 256 := by
    calc
      _ = ((BitVec.ofInt 64 z).toInt % (2 ^ 64 : Int)) % 256 :=
        (Int.emod_emod_of_dvd _ hd).symm
      _ = (z % (2 ^ 64 : Int)) % 256 := congrArg (fun n : Int => n % 256) hm
      _ = z % 256 := Int.emod_emod_of_dvd _ hd
  simp only [narrow8, he]

/-- A half payload occupies only the low sixteen bits of the structural
floating carrier. The dtype tag selects this interpretation. -/
def packHalf (h : Half) : Word := FP.Value.ofBits .fp32 h.val

def unpackHalf (w : Word) : Half := ⟨w.bits.toNat % 65536, Nat.mod_lt _ (by decide)⟩

theorem half_roundtrip (h : Half) : unpackHalf (packHalf h) = h := by
  apply Fin.ext
  simp only [unpackHalf, packHalf, FP.Value.ofBits, BitVec.toNat_ofNat]
  change h.val % 4294967296 % 65536 = h.val
  have h32 : h.val < 4294967296 := by have := h.isLt; omega
  rw [Nat.mod_eq_of_lt h32, Nat.mod_eq_of_lt h.isLt]

#axiomsClean narrow64
#axiomsClean half_roundtrip
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelEncoding
