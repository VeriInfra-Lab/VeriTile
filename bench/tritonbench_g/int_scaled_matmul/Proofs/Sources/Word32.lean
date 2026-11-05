import VeriTile.Triton.Float.WordIntegerDot
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Word32
open VeriTile.Triton
@[simp] theorem dot_zero {m k n : Nat} (i : TileIndex [m,n]) :
    (FP.IntegerDot.wordDot [] (fun _ : TileIndex [m,k]=>0) (fun _ : TileIndex [k,n]=>0) i).toInt=0 := by
  rcases i with ⟨r,c,⟨⟩⟩
  simp [FP.IntegerDot.wordDot]
/-- PTX's unsigned low-word intermediate and the signed output decode retain
exactly the same 32 bits, including arbitrary carries and negative sums. -/
theorem unsigned_then_signed (x : Int) :
    (BitVec.ofInt 32 (max (x % 4294967296) 0)).toInt = (BitVec.ofInt 32 x).toInt := by
  rw [max_eq_left (Int.emod_nonneg x (by decide))]
  congr 1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofInt]
  norm_num
theorem unsigned_then_bmod (x : Int) :
    Int.bmod (max (x % 4294967296) 0) 4294967296=Int.bmod x 4294967296 := by
  simpa only [BitVec.toInt_ofInt] using unsigned_then_signed x
#axiomsClean unsigned_then_signed
#axiomsClean unsigned_then_bmod
end VeriTile.Bench.Repairs.IntScaledMatmul.Word32
