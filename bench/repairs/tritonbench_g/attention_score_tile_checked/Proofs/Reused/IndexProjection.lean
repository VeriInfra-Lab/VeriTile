import VeriTile.Triton.Semantics.Integer
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.IndexProjection
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

/-- The explicit source int32 block-offset cast is exact within the original
wrapper's padded-key bound. The theorem does not discard the source cast. -/
theorem key_offset (s : BlockState) (value : Op .int []) (n : Nat)
    (evaluates : evalOp value s = some (Tile.scalar (n : Int)))
    (bound : n < 2147483648) :
    evalOp (.castIntToNat (Op.intToSigned 32 value)) s = some (Tile.scalar n) := by
  have exactCast : (BitVec.ofInt 32 (n : Int)).toInt = (n : Int) :=
    BitVec.toInt_ofInt_eq_self (by decide) (by norm_num) (by norm_num; omega)
  rw [evalOp, eval_intToSigned, evaluates]
  simp only [bind, Option.bind_some, pure]
  apply congrArg some
  apply Tile.ext
  intro i
  change (BitVec.ofInt 32 (n : Int)).toInt.toNat = n
  rw [exactCast]
  rfl

#axiomsClean key_offset
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.IndexProjection
