import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Core.SignedCast
import VeriTile.Meta.StatementAudit

/-! Precision erasure commutes with the derived integer constructors. Keeping
these equalities separate avoids repeatedly expanding the signed division and
narrowing trees inside complete-source decomposition proofs. -/
namespace VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000

@[simp] theorem Op.eraseDType_truncDiv {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .int a) (y : Op .int b) :
    (Op.truncDiv bc x y).eraseDType = Op.truncDiv bc x.eraseDType y.eraseDType := by
  simp only [Op.truncDiv, Op.eraseDType, eraseDType,
    NumericDType.eraseDType, IntegralDType.eraseDType, ComparableDType.eraseDType]
  rfl

@[simp] theorem Op.eraseDType_truncMod {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .int a) (y : Op .int b) :
    (Op.truncMod bc x y).eraseDType = Op.truncMod bc x.eraseDType y.eraseDType := by
  simp only [Op.truncMod, Op.eraseDType, eraseDType,
    NumericDType.eraseDType, IntegralDType.eraseDType, ComparableDType.eraseDType]
  rfl

@[simp] theorem Op.eraseDType_intToSigned (width : Nat) (x : Op .int shape) :
    (Op.intToSigned width x).eraseDType = Op.intToSigned width x.eraseDType := by
  simp only [Op.intToSigned, Op.unsignedToInt, Op.natToUnsigned, Op.intToUnsigned,
    Op.eraseDType, eraseDType,
    NumericDType.eraseDType, IntegralDType.eraseDType, ComparableDType.eraseDType]
  rfl

#axiomsClean Op.eraseDType_truncDiv
#axiomsClean Op.eraseDType_truncMod
#axiomsClean Op.eraseDType_intToSigned
end VeriTile.Triton
