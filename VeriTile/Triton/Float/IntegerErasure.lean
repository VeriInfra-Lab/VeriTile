import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Core.Integer
import VeriTile.Meta.StatementAudit
/-! Signed runtime quotient/remainder keep their complete control graph under
floating-dtype erasure. These structural equations avoid expanding the signed
branch graph repeatedly in source-correspondence proofs. -/
namespace VeriTile.Triton
set_option maxRecDepth 12000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem Op.eraseDType_truncDiv {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) :
    (Op.truncDiv bc lhs rhs).eraseDType=Op.truncDiv bc lhs.eraseDType rhs.eraseDType := by
  simp only [Op.truncDiv,Op.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,
    ComparableDType.eraseDType,VeriTile.Triton.eraseDType]

theorem Op.eraseDType_truncMod {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) :
    (Op.truncMod bc lhs rhs).eraseDType=Op.truncMod bc lhs.eraseDType rhs.eraseDType := by
  simp only [Op.truncMod,Op.eraseDType_truncDiv,Op.eraseDType,NumericDType.eraseDType,
    VeriTile.Triton.eraseDType]

#axiomsClean Op.eraseDType_truncDiv
#axiomsClean Op.eraseDType_truncMod
end VeriTile.Triton
