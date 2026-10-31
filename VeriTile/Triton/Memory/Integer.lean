import VeriTile.Triton.Core.Integer
import VeriTile.Triton.Memory.Flatten
import VeriTile.Meta.StatementAudit

/-! Safety and flattening of signed truncation preserve exactly the operand obligations. -/
namespace VeriTile.Triton
set_option maxRecDepth 12000
set_option maxHeartbeats 400000

@[simp 1600] theorem Op.safeAt_truncDiv {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (bounds : RegionBounds) (s : BlockState) :
    (Op.truncDiv bc lhs rhs).SafeAt bounds s ↔ lhs.SafeAt bounds s ∧ rhs.SafeAt bounds s := by
  simp only [Op.truncDiv,Op.SafeAt]
  tauto

@[simp 1600] theorem Op.safeAt_truncMod {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (bounds : RegionBounds) (s : BlockState) :
    (Op.truncMod bc lhs rhs).SafeAt bounds s ↔ lhs.SafeAt bounds s ∧ rhs.SafeAt bounds s := by
  simp only [Op.truncMod,Op.safeAt_truncDiv,Op.SafeAt]
  tauto

@[simp 1600] theorem Op.flattenOk_truncDiv {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) :
    (Op.truncDiv bc lhs rhs).FlattenOk ↔ lhs.FlattenOk ∧ rhs.FlattenOk := by
  simp only [Op.truncDiv,Op.FlattenOk]
  tauto

@[simp 1600] theorem Op.flattenOk_truncMod {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) :
    (Op.truncMod bc lhs rhs).FlattenOk ↔ lhs.FlattenOk ∧ rhs.FlattenOk := by
  simp only [Op.truncMod,Op.flattenOk_truncDiv,Op.FlattenOk]
  tauto

#axiomsClean Op.safeAt_truncDiv
#axiomsClean Op.safeAt_truncMod
#axiomsClean Op.flattenOk_truncDiv
#axiomsClean Op.flattenOk_truncMod
end VeriTile.Triton
