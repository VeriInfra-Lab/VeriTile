import VeriTile.Triton.Float.MaximumPrimitive
namespace VeriTile.Triton

/-- Exact WithBot Real compatibility for vector maximum. Floating backends
retain their independent maximum primitive; no comparison/select law is assumed. -/
theorem evalOp_maximumFloat_real_vec (x y : Op .real [n]) (s : BlockState) :
    evalOp (.maximumFloat .real (.consSame .nil) x y) s =
      evalOp (.where (.gt .real (.consSame .nil) x y) x y) s := by
  have h := evalOp_maximumFloat_real_select (Broadcast.consSame Broadcast.nil) x y s
  simp only [FloatDType.toTileDType] at h
  refine h.trans ?_
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy, Tile.remap, Broadcast.leftIndex, Broadcast.rightIndex]

theorem evalOp_max2_vec (x y : Op .real [n]) (s : BlockState) :
    evalOp (.max2 (.consSame .nil) x y) s =
      evalOp (.where (.gt .real (.consSame .nil) x y) x y) s := by
  rw [← evalOp_maximumFloat_real, evalOp_maximumFloat_real_vec]

theorem evalOp_max2_of_where {x y : Op .real [n]} {s : BlockState} {result : Option (Tile .real [n])}
    (h : evalOp (.where (.gt .real (.consSame .nil) x y) x y) s = result) :
    evalOp (.max2 (.consSame .nil) x y) s = result := by
  rw [evalOp_max2_vec]; exact h

theorem evalOp_maximumFloat_real_of_where {x y : Op .real [n]} {s : BlockState} {result : Option (Tile .real [n])}
    (h : evalOp (.where (.gt .real (.consSame .nil) x y) x y) s = result) :
    evalOp (.maximumFloat .real (.consSame .nil) x y) s = result := by
  rw [evalOp_maximumFloat_real_vec]; exact h

#axiomsClean evalOp_maximumFloat_real_vec
#axiomsClean evalOp_max2_vec
#axiomsClean evalOp_max2_of_where
#axiomsClean evalOp_maximumFloat_real_of_where
end VeriTile.Triton
