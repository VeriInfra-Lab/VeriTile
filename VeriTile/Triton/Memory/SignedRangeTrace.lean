import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.TraceProofs
/-- Access safety for an actual signed loop cursor and nonnegative stride.
The invariant is required only at reached iterations. -/
theorem range_int_safe {bounds : RegionBounds} {index : RegName} {stop : Int} {stride : Nat}
    {body : List Stmt} {P : Int → BlockState → Prop}
    (hsafe : ∀ i s, i < stop → P i s →
      Stmt.TraceSafeList bounds body (s.setReg index .int [] (Tile.scalar i)))
    (hnext : ∀ i s t, i < stop → P i s →
      stepStmts body (s.setReg index .int [] (Tile.scalar i)) = some t → P (i + stride) t)
    (cur : Int) (s : BlockState) (hp : P cur s) :
    Stmt.forRangeIntTraceSafe bounds index cur stop stride body s := by
  rw [Stmt.forRangeIntTraceSafe]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · simp only [hc, ↓reduceIte]
      refine ⟨hsafe cur s hc hp, ?_⟩
      cases hb : stepStmts body (s.setReg index .int [] (Tile.scalar cur)) with
      | none => trivial
      | some t => exact range_int_safe hsafe hnext (cur + stride) t (hnext cur s t hc hp hb)
    · simp only [hc, ↓reduceIte]
termination_by (stop - cur).toNat
decreasing_by omega

#axiomsClean range_int_safe
end VeriTile.Triton.TraceProofs
