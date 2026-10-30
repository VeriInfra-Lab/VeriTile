import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Successful executions and safety for the existing increasing signed
range interpreter. Reverse Python loops use the mirrored signed index; these
lemmas impose no precomputed values or intermediate-execution hypotheses. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

theorem executes (index : RegName) (stop : Int) (stride : Nat) (body : List Stmt)
    (P : Int → BlockState → Prop) (hstride : 0 < stride)
    (hstep : ∀ i u, i < stop → P i u →
      ∃ v, stepStmts body (u.setReg index .int [] (Tile.scalar i)) = some v ∧
        P (i + stride) v)
    (cur : Int) (u : BlockState) (hp : P cur u) :
    ∃ final v, stepForRangeIntAux index cur stop stride body u = some v ∧
      stop ≤ final ∧ P final v := by
  by_cases hc : cur < stop
  · obtain ⟨v, hv, hpv⟩ := hstep cur u hc hp
    obtain ⟨final, w, hw, hf, hpw⟩ := executes index stop stride body P hstride hstep
      (cur + stride) v hpv
    refine ⟨final, w, ?_, hf, hpw⟩
    rw [stepForRangeIntAux.step_lt (Nat.ne_of_gt hstride) hc, hv, Option.bind_some]
    exact hw
  · exact ⟨cur, u, stepForRangeIntAux.step_ge (Nat.ne_of_gt hstride) (by omega),
      by omega, hp⟩
termination_by (stop - cur).toNat
decreasing_by
  have hs : (0 : Int) < stride := by exact_mod_cast hstride
  omega

theorem safe (bounds : RegionBounds) (index : RegName) (stop : Int)
    (stride : Nat) (body : List Stmt) (P : Int → BlockState → Prop)
    (hsafe : ∀ i u, i < stop → P i u →
      Stmt.TraceSafeList bounds body (u.setReg index .int [] (Tile.scalar i)))
    (hnext : ∀ i u v, i < stop → P i u →
      stepStmts body (u.setReg index .int [] (Tile.scalar i)) = some v →
        P (i + stride) v)
    (cur : Int) (u : BlockState) (hp : P cur u) :
    Stmt.forRangeIntTraceSafe bounds index cur stop stride body u := by
  rw [Stmt.forRangeIntTraceSafe]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · simp only [hc, ↓reduceIte]
      refine ⟨hsafe cur u hc hp, ?_⟩
      split
      next v hv =>
        exact safe bounds index stop stride body P hsafe hnext (cur + stride) v
          (hnext cur u v hc hp hv)
      next => trivial
    · simp only [hc, ↓reduceIte]
termination_by (stop - cur).toNat
decreasing_by
  have hs : (0 : Int) < stride := by exact_mod_cast (Nat.pos_of_ne_zero hz)
  omega

#axiomsClean executes
#axiomsClean safe
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange
