import bench.tritonbench_g.attn_fwd_triton.Proofs.RegisterViews
import VeriTile.Triton.Memory.TraceProofs

/-! Memory-access trace safety is preserved by the same bijective register
views used for execution, including failure and signed-range behavior. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.RegisterSafetyViews
open VeriTile.Triton
open RegisterViews
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

def Corresponds (names : RegName ≃ RegName) (view actual : List Stmt) : Prop :=
  ∀ bounds s, Stmt.TraceSafeList bounds actual s ↔ Stmt.TraceSafeList bounds view (pull names s)

theorem nil (names : RegName ≃ RegName) : Corresponds names [] [] := by
  intro bounds s
  simp only [Stmt.TraceSafeList]

theorem assign (names : RegName ≃ RegName) (name : RegName)
    {d : TileDType} {sh : TileShape} (view actual : Op d sh)
    (expression : ∀ s,evalOp actual s=evalOp view (pull names s))
    (access : ∀ bounds s,actual.SafeAt bounds s ↔ view.SafeAt bounds (pull names s))
    (before after : List Stmt) (rest : Corresponds names before after) :
    Corresponds names (.assign d sh name view::before)
      (.assign d sh (names name) actual::after) := by
  intro bounds s
  rw [Stmt.TraceSafeList,Stmt.TraceSafeList]
  simp only [Stmt.TraceSafe]
  simp only [stepStmt]
  rw [←expression,←access]
  cases h : evalOp actual s with
  | none=>simp [stepStmt,h,expression,bind,Option.bind]
  | some value=>
    simpa only [stepStmt,←expression,h,bind,Option.bind,pull_setReg] using
      and_congr_right (fun _=>rest bounds (s.setReg (names name) d sh value))

theorem signed_range (names : RegName ≃ RegName) (index : RegName)
    (view actual : List Stmt) (body : RegisterViews.Corresponds names view actual)
    (safe : Corresponds names view actual) (bounds : RegionBounds)
    (cur stop : Int) (stride : Nat) (s : BlockState) :
    Stmt.forRangeIntTraceSafe bounds (names index) cur stop stride actual s ↔
      Stmt.forRangeIntTraceSafe bounds index cur stop stride view (pull names s) := by
  generalize hm : (stop-cur).toNat=remaining
  induction remaining using Nat.strong_induction_on generalizing cur s with
  | h remaining ih=>
    rw [Stmt.forRangeIntTraceSafe,Stmt.forRangeIntTraceSafe]
    by_cases hz : stride=0
    · simp [hz]
    by_cases hc : cur<stop
    · simp only [hz,hc,↓reduceIte]
      have first := safe bounds (s.setReg (names index) .int [] (Tile.scalar cur))
      rw [pull_setReg] at first
      apply and_congr first
      have matched := body (s.setReg (names index) .int [] (Tile.scalar cur))
      rw [pull_setReg] at matched
      cases ran : stepStmts actual (s.setReg (names index) .int [] (Tile.scalar cur)) with
      | none=>
        simp only [ran,Option.map_none] at matched
        simp only [ran,←matched]
      | some next=>
        simp only [ran,Option.map_some] at matched
        simp only [ran,←matched]
        exact ih (stop-(cur+stride)).toNat (by omega) (cur+stride) next rfl
    · simp [hz,hc]

#axiomsClean assign
#axiomsClean signed_range
end
end VeriTile.Bench.Repairs.AttnFwdTriton.RegisterSafetyViews
