import VeriTile.Triton.KernelLemmas.LoopInvariant
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Memory.TraceProofs

namespace VeriTile.Bench.Repairs.DiagSsmTriton.SignedRealLoop
open VeriTile.Triton

/-- Natural iteration counts index a proof invariant; the actual source runner
and its counter register remain signed. -/
theorem invariant (index : RegName) (stop : Nat) (body : List Stmt)
    (P : Nat → BlockState → Prop)
    (stepBody : ∀ n s, n<stop → P n s → ∃ t,
      stepStmts body (s.setReg index .int [] (Tile.scalar (n : Int)))=some t ∧ P (n+1) t)
    (start : Nat) (bound : start≤stop) (s : BlockState) (initial : P start s) :
    ∃ t, stepForRangeIntAux index (start : Int) (stop : Int) 1 body s=some t ∧ P stop t := by
  by_cases active : start<stop
  · have less : (start : Int)<stop := by exact_mod_cast active
    rw [stepForRangeIntAux.step_lt (by decide) less]
    obtain ⟨next,ran,preserved⟩:=stepBody start s active initial
    rw [ran]
    have bound' : start+1≤stop := by omega
    simpa only [Option.bind_some,Nat.cast_add,Nat.cast_one] using
      invariant index stop body P stepBody (start+1) bound' next preserved
  · have same : start=stop := by omega
    subst start
    rw [stepForRangeIntAux.step_ge (by decide) (by omega)]
    exact ⟨s,rfl,initial⟩
termination_by stop-start
decreasing_by omega

/-- Safety follows the actual signed runner and every state it reaches. -/
theorem safe (bounds : RegionBounds) (index : RegName) (stop : Nat) (body : List Stmt)
    (P : BlockState → Prop)
    (bodySafe : ∀ n s, n<stop → P s →
      Stmt.TraceSafeList bounds body (s.setReg index .int [] (Tile.scalar (n : Int))))
    (preserved : ∀ n s t, n<stop → P s →
      stepStmts body (s.setReg index .int [] (Tile.scalar (n : Int)))=some t → P t)
    (start : Nat) (bound : start≤stop) (s : BlockState) (initial : P s) :
    Stmt.forRangeIntTraceSafe bounds index (start : Int) (stop : Int) 1 body s := by
  rw [Stmt.forRangeIntTraceSafe]
  simp only [Nat.one_ne_zero,if_false]
  by_cases active : start<stop
  · have less : (start : Int)<stop := by exact_mod_cast active
    simp only [less,if_true]
    refine ⟨bodySafe start s active initial,?_⟩
    cases ran : stepStmts body (s.setReg index .int [] (Tile.scalar (start : Int))) with
    | none => trivial
    | some next =>
      have nextP := preserved start s next active initial ran
      simpa only [Nat.cast_add,Nat.cast_one] using
        safe bounds index stop body P bodySafe preserved (start+1) (by omega) next nextP
  · have done : ¬(start : Int)<stop := by omega
    simp only [done,if_false]
termination_by stop-start
decreasing_by omega

#axiomsClean safe
#axiomsClean invariant
end VeriTile.Bench.Repairs.DiagSsmTriton.SignedRealLoop
