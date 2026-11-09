import VeriTile.Triton.KernelLemmas.LoopInvariant
import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Meta.StatementAudit

/-! Unit-step signed loops, observed by a natural iteration invariant.
The actual stop remains Int, including negative values; only the number of
reached nonnegative counter values is expressed by stop.toNat. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.SignedLoop
open VeriTile.Triton
noncomputable section

theorem executes (index : RegName) (stop : Int) (body : List Stmt)
    (P : Nat → BlockState → Prop)
    (advance : ∀ c s, c < stop.toNat → P c s →
      ∃ t, stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) = some t ∧ P (c+1) t)
    (c : Nat) (s : BlockState) (hc : c ≤ stop.toNat) (hp : P c s) :
    ∃ t, stepForRangeIntAux index (c : Int) stop 1 body s = some t ∧ P stop.toNat t := by
  by_cases h : c < stop.toNat
  · obtain ⟨t, he, ht⟩ := advance c s h hp
    obtain ⟨u, hu, hpu⟩ := executes index stop body P advance (c+1) t (by omega) ht
    refine ⟨u, ?_, hpu⟩
    rw [stepForRangeIntAux.step_lt (by decide) (by omega), he]
    simpa only [Option.bind_some, Nat.cast_add, Nat.cast_one] using hu
  · refine ⟨s, stepForRangeIntAux.step_ge (by decide) (by omega), ?_⟩
    have : c = stop.toNat := by omega
    simpa only [this] using hp
termination_by stop.toNat - c

theorem safe {bounds : RegionBounds} {index : RegName} {stop : Int}
    {body : List Stmt} {P : Nat → BlockState → Prop}
    (bodySafe : ∀ c s, c < stop.toNat → P c s →
      Stmt.TraceSafeList bounds body (s.setReg index .int [] (Tile.scalar (c : Int))))
    (next : ∀ c s t, c < stop.toNat → P c s →
      stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) = some t → P (c+1) t)
    (c : Nat) (s : BlockState) (hp : P c s) :
    Stmt.forRangeIntTraceSafe bounds index (c : Int) stop 1 body s := by
  rw [Stmt.forRangeIntTraceSafe]
  simp only [one_ne_zero, ↓reduceIte]
  by_cases h : (c : Int) < stop
  · simp only [h, ↓reduceIte]
    have hc : c < stop.toNat := by omega
    refine ⟨bodySafe c s hc hp, ?_⟩
    cases he : stepStmts body (s.setReg index .int [] (Tile.scalar (c : Int))) with
    | none => trivial
    | some t =>
      simpa only [Nat.cast_add, Nat.cast_one] using
        safe bodySafe next (c+1) t (next c s t hc hp he)
  · simp only [h, ↓reduceIte]
termination_by stop.toNat - c

#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.SignedLoop
