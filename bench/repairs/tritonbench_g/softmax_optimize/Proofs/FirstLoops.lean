import bench.repairs.tritonbench_g.softmax_optimize.Proofs.FirstPass

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.FirstLoops
open VeriTile.Triton FirstPass
set_option maxHeartbeats 4000000
noncomputable section

theorem full_from (X : RegionName) (N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (initial : BlockState) (remaining current : Nat) (state : BlockState)
    (hc : current + remaining = Config.fullTiles N B)
    (h : Invariant X N B initial current state) :
    ∃ finalState,
      stepForRangeIntAux "start_n" ((current * B : Nat) : Int)
        ((Config.fullTiles N B * B : Nat) : Int) B (Source.inputFull X N B) state = some finalState ∧
      Invariant X N B initial (Config.fullTiles N B) finalState := by
  induction remaining generalizing current state with
  | zero =>
    have he : current = Config.fullTiles N B := by omega
    subst current
    rw [stepForRangeIntAux.step_ge (by omega) (le_refl _)]
    exact ⟨state, rfl, h⟩
  | succ remaining ih =>
    have hcurrent : current < Config.fullTiles N B := by omega
    have hlt : ((current * B : Nat) : Int) < ((Config.fullTiles N B * B : Nat) : Int) := by
      exact_mod_cast Nat.mul_lt_mul_of_pos_right hcurrent hB
    obtain ⟨next, hnext, hinv⟩ := full_step X h hN hcurrent
    rw [stepForRangeIntAux.step_lt (by omega) hlt, hnext]
    simp only [Option.bind_some]
    rw [show ((current * B : Nat) : Int) + (B : Int) = (((current + 1) * B : Nat) : Int) by
      push_cast; ring]
    exact ih (current + 1) next (by omega) hinv

def fullStatement (X : RegionName) (N B : Nat) : Stmt :=
  .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
    (Source.inputFull X N B)

def tailStatement (X : RegionName) (N B : Nat) : Stmt :=
  .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "N") (.constNat B)
    (Source.inputTail X N B)

theorem full_run (X : RegionName) (N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (initial state : BlockState) (h : Invariant X N B initial 0 state) :
    ∃ finalState, stepStmt (fullStatement X N B) state = some finalState ∧
      Invariant X N B initial (Config.fullTiles N B) finalState := by
  simp only [fullStatement, stepStmt, evalOp.eq_def, h.previousReg,
    Option.bind_eq_bind, Option.bind_some, Tile.scalar_data]
  rw [Config.previous_eq_full_tiles hN hB]
  simpa only [Nat.zero_add, Nat.zero_mul, Int.natCast_zero] using
    full_from X N B hN hB initial (Config.fullTiles N B) 0 state (by omega) h

theorem tail_run (X : RegionName) (N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (initial state : BlockState) (h : Invariant X N B initial (Config.fullTiles N B) state) :
    ∃ finalState, stepStmt (tailStatement X N B) state = some finalState ∧
      Invariant X N B initial (Config.fullTiles N B + 1) finalState := by
  simp only [tailStatement, stepStmt, evalOp.eq_def, h.previousReg, h.colsReg,
    Option.bind_eq_bind, Option.bind_some, Tile.scalar_data]
  rw [Config.previous_eq_full_tiles hN hB]
  have hlt : ((Config.fullTiles N B * B : Nat) : Int) < (N : Int) := by
    exact_mod_cast Config.tail_start_lt (B := B) hN
  rw [stepForRangeIntAux.step_lt (by omega) hlt]
  obtain ⟨next, hnext, hinv⟩ := tail_step X h
  rw [hnext]
  simp only [Option.bind_some]
  have hstop : (N : Int) ≤ ((Config.fullTiles N B * B : Nat) : Int) + (B : Int) := by
    have hc := Config.tail_covers hN hB
    push_cast
    exact_mod_cast (show N ≤ Config.fullTiles N B * B + B by nlinarith)
  rw [stepForRangeIntAux.step_ge (by omega) hstop]
  exact ⟨next, rfl, hinv⟩

/-- Both actual signed input loops execute successfully and preserve all
memory and launch metadata; their tile registers equal the proved recurrence. -/
theorem inputs_run (X : RegionName) (N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (initial state : BlockState) (h : Invariant X N B initial 0 state) :
    ∃ finalState,
      stepStmts [fullStatement X N B, tailStatement X N B] state = some finalState ∧
      Invariant X N B initial (Config.fullTiles N B + 1) finalState := by
  obtain ⟨middle, hm, hi⟩ := full_run X N B hN hB initial state h
  obtain ⟨finalState, hf, hfinal⟩ := tail_run X N B hN hB initial middle hi
  refine ⟨finalState, ?_, hfinal⟩
  rw [stepStmts.cons_some hm, stepStmts.cons_some hf, stepStmts.nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.FirstLoops
