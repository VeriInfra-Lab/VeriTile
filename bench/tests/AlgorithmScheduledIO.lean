import VeriTile.Triton.Memory.RegionalScheduledKernelIO

namespace VeriTile.Triton.AlgorithmScheduleRegression
noncomputable section

def axis0 (pid : Nat) : AlgorithmLaunch where
  pids := fun _ => pid
  setPids := fun ax => decide (ax = 0)

/-- Installing one axis leaves every other axis and all local/memory state intact. -/
theorem axis0_preserves_other_state (pid : Nat) (s : BlockState) :
    ((axis0 pid).initial s).pids 0 = pid ∧
    (∀ ax, ax ≠ 0 → ((axis0 pid).initial s).pids ax = s.pids ax) ∧
    ((axis0 pid).initial s).numPids = s.numPids ∧
    ((axis0 pid).initial s).regs = s.regs ∧
    ((axis0 pid).initial s).mem = s.mem ∧
    ((axis0 pid).initial s).undef = s.undef := by
  refine ⟨?_, ?_, rfl, rfl, rfl, rfl⟩
  · simp [axis0, AlgorithmLaunch.initial]
  · intro ax other
    simp [axis0, AlgorithmLaunch.initial, other]

def increment : Kernel :=
  ⟨["x"], ["x"],
    [.store .nat [] (.region ⟨"x"⟩ (.constNat 0))
      (.add .nat .nil (.load .nat (.region ⟨"x"⟩ (.constNat 0)) .none .zero)
        (.constNat 1)) .none]⟩

def initial : BlockState :=
  { (default : BlockState) with mem := fun _ _ => MemCell.of .nat 0 }

/-- The second invocation increments the first invocation's actual stored value. -/
theorem second_launch_reads_first_write :
    (runAlgorithmSchedule increment [axis0 0, axis0 1] initial).map
      (fun s => s.mem "x" 0) = some (MemCell.of .nat 2) := by
  simp [runAlgorithmSchedule, AlgorithmLaunch.run, exec, stepStmts, stepStmt,
    evalOp, increment, Tile.bop, Tile.scalar, TileShape.allIndices,
    BlockState.readMemValue, BlockState.readMemTyped, BlockState.writeMemTyped,
    MemCell.of, MemCell.readAs, initial, AlgorithmLaunch.initial, Region.cast,
    NumericDType.add, -Tile.scalar_eta]

def outsideRead : Kernel :=
  ⟨["x"], [], [.assign .nat [] "value"
    (.load .nat (.region ⟨"x"⟩ (.constNat 1)) .none .zero)]⟩

/-- A source load stays unsafe even though regional memory is a total function. -/
theorem active_out_of_bounds_rejected (s : BlockState) :
    ¬ AlgorithmScheduleSafe outsideRead (fun _ => 1) [axis0 0] s := by
  intro safe
  have first := safe.1
  change Kernel.TraceSafeR .triv (fun _ => 1) outsideRead ((axis0 0).initial s) at first
  simp only [Kernel.TraceSafeR, outsideRead, Stmt.TraceSafeListR, Stmt.TraceSafeR,
    Op.SafeAtR.eq_def] at first
  have bound := first.1.2.2
  have impossible := bound (Tile.scalar 1) (by simp [evalOpR.eq_def]) PUnit.unit trivial
  exact (Nat.lt_irrefl 1) impossible

/-- Zero launches preserve the complete input state without any capacity premise. -/
theorem empty_schedule_preserves_state (kernel : Kernel) (bounds : RegionBounds)
    (s : BlockState) :
    runAlgorithmSchedule kernel [] s = some s ∧
    AlgorithmScheduleSafe kernel bounds [] s := ⟨rfl, trivial⟩

end
end VeriTile.Triton.AlgorithmScheduleRegression
