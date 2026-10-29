import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopState
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSchedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealIteration
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def scheduled (p : Parameters) (s : BlockState) (pid i : Nat) :=
  RealSchedule.result p s (tile p pid i) (phase p i)

def result (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat) :=
  RealStep.result p (scheduled p s pid i) xs (row p pid i) (column p pid i) (phase p i)

theorem scheduled_memory (p : Parameters) (s : BlockState) (pid i : Nat) :
    (scheduled p s pid i).mem = s.mem := by
  have h := congrArg BlockState.mem (RealSchedule.frame p s (tile p pid i) (phase p i))
  exact h

structure Ready (p : Parameters) (xs : RealMath.Inputs) (pid i : Nat) (s : BlockState) : Prop where
  fixed : Fixed p s
  ki : s.regs .int [] "ki" = some (Tile.scalar (phase p i : Int))
  tile : s.regs .int [] "tile_id" = some (Tile.scalar (tile p pid i : Int))
  coords : Coordinates p pid i s
  acc : s.regs .real [p.BM,p.BN] "accumulator" = some (accumulator p xs pid i)

theorem scheduled_fixed (p : Parameters) (s : BlockState) (pid i : Nat)
    (h : Fixed p s) : Fixed p (scheduled p s pid i) := by
  constructor <;> by_cases hz : phase p i=0 <;>
    simp [scheduled,RealSchedule.result,hz,RealSchedule.prepared,
      RealOffsets.result,RealCoordinates.result,RealGroup.result,
      h.count,h.rows,h.group,h.lanes]

theorem ready (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat)
    (hk : 0<count p) (h : Holds p xs pid i s) :
    Ready p xs pid i (scheduled p s pid i) := by
  refine ⟨scheduled_fixed p s pid i h.fixed, ?_, ?_, ?_, ?_⟩
  · by_cases hz : phase p i=0 <;>
      simp [scheduled,RealSchedule.result,hz,RealSchedule.prepared,
        RealOffsets.result,RealCoordinates.result,RealGroup.result]
  · by_cases hz : phase p i=0
    · simp [scheduled,RealSchedule.result,hz,RealSchedule.prepared,
        RealOffsets.result,RealCoordinates.result,RealGroup.result]
    · simp only [scheduled,RealSchedule.result,if_neg hz,
        BlockState.setReg_ne_name,ne_eq,String.reduceEq,not_false_eq_true,h.tile]
      exact congrArg (fun x => some (Tile.scalar x)) (FlatSchedule.tile_continue p.SMS pid (count p) i hk hz)
  · by_cases hz : phase p i=0
    · constructor <;>
        simp [scheduled,RealSchedule.result,hz,RealSchedule.prepared,
          RealOffsets.result,RealCoordinates.result,RealGroup.result,
          RealSchedule.row,RealSchedule.column,row,column]
    · exact ⟨by simpa [scheduled,RealSchedule.result,hz] using (h.coords hz).row,
        by simpa [scheduled,RealSchedule.result,hz] using (h.coords hz).column,
        by simpa [scheduled,RealSchedule.result,hz] using (h.coords hz).a,
        by simpa [scheduled,RealSchedule.result,hz] using (h.coords hz).b⟩
  · by_cases hz : phase p i=0 <;>
      simpa [scheduled,RealSchedule.result,hz,RealSchedule.prepared,
        RealOffsets.result,RealCoordinates.result,RealGroup.result] using h.acc

/-- One complete actual source iteration, including signed control, both
scheduling branches, clamped masked loads, fused dot, masked store and reset. -/
theorem executes (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat)
    (hM : 0<p.M) (hN : 0<p.N) (hk : 0<count p) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : pid<p.SMS) (hi : i<stop p pid)
    (hinput : RealInput.Represents p xs s) (h : Holds p xs pid i s) :
    stepStmts (RealProgram.body p) s = some (result p s xs pid i) := by
  have hschedule := RealSchedule.executes p s (tile p pid i) (phase p i) (count p)
    (previousK p i) hGM (tile_bound p pid i hk hs hp hi) h.ki h.fixed.count
    (next_phase p i hk)
    (by intro hz; rw [h.tile,FlatSchedule.tile_restart p.SMS pid (count p) i hk hz]; rfl)
    h.fixed.rows h.fixed.group
  have hr := ready p s xs pid i hk h
  have hinputs : RealInput.Represents p xs (scheduled p s pid i) := by
    apply RealInput.preserved p xs s _ hinput
    all_goals intro o; rw [scheduled_memory]
  have hstep := RealStep.executes p (scheduled p s pid i) xs (row p pid i) (column p pid i)
    (phase p i) hM hN hinputs hr.ki hr.fixed.count hr.coords.row hr.coords.column
    hr.coords.a hr.coords.b hr.fixed.lanes hr.acc
  have hparts : RealProgram.body p = RealProgram.schedule p ++ (RealProgram.body p).drop 2 := by
    exact (List.take_append_drop 2 (RealProgram.body p)).symm
  rw [hparts,stepStmts.append_some hschedule]
  exact hstep

/-- Full raw-cell frame of the complete actual iteration; no assumption is
made about the initial contents or tags of cells outside active output lanes. -/
theorem frame (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat)
    (r : RegionName) (o : Nat)
    (h : ∀ j, RealStore.active p (row p pid i) (column p pid i) j →
      ¬(p.C=r ∧ RealStore.offset p (row p pid i) (column p pid i) j=o)) :
    (result p s xs pid i).mem r o = s.mem r o := by
  rw [result,RealStep.frame p _ xs _ _ _ r o h]
  rw [scheduled_memory]

theorem input_preserved (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat)
    (hA : p.C≠p.A) (hB : p.C≠p.B) (h : RealInput.Represents p xs s) :
    RealInput.Represents p xs (result p s xs pid i) := by
  apply RealInput.preserved p xs s _ h
  · intro o; exact frame p s xs pid i p.A o (by intros j hj hn; exact hA hn.1)
  · intro o; exact frame p s xs pid i p.B o (by intros j hj hn; exact hB hn.1)

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealIteration
