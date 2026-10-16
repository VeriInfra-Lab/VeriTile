import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStepSafety
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoop

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSafety
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem iteration_safe (p : Parameters) (bounds : RegionBounds)
    (xs : RealMath.Inputs) (pid i : Nat) (s : BlockState)
    (hM : 0<p.M) (hN : 0<p.N) (hk : 0<count p) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : pid<p.SMS) (hi : i<stop p pid)
    (hb : RealAccess.Bounds p bounds) (hinput : RealInput.Represents p xs s)
    (h : RealLoopState.Holds p xs pid i s) :
    Stmt.TraceSafeList bounds (RealProgram.body p) s := by
  have he := RealSchedule.executes p s (tile p pid i) (phase p i) (count p)
    (previousK p i) hGM (tile_bound p pid i hk hs hp hi) h.ki h.fixed.count
    (next_phase p i hk)
    (by intro hz; rw [h.tile,FlatSchedule.tile_restart p.SMS pid (count p) i hk hz]; rfl)
    h.fixed.rows h.fixed.group
  have hparts : RealProgram.body p = RealProgram.schedule p ++ (RealProgram.body p).drop 2 :=
    (List.take_append_drop 2 (RealProgram.body p)).symm
  rw [hparts]
  apply TraceProofs.append_safe _ _ (RealPureSafety.schedule_safe p bounds s)
  intro t ht
  rw [he] at ht
  cases ht
  have hinput' : RealInput.Represents p xs (RealIteration.scheduled p s pid i) := by
    apply RealInput.preserved p xs s _ hinput
    all_goals intro o; rw [RealIteration.scheduled_memory]
  exact RealStepSafety.safe p bounds (RealIteration.scheduled p s pid i) xs pid i hM hN hb
    hinput' (RealIteration.ready p s xs pid i hk h)

theorem loop_safe (p : Parameters) (bounds : RegionBounds) (xs : RealMath.Inputs) (s : BlockState)
    (hM : 0<p.M) (hN : 0<p.N) (hk : 0<count p) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : s.pids 0<p.SMS) (hA : p.C≠p.A) (hB : p.C≠p.B)
    (hb : RealAccess.Bounds p bounds) (hinput : RealInput.Represents p xs s) :
    Stmt.TraceSafe bounds (RealLoopProgram.loop p) (RealPrefix.result p s) := by
  rw [RealLoopProgram.loop,Stmt.TraceSafe]
  simp only [RealLoopProgram.bound_evaluates,evalOp_constNat,Tile.scalar]
  simp only [RealLoopProgram.bound,Op.SafeAt.eq_def,true_and]
  apply TraceProofs.range_safe (P := RealLoop.Holds p xs (s.pids 0) s)
  · intro i t hi h
    exact iteration_safe p bounds xs (s.pids 0) i (t.setReg "_i" .nat [] (Tile.scalar i))
      hM hN hk hGM hs hp hi hb
      (RealInput.preserved p xs t _ h.input (fun _ => rfl) (fun _ => rfl))
      (RealLoopState.set_index p xs (s.pids 0) i t h.registers)
  · intro i t u hi h he
    obtain ⟨v,hv,hnext⟩ := RealLoop.iteration p xs (s.pids 0) s hM hN hk hGM hs hp hA hB i t hi h
    rw [he] at hv
    cases hv
    exact hnext
  · exact RealLoop.initial p xs s hinput

/-- Whole-source trace safety under bounds on original logical A/B/C cells.
The proof follows every reached state of the actual flat persistent range. -/
theorem safe (p : Parameters) (bounds : RegionBounds) (xs : RealMath.Inputs) (s : BlockState)
    (hM : 0<p.M) (hN : 0<p.N) (hK : 0<p.K) (hBK : 0<p.BK) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : s.pids 0<p.SMS) (hA : p.C≠p.A) (hB : p.C≠p.B)
    (hb : RealAccess.Bounds p bounds) (hinput : RealInput.Represents p xs s) :
    (RealSource.kernel p Bool.false).TraceSafe bounds s := by
  have hk : 0<count p := RealMath.blocks_positive p.K p.BK hK hBK
  rw [Kernel.TraceSafe,RealLoopProgram.matches_source]
  apply TraceProofs.append_safe _ _ (RealPureSafety.prefix_safe p bounds s)
  intro t ht
  rw [RealPrefix.executes] at ht
  cases ht
  apply Stmt.TraceSafeList.cons_intro (loop_safe p bounds xs s hM hN hk hGM hs hp hA hB hb hinput)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSafety
