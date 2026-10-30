import bench.repairs.tritonbench_g.logsumexp_fwd.IO
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.LogsumexpFwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

theorem initial_eq [Inhabited A] (engine : Engine A) (a : RawKernel.Geometry)
    (scale : A) (j : Nat) (s : State A) :
    (invocation engine a scale j).initial s = RawKernel.programState a scale j s := rfl

theorem run_eq [Inhabited A] (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map (fun j => invocation (engines j) a scale j)) s =
      RawKernel.run engines a scale jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j jobs ih =>
    simp only [List.map_cons, runSchedule, RawKernel.run, Invocation.run, initial_eq]
    change Option.bind _ _ = Option.bind _ _
    congr 1
    funext t
    exact ih t

theorem schedule_safe [Inhabited A] (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) (valid : ∀ j ∈ jobs, j < a.rows * a.blocks)
    (values : Inputs A a) (bounds : RegionName → Nat)
    (hb : (programIO engines a scale jobs).InBounds bounds values) (s : State A) :
    ScheduleSafe bounds (jobs.map (fun j => invocation (engines j) a scale j)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons j jobs ih =>
    refine ⟨?_, ?_⟩
    · change FP.Structural.TraceSafe (engines j).algebra bounds
        (rowBlock "input" "scratch" a.columns a.block a.hasScale)
        ((invocation (engines j) a scale j).initial s)
      rw [initial_eq]
      have hj := valid j (by simp)
      have blocks : 0 < a.blocks := by nlinarith
      have row : j / a.blocks < a.rows := (Nat.div_lt_iff_lt_mul blocks).mpr hj
      apply block_traceSafe _ _ _ _ _ _ scale _ _ a.positive
      · simp [RawKernel.programState]
      · intro i active
        have hcol : j % a.blocks * a.block + i.val < a.columns := by
          simpa [RawKernel.programState, MaxReduction.SourceGrid.positioned] using active
        have address : j / a.blocks * a.columns + (j % a.blocks * a.block + i.val) <
            a.rows * a.columns := by nlinarith
        simpa [RawKernel.programState, MaxReduction.SourceGrid.positioned, Source.inputAddress]
          using hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩ (⟨_, address⟩, PUnit.unit) trivial
      · simpa [RawKernel.programState, MaxReduction.SourceGrid.positioned,
          Source.outputAddress, RawKernel.Geometry.blocks, Nat.div_add_mod, Nat.mul_comm]
          using hb.2.2.1 ⟨0, by simp [programIO, memory, outputs]⟩ (⟨j, hj⟩, PUnit.unit) trivial
    · intro t _
      exact ih (fun j h => valid j (by simp [h])) hb t

/-- The target contains only input values and the selected pure operation
graph. Successful execution, source safety and full frame come from shared IO. -/
theorem implements [Inhabited A] (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows * a.blocks))) :
    (programIO engines a scale jobs).Implements (fun values =>
      result (fun j => RawKernel.compute engines a scale (extend values) j.val)) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    apply schedule_safe engines a scale jobs _ values bounds hb s
    intro j h
    exact List.mem_range.mp (schedule.mem_iff.mp h)
  · intro values s hin
    obtain ⟨t, ran, output, frame⟩ := RawKernel.ordered engines a scale s jobs schedule
    refine ⟨t, ?_, ?_, ?_⟩
    · exact (run_eq engines a scale jobs s).trans ran
    · intro o j _
      fin_cases o
      have hv := output j.1.val j.1.isLt
      have positions : j.1.val / a.blocks < a.rows := by
        have blocks : 0 < a.blocks := by have := j.1.isLt; nlinarith
        exact (Nat.div_lt_iff_lt_mul blocks).mpr j.1.isLt
      have same : RawKernel.compute engines a scale (fun i => (s.mem "input" i).read .real) j.1.val =
          RawKernel.compute engines a scale (extend values) j.1.val := by
        apply block_local
        intro i hi
        have bound : j.1.val / a.blocks * a.columns + i < a.rows * a.columns := by nlinarith
        simpa [programIO, memory, inputs, inputLane, mapValue, extend, value, bound]
          using hin ⟨0, by simp [programIO, memory, inputs]⟩ (⟨_, bound⟩, PUnit.unit) trivial
      simpa [programIO, memory, outputs, outputLane, result, mapCell, mapValue, same]
        using congrArg (mapCell (id : A → A)) hv
    · intro r offset outside
      apply frame
      by_cases hr : r = "scratch"
      · right
        by_contra h
        have bound : offset < a.rows * a.blocks := by omega
        have miss := outside.1 ⟨0, by simp [programIO, memory, outputs]⟩
          (⟨offset, bound⟩, PUnit.unit) trivial
        exact (miss.resolve_left (by simpa [outputs] using not_ne_iff.mpr hr)) rfl
      · exact Or.inl hr

#axiomsClean run_eq
#axiomsClean schedule_safe
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Interface
