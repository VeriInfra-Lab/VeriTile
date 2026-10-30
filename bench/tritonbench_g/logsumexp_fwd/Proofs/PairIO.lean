import bench.tritonbench_g.logsumexp_fwd.PairIO
import bench.tritonbench_g.logsumexp_fwd.Proofs.PairSource
import bench.tritonbench_g.logsumexp_fwd.Proofs.PairSafety
import bench.tritonbench_g.logsumexp_fwd.Proofs.Interface

namespace VeriTile.Bench.Repairs.LogsumexpFwd.PairIO
open VeriTile.Triton FP.Structural Interface
noncomputable section
set_option maxHeartbeats 3000000
variable {A : Type} [Inhabited A]

theorem invocation_exact (engine : Engine A) (a : RawKernel.Geometry) (scale : A) (j : Nat) (s : State A) :
    (candidateInvocation engine a scale j).run s = (invocation engine a scale j).run s := by
  exact (PairSource.exact engine "input" "scratch" a.columns a.block a.hasScale
    ((invocation engine a scale j).initial s)).symm

theorem schedule_exact (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map (fun j => candidateInvocation (engines j) a scale j)) s =
      runSchedule (jobs.map (fun j => invocation (engines j) a scale j)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j rest ih =>
    simp only [List.map_cons, runSchedule, invocation_exact]
    cases h : (invocation (engines j) a scale j).run s <;> simp [h, ih]

theorem candidate_safe [Inhabited A] (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) (valid : ∀ j ∈ jobs, j < a.rows * a.blocks)
    (values : Inputs A a) (bounds : RegionName → Nat)
    (hb : (programIO engines a scale jobs).InBounds bounds values) (s : State A) :
    ScheduleSafe bounds (jobs.map (fun j => candidateInvocation (engines j) a scale j)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons j jobs ih =>
    refine ⟨?_, ?_⟩
    · change FP.Structural.TraceSafe (engines j).algebra bounds
        (rowBlockCandidate "input" "scratch" a.columns a.block a.hasScale)
        ((candidateInvocation (engines j) a scale j).initial s)
      change FP.Structural.TraceSafe _ _ _ (RawKernel.programState a scale j s)
      have hj := valid j (by simp)
      have blocks : 0 < a.blocks := by nlinarith
      have row : j / a.blocks < a.rows := (Nat.div_lt_iff_lt_mul blocks).mpr hj
      apply candidate_traceSafe _ _ _ _ _ _ scale _ _ a.positive
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

/-- Both full row/block schedules preserve every output and every exterior cell,
under exactly the original all-value input domain and job permutation. -/
theorem equivalent (engines : Nat → Engine A) (a : RawKernel.Geometry)
    (scale : A) (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows * a.blocks))) :
    (programIO engines a scale jobs).Equivalent (variant engines a scale jobs) := by
  intro values _ bounds hb initial binding
  obtain ⟨safe, final, ran, output, frame⟩ :=
    Interface.implements engines a scale jobs schedule values trivial bounds hb initial binding
  refine ⟨safe, ?_, final, final, ran, ?_, ?_, frame, frame⟩
  · apply candidate_safe engines a scale jobs _ values bounds hb initial
    intro j h
    exact List.mem_range.mp (schedule.mem_iff.mp h)
  · exact (schedule_exact engines a scale jobs initial).trans ran
  · intro o j active
    rfl

#axiomsClean invocation_exact
#axiomsClean schedule_exact
#axiomsClean candidate_safe
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.LogsumexpFwd.PairIO
