import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Frames

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 5000000

theorem launch_frame {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M) (lib : Library ops neg rt)
    (q : Prepared) (valid : Valid q.layout) (eps : γ) (s : World M)
    (old : List Allocation) (live : ∀ allocation∈old,allocation∈s.allocations)
    (fresh : ∀ allocation∈old,q.output.view.storage.separate allocation) :
    Frame ops old s (launch ops a rt q eps s).state ∧
      (∀ old∈s.allocations,old∈(launch ops a rt q eps s).state.allocations) := by
  by_cases empty : q.layout.rows=0
  · simp only [launch,empty,if_true,Result.state]
    exact ⟨Frame.refl ops old s,fun _ h => h⟩
  · cases hs : rt.setup q.layout s with
    | error reason after =>
      have keep := lib.setup q.layout s
      rw [hs] at keep
      simp only [Result.state] at keep
      exact ⟨by simpa [launch,empty,hs,Result.bind,Result.state] using keep.frame ops old live,
        by simpa [launch,empty,hs,Result.bind,Result.state] using keep.live⟩
    | ok plan ready =>
      have keep := lib.setup q.layout s
      rw [hs] at keep
      simp only [Result.state] at keep
      have setupFrame := keep.frame ops old live
      cases plan with
      | complete schedule allRows =>
        have bounded : ∀ row∈schedule,row<q.layout.rows :=
          fun row hr => List.mem_range.mp (allRows.mem_iff.mp hr)
        exact ⟨by simpa [launch,empty,hs,Result.bind,Result.state] using
          Frame.trans ops setupFrame (private_run_frame ops (a.compute q.layout eps) q.layout valid old fresh schedule bounded ready),
          by simpa [launch,empty,hs,Result.bind,Result.state] using keep.live⟩
      | interrupted completed storesDone bounded partialBound reason =>
        exact ⟨by simpa [launch,empty,hs,Result.bind,Result.state] using
          Frame.trans ops setupFrame (private_interrupted_frame ops (a.compute q.layout eps)
            q.layout valid old fresh completed bounded storesDone partialBound ready),
          by simpa [launch,empty,hs,Result.bind,Result.state] using keep.live⟩

theorem launch_values {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (neg : Negation α) (a : Source.Adapter α γ) (rt : Runtime M) (lib : Library ops neg rt)
    (q : Prepared) (valid : Valid q.layout) (eps : γ) (s after : World M)
    (inputLive : q.snapshot.view.storage∈s.allocations)
    (success : launch ops a rt q eps s=.ok () after) :
    ∀ row col,row<q.snapshot.rows → col<q.snapshot.cols →
      ops.read after.memory (q.layout.ykey row col)=
        a.compute q.layout eps (rowInputs ops q.layout s.memory row) col := by
  intro row col hr hc
  by_cases empty : q.layout.rows=0
  · change row<q.layout.rows at hr
    omega
  · cases hs : rt.setup q.layout s with
    | error reason final => simp [launch,empty,hs,Result.bind] at success
    | ok plan ready =>
      have keep := lib.setup q.layout s
      rw [hs] at keep
      simp only [Result.state] at keep
      cases plan with
      | interrupted completed storesDone bounded partialBound reason =>
        simp [launch,empty,hs,Result.bind] at success
      | complete schedule allRows =>
        simp only [launch,empty,if_false,hs,Result.bind,Result.ok.injEq] at success
        obtain ⟨_,rfl⟩ := success
        have bounded : ∀ row∈schedule,row<q.layout.rows :=
          fun row hr => List.mem_range.mp (allRows.mem_iff.mp hr)
        rw [run_values ops (a.compute q.layout eps) q.layout valid schedule bounded
          (allRows.nodup_iff.mpr List.nodup_range) row col
          (allRows.mem_iff.mpr (List.mem_range.mpr hr)) hc]
        rw [keep.rowInputs ops q.layout valid inputLive row hr]

/-- Successful and interrupted setup schedules receive complete source
certificates at every actual prefix heap. No execution law is a numeric port. -/
theorem launch_source_certificates {M α β γ : Type} [Inhabited α] [Inhabited γ]
    (ops : Ops M α β) (a : Source.Adapter α γ) (q : Prepared) (valid : Valid q.layout)
    (eps : γ) (schedule : List Nat) (bounded : ∀ row∈schedule,row<q.layout.rows) (s : World M) :
    Source.Certificates ops a q.layout eps schedule s.memory :=
  Source.certificates ops a q.layout valid eps schedule bounded s.memory

#axiomsClean launch_frame
#axiomsClean launch_values
#axiomsClean launch_source_certificates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
