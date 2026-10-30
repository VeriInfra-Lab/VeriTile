import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Kernels
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Completion
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_SourceCorrect

noncomputable section
open Classical
namespace VeriTile.Bench.TritonBenchG.SpinningLock
open Frontend Kernels

/-- Full interface, source/control connection, values, accesses and frames.
No final value or successful-completion premise occurs in this contract. -/
structure FreshFacts (b : Backend V) (h : HostArgs) (buffers : HostBuffers h)
    (schedule : List Nat) (mem : Memory V) (result : WrapperResult V) : Prop where
  execution : launch b h schedule mem = some result
  trace : result.events = controlTrace originalPlan h.config schedule mem.locks
  returned : result.returned = none
  zeroAcc : ∀ p l, result.final.acc p l = b.zero
  ready : ∀ q : I32, result.final.mem.locks q.toInt = 1 →
    ∀ l, l < h.config.lanes → result.final.mem.partials (h.config.pAddr q l) = b.zero
  output : ∀ a, result.final.mem.output a =
    if Visited result.events a then b.zero else mem.output a
  accesses : ∀ e ∈ result.events, EventInBounds buffers e
  footprint : ∀ a, Visited result.events a → h.outputFootprint a
  completed : allDone h.config result.final = true →
    ∀ a, Visited result.events a ↔ h.outputFootprint a
  frame : ∀ a,
    ((∀ p l, Event.publish p l a ∉ result.events) → result.final.mem.partials a = mem.partials a) ∧
    ((∀ p l, Event.output p l a ∉ result.events) → result.final.mem.output a = mem.output a) ∧
    (Event.release a ∉ result.events → result.final.mem.locks a = mem.locks a)

theorem ready_source_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buffers : HostBuffers h) (schedule : List Nat)
    (mem : Memory V) (ready : PublishedReady b h.config mem) :
    ∃ result, FreshFacts b h buffers schedule mem result := by
  let result : WrapperResult V := ⟨(run b h.config schedule (initial b mem)).1,
    (run b h.config schedule (initial b mem)).2, none⟩
  have hz := run_zeroInvariant b hadd h.config schedule (initial b mem)
    (initial_readyInvariant b h.config mem ready)
  refine ⟨result, ⟨launch_execution b h schedule mem,
    original_control_trace b h.config schedule mem, rfl, hz.acc, hz.ready,
    ?_, ?_, ?_, ?_, ?_⟩⟩
  · exact run_zero_output b hadd h.config schedule (initial b mem)
      (initial_readyInvariant b h.config mem ready)
  · exact run_in_bounds h buffers b schedule (initial b mem) (fun _ => True.intro)
  · intro a ⟨p, l, he⟩
    obtain ⟨hp, hl, hm, ha⟩ := run_output_origin h.config b schedule (initial b mem) p l a he
    exact ⟨p, hp, l, hl, hm, ha⟩
  · exact completed_footprint h b schedule mem
  · exact full_schedule_frame b h.config schedule mem

theorem fresh_source_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buffers : HostBuffers h) (schedule : List Nat)
    (mem : Memory V) (fresh : ∀ a, mem.locks a = 0) :
    ∃ result, FreshFacts b h buffers schedule mem result :=
  ready_source_backend b hadd h buffers schedule mem (fresh_publishedReady b h.config mem fresh)

/-- The actual test changes k and num_sms while retaining P, C and locks.
Readiness transfers because its P tile layout keeps the same block dimensions. -/
theorem relaunch_source_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (oldFacts : FreshFacts b oldHost oldBuffers oldSchedule oldMem oldResult)
    (h : HostArgs) (buffers : HostBuffers h) (schedule : List Nat)
    (hbm : h.bm = oldHost.bm) (hbn : h.bn = oldHost.bn) :
    ∃ result, FreshFacts b h buffers schedule oldResult.final.mem result := by
  apply ready_source_backend b hadd h buffers schedule oldResult.final.mem
  intro q hq l hl
  have hold := oldFacts.ready q hq l
  simpa only [Config.pAddr, Config.lanes, HostArgs.config, hbm, hbn] using
    hold (by simpa only [Config.lanes, HostArgs.config, hbm, hbn] using hl)

theorem fresh_completed_output (facts : FreshFacts b h buffers schedule mem result)
    (hd : allDone h.config result.final = true) (a : Int) :
    result.final.mem.output a = if h.outputFootprint a then b.zero else mem.output a := by
  rw [facts.output a]
  simp only [facts.completed hd a]

theorem fresh_outside_footprint (facts : FreshFacts b h buffers schedule mem result)
    (a : Int) (ha : ¬h.outputFootprint a) : result.final.mem.output a = mem.output a := by
  rw [facts.output a, if_neg (fun hv => ha (facts.footprint a hv))]

#axiomsClean fresh_source_backend
#axiomsClean fresh_completed_output
end VeriTile.Bench.TritonBenchG.SpinningLock
