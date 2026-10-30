import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.IOFrame
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.FiniteResults
import VeriTile.Triton.Concurrent.EventBoundary

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.IOProof
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

theorem implements (b : Backend V) (addzero : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage) :
    (IO.io V h buf.cElements stage).ImplementsOn (ConcurrentSource.backend b)
      (IO.domain b.zero h buf.cElements stage) (IO.mathematical b.zero h buf.cElements stage) := by
  intro input admitted before binding
  let logical := IO.inputs h buf.cElements stage input
  let mem := FiniteBoundary.memory b h logical
  let buf' := FiniteBoundary.activeBuffers h buf
  have published : Wide.PublishedReady b h mem :=
    FiniteBoundary.memory_published b h logical admitted.1
  have completed : ∃ schedule result,KernelProof.Facts b h buf' stage schedule mem result := by
    cases stage with
    | producers => exact KernelProof.producer_complete b addzero h buf' mem published
    | consumers =>
        exact KernelProof.consumer_complete b addzero h buf' mem published
          (FiniteBoundary.memory_ready b h logical (admitted.2 rfl))
  obtain ⟨schedule,result,facts⟩ := completed
  let start := canonical b h logical before
  obtain ⟨events,final,exec,projection,related,done⟩ := ConcurrentSource.facts_run b h buf' stage
    schedule mem result facts start (canonical_memory b h logical before)
  let program : Concurrent.Program := .controller (ConcurrentSource.controller SourceReader.sourcePlan h stage)
  have ran : program.Runs (ConcurrentSource.backend b) start events final.values :=
    ⟨schedule,final,exec,done,rfl⟩
  have safe : ∀ bounds,(IO.io V h buf.cElements stage).InBounds bounds → Concurrent.TapeSafe bounds events := by
    intro bounds allocated
    have sizes := bound_sizes h buf.cElements stage bounds allocated
    exact ConcurrentSource.related_safe h buf' bounds sizes.1 sizes.2.1 sizes.2.2
      events result.1 related facts.bounded
  have output : (IO.io V h buf.cElements stage).OutputBinding
      (IO.mathematical b.zero h buf.cElements stage input) final.values := by
    apply output_binding h buf.cElements stage (Formula.stage b.zero h stage logical) final.values
    · intro i
      change (ConcurrentSource.projectState final).values.mem.partials i.val = _
      rw [projection]
      simpa only [Formula.stage,FiniteBoundary.memory,FiniteBoundary.extend_at,mem] using
        FiniteResults.partials b h buf' stage schedule mem result facts i.val i.isLt
    · intro i
      change (ConcurrentSource.projectState final).values.mem.output i.val = _
      rw [projection]
      simpa only [Formula.stage,FiniteBoundary.memory,FiniteBoundary.extend_at,mem] using
        FiniteResults.output b h buf' stage schedule mem result facts i.val
    · intro i
      change (ConcurrentSource.projectState final).values.mem.locks i.val = _
      rw [projection]
      simpa only [Formula.stage,FiniteBoundary.memory,FiniteBoundary.extend_at,mem] using
        FiniteResults.locks b h buf' stage schedule mem result facts i.val i.isLt
  have minimumSafe := safe (IO.bounds h buf.cElements) (capacity h buf.cElements stage)
  obtain ⟨actual,actualRun,agreement⟩ := Concurrent.Program.runs_transfer
    (ConcurrentSource.backend b) (IO.bounds h buf.cElements) program start before final.values events
    (canonical_agree b h buf.cElements stage input before binding) ran minimumSafe
  refine ⟨program,ConcurrentSource.source_decoded h stage,events,actual,actualRun,safe,?_,?_⟩
  · exact Concurrent.EventProgramIO.OutputBinding.transfer _ (IO.bounds h buf.cElements) _
      final.values actual (capacity h buf.cElements stage) agreement output
  · exact frame (ConcurrentSource.backend b) h buf.cElements stage program before actual events actualRun minimumSafe

#axiomsClean implements
end VeriTile.Bench.Repairs.SpinningLock.IOProof
