import bench.tritonbench_g.spinning_lock_reduction.Proofs.ConcurrentRun
import bench.tritonbench_g.spinning_lock_reduction.Proofs.KernelStages
import VeriTile.Triton.Concurrent.EventProgramIO

namespace VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton SourceReader

def decode (h : HostArgs) (stage : Phased.Stage) (source : List Statement) : Option Concurrent.Program := do
  let plan ← compile source
  if evalB h stage 0 0 0 plan.assertion then some (.controller (controller plan h stage)) else none

theorem source_decoded (h : HostArgs) (stage : Phased.Stage) :
    decode h stage kernel = some (.controller (controller sourcePlan h stage)) := by
  simp [decode,source_compiles]

theorem initial_project (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (s : Concurrent.State V) :
    projectState ((controller plan h stage).initial (Concurrent.Program.initialValues (backend b) s)) =
      initialPlan b plan (values s).mem := rfl

theorem initial_valid (b : Backend V) (plan : Plan) (h : HostArgs) (stage : Phased.Stage)
    (s : Concurrent.State V) :
    Valid ((controller plan h stage).initial (Concurrent.Program.initialValues (backend b) s)) := by
  intro p; exact Nat.zero_lt_succ 8

theorem phase_done (label next lane : Nat) : phase label next lane = .done ↔ label = 7 := by
  unfold phase
  split <;> simp_all

theorem done_of_source (h : HostArgs) (buf : Wide.Buffers h) (stage : Phased.Stage)
    (s : Concurrent.ControlState V) (done : Phased.StageDone h stage (projectState s)) :
    (controller sourcePlan h stage).Done s := by
  intro p hp selected
  have selected' : Phased.selected h stage p = true := by
    simpa [controller,Concurrent.Test.eval,lowerB_eval,Concurrent.NumExpr.eval,
      source_skip,Phased.selected64_exact h buf stage p hp] using selected
  have hd := done p hp selected'
  have label : s.pc p = 7 := (phase_done _ _ _).mp hd
  simpa [controller,label]

/-- Every shared step has the original source event at the same position and
the same complete memory/control projection. This includes all failed CAS
retries and explicit barriers; no completed-CTA serialization is introduced. -/
theorem facts_run (b : Backend V) (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) (schedule : List Nat) (mem : Memory V)
    (result : List SourceEvent × Wide.State V)
    (facts : KernelProof.Facts b h buf stage schedule mem result)
    (s : Concurrent.State V) (binding : (values s).mem = mem) :
    ∃ trace final,
      (controller sourcePlan h stage).run (backend b) schedule
        ((controller sourcePlan h stage).initial (Concurrent.Program.initialValues (backend b) s)) =
          some (trace,final) ∧
      projectState final = result.2 ∧ EventsRelated trace result.1 ∧
      (controller sourcePlan h stage).Done final := by
  obtain ⟨trace,final,exec,projection,related,_⟩ := run_projection b sourcePlan h stage schedule
    ((controller sourcePlan h stage).initial (Concurrent.Program.initialValues (backend b) s))
    (initial_valid b sourcePlan h stage s)
  have original : runPlan b sourcePlan h stage schedule (initialPlan b sourcePlan mem) = result := by
    have he := facts.execution
    simpa [runKernel,compileAndRun,source_compiles,source_assertion] using he
  rw [initial_project,binding,original] at projection related
  exact ⟨trace,final,exec,projection,related,done_of_source h buf stage final (projection ▸ facts.done)⟩

theorem event_safe (h : HostArgs) (buf : Wide.Buffers h) (bounds : Concurrent.Bounds)
    (pBound : buf.pElements ≤ bounds.mem 0) (cBound : buf.cElements ≤ bounds.mem 1)
    (lBound : buf.lockElements ≤ bounds.words 0) (p : Nat) (e : SourceEvent)
    (safe : Wide.EventInBounds buf (project e)) : (event p e).Safe bounds := by
  cases e with
  | memory e =>
      cases e <;> simp_all [event,project,Wide.EventInBounds,Concurrent.Event.Safe,
        Concurrent.Expr.Safe,Concurrent.InWindow] <;> omega
  | cas a compare replace | exchange a value =>
      simp_all [event,project,Wide.EventInBounds,Concurrent.Event.Safe,Concurrent.InWindow]
      omega
  | barrier => trivial

theorem related_safe (h : HostArgs) (buf : Wide.Buffers h) (bounds : Concurrent.Bounds)
    (pBound : buf.pElements ≤ bounds.mem 0) (cBound : buf.cElements ≤ bounds.mem 1)
    (lBound : buf.lockElements ≤ bounds.words 0)
    (trace : List Concurrent.Event) (source : List SourceEvent) (related : EventsRelated trace source)
    (safe : ∀ e ∈ source.map project,Wide.EventInBounds buf e) : Concurrent.TapeSafe bounds trace := by
  induction related with
  | nil => simp [Concurrent.TapeSafe]
  | @cons a e ir src head tail ih =>
      rcases head with ⟨p,rfl⟩
      intro current member
      rcases List.mem_cons.mp member with rfl | rest
      · exact event_safe h buf bounds pBound cBound lBound p e (safe _ (by simp))
      · exact ih (fun e he => safe e (by simp [he])) current rest

end VeriTile.Bench.Repairs.SpinningLock.ConcurrentSource
