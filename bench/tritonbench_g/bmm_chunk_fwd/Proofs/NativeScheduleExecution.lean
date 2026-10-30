import bench.tritonbench_g.bmm_chunk_fwd.NativeIO
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.KernelExecutionRaw

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

@[simp] theorem main_initial (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (version : Bool)
    (pid : Host.PID (parameters p) c) (s : State Expr) :
    (mainInvocation rest p c version pid).initial s=FPGrid.atIndex s pid := rfl

@[simp] theorem cleanup_initial (rest : Algebra Expr) (p : Parameters)
    (pid : Cleanup.PID (parameters p)) (s : State Expr) :
    (cleanupInvocation rest p pid).initial s=CleanupSource.atIndex (parameters p) s pid := rfl

theorem main_run (rest : Algebra Expr) (p : Parameters) (c : Host.Config)
    (order : List (Host.PID (parameters p) c)) (s : State Expr) :
    runSchedule (order.map (mainInvocation rest p c Bool.false)) s=
      FPGrid.schedule (backend rest c) (parameters p) c order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,FPGrid.schedule,Invocation.run,main_initial,
      mainInvocation,Bool.false_eq_true,↓reduceIte,Kernels.fp32Source_eq (parameters p) c rfl rfl]
    congr 1
    funext next
    exact ih next

theorem candidate_run (rest : Algebra Expr) (p : Parameters) (c : Host.Config)
    (order : List (Host.PID (parameters p) c)) (s : State Expr) :
    runSchedule (order.map (mainInvocation rest p c Bool.true)) s=
      KernelCandidate.schedule (backend rest c) (parameters p) c order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,KernelCandidate.schedule,Invocation.run,main_initial,
      mainInvocation,↓reduceIte]
    congr 1
    funext next
    exact ih next

theorem cleanup_run (rest : Algebra Expr) (p : Parameters)
    (order : List (Cleanup.PID (parameters p))) (s : State Expr) :
    runSchedule (order.map (cleanupInvocation rest p)) s=
      CleanupSource.schedule rest (.leaf .fp32 0) (parameters p) order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,CleanupSource.schedule,Invocation.run,cleanup_initial,cleanupInvocation]
    congr 1
    funext next
    exact ih next

theorem source_run (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (s : State Expr) :
    runSchedule (schedule rest p c Bool.false) s=KernelExecution.Raw.source rest (parameters p) c s := by
  by_cases empty : p.shape.outputSize=0
  · simp only [schedule,KernelExecution.Raw.source,show (parameters p).shape=p.shape from rfl,if_pos empty,runSchedule]
  · simp only [schedule,KernelExecution.Raw.source,show (parameters p).shape=p.shape from rfl,if_neg empty,
      runSchedule_append,main_run]
    cases causal : p.causal
    · simp only [show (parameters p).causal=p.causal from rfl,causal,Bool.false_eq_true,↓reduceIte,runSchedule]
      rfl
    · simp only [show (parameters p).causal=p.causal from rfl,causal,↓reduceIte,cleanup_run]
      rfl

theorem candidate_source_run (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (s : State Expr) :
    runSchedule (schedule rest p c Bool.true) s=KernelExecution.Raw.candidate rest (parameters p) c s := by
  by_cases empty : p.shape.outputSize=0
  · simp only [schedule,KernelExecution.Raw.candidate,show (parameters p).shape=p.shape from rfl,if_pos empty,runSchedule]
  · simp only [schedule,KernelExecution.Raw.candidate,show (parameters p).shape=p.shape from rfl,if_neg empty,
      runSchedule_append,candidate_run]
    cases causal : p.causal
    · simp only [show (parameters p).causal=p.causal from rfl,causal,Bool.false_eq_true,↓reduceIte,runSchedule]
      rfl
    · simp only [show (parameters p).causal=p.causal from rfl,causal,↓reduceIte,cleanup_run]
      rfl

/-- Total native execution and its full typed memory reference, reusing the
original source proofs without requiring a new input materialization premise.
The old stronger materialization-invariant theorem is retained unchanged. -/
theorem successful (rest : Algebra Expr) (p : Parameters) (c : Host.Config) (s : State Expr) :
    ∃ final, runSchedule (schedule rest p c Bool.false) s=some final ∧
      runSchedule (schedule rest p c Bool.true) s=some final ∧
      final.mem=NativeReference.kernelMemory (parameters p) c s.mem := by
  have same := KernelExecution.Raw.equivalent rest (parameters p) c rfl rfl s
  rw [source_run,candidate_source_run,←same]
  suffices ∃ final, KernelExecution.Raw.source rest (parameters p) c s=some final ∧
      final.mem=NativeReference.kernelMemory (parameters p) c s.mem by
    obtain ⟨final,run,mem⟩ := this
    exact ⟨final,run,run,mem⟩
  by_cases empty : p.shape.outputSize=0
  · exact ⟨s,by simp [KernelExecution.Raw.source,parameters,empty],by simp [NativeReference.kernelMemory,parameters,empty]⟩
  · obtain ⟨main,runMain,memMain⟩ := NativeReference.grid_reference rest (parameters p) c rfl rfl
      (FPGrid.allPrograms (parameters p) c) s
    cases causal : p.causal
    · refine ⟨main,?_,?_⟩
      · simp only [KernelExecution.Raw.source,show (parameters p).shape=p.shape from rfl,
          if_neg empty,show (parameters p).causal=p.causal from rfl,causal,runMain,Option.bind_some,
          Bool.false_eq_true,↓reduceIte]
        rfl
      · simpa only [NativeReference.kernelMemory,NativeReference.clearMemory,
          show (parameters p).shape=p.shape from rfl,if_neg empty,show (parameters p).causal=p.causal from rfl,
          causal,Bool.false_eq_true,↓reduceIte] using memMain
    · obtain ⟨final,runCleanup,memCleanup⟩ := CleanupSource.complete rest (.leaf .fp32 0) (parameters p) main
      refine ⟨final,?_,?_⟩
      · simpa only [KernelExecution.Raw.source,show (parameters p).shape=p.shape from rfl,
          if_neg empty,show (parameters p).causal=p.causal from rfl,causal,runMain,Option.bind_some,↓reduceIte] using runCleanup
      · funext r off
        rw [memCleanup,memMain]
        simp only [NativeReference.kernelMemory,NativeReference.clearMemory,
          show (parameters p).shape=p.shape from rfl,if_neg empty,show (parameters p).causal=p.causal from rfl,
          causal,↓reduceIte]

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
