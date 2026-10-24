import bench.repairs.tritonbench_g.matmul_dequantize_int4.Candidate
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONative

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.GroupExtent
open VeriTile Triton FP.Structural FP.Structural.Safety
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4 MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]
/-- Task-local transport across a pure source prefix; the remainder is the
same complete quoted kernel, including every branch, load, dot and store. -/
theorem prefix_execution (M : Algebra A) (original candidate : ComputeKernel) (n : Nat)
    (frontRun : ∀ s, run M (original.surfaceBody.take n) s=run M (candidate.surfaceBody.take n) s)
    (suffix : original.surfaceBody.drop n=candidate.surfaceBody.drop n) (s : State A) :
    exec M original s=exec M candidate s := by
  change run M original.surfaceBody s=run M candidate.surfaceBody s
  rw [←List.take_append_drop n original.surfaceBody,←List.take_append_drop n candidate.surfaceBody,
    run_append,run_append,frontRun,suffix]

theorem prefix_safety (M : Algebra A) (original candidate : ComputeKernel) (n : Nat)
    (frontRun : ∀ s, run M (original.surfaceBody.take n) s=run M (candidate.surfaceBody.take n) s)
    (suffix : original.surfaceBody.drop n=candidate.surfaceBody.drop n)
    (pureOriginal : ∀ bounds s,traceSafeList M bounds (original.surfaceBody.take n) s)
    (pureCandidate : ∀ bounds s,traceSafeList M bounds (candidate.surfaceBody.take n) s)
    (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds original s ↔ TraceSafe M bounds candidate s := by
  unfold TraceSafe
  rw [←List.take_append_drop n original.surfaceBody,←List.take_append_drop n candidate.surfaceBody,
    traceSafeList_append,traceSafeList_append]
  simp only [pureOriginal,pureCandidate,true_and,frontRun,suffix]

theorem frontRun (M : Algebra A) (p : Parameters) (s : State A) :
    run M ((FPTyped.source p).surfaceBody.take 7) s=run M ((groupedSource p).surfaceBody.take 7) s := by
  simp [FPTyped.source,groupedSource,
    FPTyped.typed,groupedTyped,
    ComputeKernel.surfaceBody,run,step,evalExpr,FP.Structural.evalOp_unfold,
    FP.Structural.numeric,FP.Structural.bop,Option.bind,bind,integralFloorDiv]
  congr 1


theorem pure_front (M : Algebra A) (p : Parameters) (bounds : RegionName → Nat) (s : State A) :
    traceSafeList M bounds ((FPTyped.source p).surfaceBody.take 7) s ∧
      traceSafeList M bounds ((groupedSource p).surfaceBody.take 7) s := by
  simp [FPTyped.source,groupedSource,FPTyped.typed,groupedTyped,
    ComputeKernel.surfaceBody,traceSafeList_cons,stepSafe_unfold,safeExpr_unfold,safeOp_unfold]

/-- The two complete source kernels have the exact same state, for any typed
numeric backend; this does not identify separate multiply/add with FMA. -/
theorem source_execution (M : Algebra A) (p : Parameters) (s : State A) :
    exec M (FPTyped.source p) s=exec M (groupedSource p) s :=
  prefix_execution M _ _ 7 (frontRun M p) (Candidate.source_tail p).symm s

theorem source_safety (M : Algebra A) (p : Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (FPTyped.source p) s ↔ TraceSafe M bounds (groupedSource p) s :=
  prefix_safety M _ _ 7 (frontRun M p) (Candidate.source_tail p).symm
    (fun b t => (pure_front M p b t).1) (fun b t => (pure_front M p b t).2) bounds s

/-- The complete compiled quotation has the same exact pure-index transport;
its existing accumulator operand and precision boundaries stay intact. -/
theorem compiled_execution (M : Algebra A) (p : Parameters) (s : State A) :
    exec M (FPCompiled.source p) s=exec M (groupedCompiledSource p) s :=
  prefix_execution M (FPCompiled.source p) (groupedCompiledSource p) 7
    (frontRun M p) (Candidate.compiled_tail p).symm s

theorem compiled_safety (M : Algebra A) (p : Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (FPCompiled.source p) s ↔ TraceSafe M bounds (groupedCompiledSource p) s :=
  prefix_safety M (FPCompiled.source p) (groupedCompiledSource p) 7
    (frontRun M p) (Candidate.compiled_tail p).symm
    (fun b t => (pure_front M p b t).1) (fun b t => (pure_front M p b t).2) bounds s

theorem prelude_first (p : Parameters) :
    (IONativeGraph.preludeKernel p).surfaceBody.take 7=(FPTyped.source p).surfaceBody.take 7 ∧
    (Candidate.preludeKernel p).surfaceBody.take 7=(groupedSource p).surfaceBody.take 7 := ⟨rfl,rfl⟩

theorem prelude_tail (p : Parameters) :
    (IONativeGraph.preludeKernel p).surfaceBody.drop 7=(Candidate.preludeKernel p).surfaceBody.drop 7 := rfl

theorem prelude_front (M : Algebra A) (p : Parameters) (s : State A) :
    run M ((IONativeGraph.preludeKernel p).surfaceBody.take 7) s=
      run M ((Candidate.preludeKernel p).surfaceBody.take 7) s := by
  rw [(prelude_first p).1,(prelude_first p).2]
  exact frontRun M p s

theorem prelude_execution (M : Algebra A) (p : Parameters) (s : State A) :
    exec M (IONativeGraph.preludeKernel p) s=exec M (Candidate.preludeKernel p) s :=
  prefix_execution M _ _ 7 (prelude_front M p) (prelude_tail p) s

theorem prelude_safety (M : Algebra A) (p : Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (IONativeGraph.preludeKernel p) s ↔ TraceSafe M bounds (Candidate.preludeKernel p) s := by
  apply prefix_safety M _ _ 7 (prelude_front M p) (prelude_tail p)
  · intro b t
    rw [(prelude_first p).1]
    exact (pure_front M p b t).1
  · intro b t
    rw [(prelude_first p).2]
    exact (pure_front M p b t).2

theorem launch_execution (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (pid : Nat) (s : State Word) :
    (IOExecution.launch (IONativeGraph.preludeKernel p) (IONativeGraph.model actual op) pid).run s=
      (IOExecution.launch (Candidate.preludeKernel p) (IONativeGraph.model actual op) pid).run s :=
  prelude_execution _ p ((IOExecution.launch (IONativeGraph.preludeKernel p) (IONativeGraph.model actual op) pid).initial s)

theorem launch_safety (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (pid : Nat) (bounds : RegionName → Nat) (s : State Word) :
    (IOExecution.launch (IONativeGraph.preludeKernel p) (IONativeGraph.model actual op) pid).Safe bounds s ↔
      (IOExecution.launch (Candidate.preludeKernel p) (IONativeGraph.model actual op) pid).Safe bounds s :=
  prelude_safety _ p bounds ((IOExecution.launch (IONativeGraph.preludeKernel p) (IONativeGraph.model actual op) pid).initial s)

theorem program_execution (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (pid : Nat) (s : State Word) :
    runSchedule (IONativeGraph.programJobs actual op p pid) s=
      runSchedule (Candidate.programJobs actual op p pid) s := by
  simp only [IONativeGraph.programJobs,Candidate.programJobs,List.singleton_append,List.cons_append,
    runSchedule,launch_execution]

theorem program_safety (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (pid : Nat) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (IONativeGraph.programJobs actual op p pid) s ↔
      ScheduleSafe bounds (Candidate.programJobs actual op p pid) s := by
  simp only [IONativeGraph.programJobs,Candidate.programJobs,List.singleton_append,List.cons_append,
    ScheduleSafe,launch_safety,launch_execution]

theorem grid_execution (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (pids : List Nat) (s : State Word) :
    runSchedule (pids.flatMap (IONativeGraph.programJobs actual op p)) s=
      runSchedule (pids.flatMap (Candidate.programJobs actual op p)) s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.flatMap_cons,runSchedule_append,program_execution]
    congr 1
    funext t
    exact ih t

theorem grid_safety (actual : Native.Instructions) (op : Opcode) (p : Parameters)
    (pids : List Nat) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (pids.flatMap (IONativeGraph.programJobs actual op p)) s ↔
      ScheduleSafe bounds (pids.flatMap (Candidate.programJobs actual op p)) s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.flatMap_cons,scheduleSafe_append,program_safety,program_execution,ih]

theorem equivalent (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    (Candidate.program actual c i).Equivalent (Candidate.variant actual c i) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    IONative.implements actual mma ports c valid i vals domain bounds bounded initial binding
  have safeTransport := grid_safety actual (tunings i).opcode (params c.call i)
    (List.range (KernelSource.count c.call i)) bounds initial
  have runTransport := grid_execution actual (tunings i).opcode (params c.call i)
    (List.range (KernelSource.count c.call i)) initial
  refine ⟨safe,safeTransport.mp safe,final,final,ran,runTransport.symm.trans ran,?_,frame,frame⟩
  intro o j active
  rfl

#axiomsClean source_execution
#axiomsClean source_safety
#axiomsClean compiled_execution
#axiomsClean compiled_safety
#axiomsClean prelude_execution
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.GroupExtent
