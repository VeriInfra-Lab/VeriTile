import bench.tritonbench_g.llama_ff_triton.IO

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem job_initial {α : Type} (M : Algebra α) (kernel : ComputeKernel) (pid : Nat)
    (first : Bool) (s : State α) :
    (job M kernel pid first).initial s = if first then GridExecution.atProgram s pid else s := by
  cases first <;> rfl

theorem job_run {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel) (pid : Nat)
    (first : Bool) (s : State α) :
    (job M kernel pid first).run s = exec M kernel
      (if first then GridExecution.atProgram s pid else s) := by
  rw [Invocation.run,job_initial]
  rfl

theorem native_jobs_run {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (pid : Nat) (s : State α) :
    runSchedule (nativeJobs a g p pid) s = nativeExecution a g p (GridExecution.atProgram s pid) := by
  simp only [nativeJobs,runSchedule,job_run,ite_true,
    Bool.false_eq_true,ite_false,FP.Structural.exec,ComputeKernel.surfaceBody]
  unfold nativeExecution
  cases hs : run a.model (prelude p) (GridExecution.atProgram s pid) with
  | none => rfl
  | some start =>
    simp only [hs,Option.bind]
    dsimp only [Bind.bind]
    cases hm : FP.Structural.step a.model
      (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)) start with
    | none => simp [run,hm,Bind.bind]
    | some mid =>
      simp only [run,hm,Option.bind]
      dsimp only [Bind.bind]
      cases hn : run a.model ((epilogue p).take 2) mid with
      | none => simp only [hn,Option.bind]
      | some norm =>
        simp only [hn,Option.bind]
        rw [NativeStages.gate_exact]
        cases hg : gateExecution a g norm <;> simp [hg,runSchedule]
        split <;> simp_all


theorem jobs_run {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (pid : Nat) (s : State α) :
    runSchedule (jobs native a g p pid) s = Grid.one native a g p pid s := by
  cases native
  · simp [jobs,runSchedule_singleton,job_run,Grid.one]
  · exact native_jobs_run a g p pid s

theorem schedule_run {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (pids : List Nat) (s : State α) :
    runSchedule (pids.flatMap (jobs native a g p)) s =
      GridExecution.sequence (Grid.one native a g p) pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    rw [List.flatMap_cons,runSchedule_append,jobs_run]
    simp only [GridExecution.sequence]
    cases h : Grid.one native a g p pid s <;> simp [h,ih]

theorem program_run {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) (s : State α) :
    runSchedule ((programIO native a g p).schedule xs) s = Grid.grid native a g p s :=
  schedule_run native a g p _ s

#axiomsClean native_jobs_run
#axiomsClean program_run
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
