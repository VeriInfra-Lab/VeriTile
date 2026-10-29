import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IntegerCandidate

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IntegerCandidate
open VeriTile.Triton FP.Structural FPValue
noncomputable section

@[simp] theorem forget_twice (s : State Word) :
    forgetLocals (forgetLocals s)=forgetLocals s := rfl

@[simp] theorem initial_forget (job : Invocation Word)
    (clear : job.clearRegisters=Bool.true) (s : State Word) :
    job.initial (forgetLocals s)=job.initial s := by
  simp [Invocation.initial,forgetLocals,clear]

theorem schedule_run_forget (jobs : List (Invocation Word))
    (clear : ∀ job∈jobs,job.clearRegisters=Bool.true) (s : State Word) :
    (runSchedule jobs (forgetLocals s)).map forgetLocals=
      (runSchedule jobs s).map forgetLocals := by
  cases jobs with
  | nil => rfl
  | cons job jobs =>
    simp only [runSchedule,Invocation.run,initial_forget job (clear job (by simp))]

theorem schedule_safe_forget (jobs : List (Invocation Word))
    (clear : ∀ job∈jobs,job.clearRegisters=Bool.true)
    (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds jobs (forgetLocals s) ↔ ScheduleSafe bounds jobs s := by
  cases jobs with
  | nil => rfl
  | cons job jobs =>
    simp only [ScheduleSafe,Invocation.Safe,Invocation.run,
      initial_forget job (clear job (by simp))]

theorem schedule_run_congr (jobs : List (Invocation Word))
    (clear : ∀ job∈jobs,job.clearRegisters=Bool.true)
    (a b : State Word) (same : forgetLocals a=forgetLocals b) :
    (runSchedule jobs a).map forgetLocals=(runSchedule jobs b).map forgetLocals := by
  rw [←schedule_run_forget jobs clear a,←schedule_run_forget jobs clear b,same]

theorem schedule_safe_congr (jobs : List (Invocation Word))
    (clear : ∀ job∈jobs,job.clearRegisters=Bool.true) (bounds : RegionName → Nat)
    (a b : State Word) (same : forgetLocals a=forgetLocals b) :
    ScheduleSafe bounds jobs a ↔ ScheduleSafe bounds jobs b := by
  rw [←schedule_safe_forget jobs clear bounds a,
    ←schedule_safe_forget jobs clear bounds b,same]

#axiomsClean initial_forget
#axiomsClean schedule_run_congr
#axiomsClean schedule_safe_congr
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IntegerCandidate
