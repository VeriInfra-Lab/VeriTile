import bench.tritonbench_g.matmul_dequant_int4.Proofs.PublicIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.UnitStrideEquivalence
open VeriTile Triton FP.Structural FP.Structural.Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- These are proof-only projections of the two ordinary complete quotations. -/
def originalAddress (f : Source.Profile) (p : Source.Parameters) :=
  (Source.source f p).surfaceBody[4]'(by cases f <;> exact Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 13)))))
def unitAddress (f : Source.Profile) (p : Source.Parameters) :=
  (unitSource f p).surfaceBody[4]'(by cases f <;> exact Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 13)))))

theorem decomposition (f : Source.Profile) (p : Source.Parameters) :
    (Source.source f p).surfaceBody = (Source.source f p).surfaceBody.take 4 ++
      originalAddress f p :: (Source.source f p).surfaceBody.drop 5 := by cases f <;> rfl

theorem candidate_decomposition (f : Source.Profile) (p : Source.Parameters) :
    (unitSource f p).surfaceBody = (Source.source f p).surfaceBody.take 4 ++
      unitAddress f p :: (Source.source f p).surfaceBody.drop 5 := by cases f <;> rfl

theorem step_equal (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (s : State α) :
    step M (originalAddress f p) s = step M (unitAddress f p) s := by
  cases f
  all_goals
    simp only [originalAddress,unitAddress,Source.source,unitSource,unitTyped,
      Source.Profile.dtype,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,
      ComputeKernel.surfaceBody,List.getElem_cons_zero,List.getElem_cons_succ]
    cases k : s.regs .nat [p.blockRows] "offs_k" <;>
      cases n : s.regs .nat [p.blockColumns] "offs_n"
    all_goals simp [FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalOp_unfold,
      k,n,unitColumn,bind,Option.bind,FP.Structural.numeric,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      congr 1
      funext i
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem execution (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (s : State α) :
    exec M (Source.source f p) s = exec M (unitSource f p) s := by
  change run M (Source.source f p).surfaceBody s = run M (unitSource f p).surfaceBody s
  rw [decomposition,candidate_decomposition,run_append,run_append]
  cases h : run M ((Source.source f p).surfaceBody.take 4) s with
  | none => rfl
  | some t => simp only [Option.bind_some,run,step_equal f M p unitColumn]

theorem address_safe (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    stepSafe M bounds (originalAddress f p) s ∧ stepSafe M bounds (unitAddress f p) s := by
  cases f <;> simp [originalAddress,unitAddress,Source.source,unitSource,unitTyped,
    Source.Profile.dtype,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,
    ComputeKernel.surfaceBody,stepSafe_unfold,safeExpr_unfold,safeOp_unfold]

/-- Pure pointer arithmetic retains every source access and its safety condition. -/
theorem safety (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (Source.source f p) s ↔ TraceSafe M bounds (unitSource f p) s := by
  unfold TraceSafe
  rw [decomposition,candidate_decomposition,traceSafeList_append,traceSafeList_append]
  constructor <;> rintro ⟨frontSafe,tail⟩ <;> refine ⟨frontSafe,?_⟩ <;> intro t ran
  all_goals have h := tail t ran
  all_goals simp only [traceSafeList_cons] at h ⊢
  · exact ⟨(address_safe f M p bounds t).2,by simpa only [step_equal f M p unitColumn] using h.2⟩
  · exact ⟨(address_safe f M p bounds t).1,by simpa only [step_equal f M p unitColumn] using h.2⟩

theorem launch_execution (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (pid : Nat) (s : State α) :
    (PublicIO.invocation f M p pid).run s = (PublicIO.unitInvocation f M p pid).run s :=
  execution f M p unitColumn ((PublicIO.invocation f M p pid).initial s)

theorem launch_safety (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (pid : Nat) (bounds : RegionName → Nat) (s : State α) :
    (PublicIO.invocation f M p pid).Safe bounds s ↔ (PublicIO.unitInvocation f M p pid).Safe bounds s :=
  safety f M p unitColumn bounds ((PublicIO.invocation f M p pid).initial s)

theorem schedule_execution (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (work : List Nat) (s : State α) :
    runSchedule (work.map (PublicIO.invocation f M p)) s =
      runSchedule (work.map (PublicIO.unitInvocation f M p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,launch_execution f M p unitColumn]
    congr 1
    funext t
    exact ih t

theorem schedule_safety (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (unitColumn : p.outputColumnStride=1) (work : List Nat) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (work.map (PublicIO.invocation f M p)) s ↔
      ScheduleSafe bounds (work.map (PublicIO.unitInvocation f M p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,ScheduleSafe,launch_safety f M p unitColumn,
      launch_execution f M p unitColumn,ih]

theorem equivalent (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) :
    (PublicIO.dequantIO f M p).Equivalent (PublicIO.unit f M p) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    PublicIO.implements f M p valid vals domain bounds bounded initial binding
  refine ⟨safe,(schedule_safety f M p valid.outputColumn _ bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_execution f M p valid.outputColumn _ initial).symm.trans ran
  · intro o i active
    rfl

#axiomsClean execution
#axiomsClean safety
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.UnitStrideEquivalence
