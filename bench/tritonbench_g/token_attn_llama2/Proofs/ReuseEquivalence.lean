import bench.tritonbench_g.token_attn_llama2.Proofs.IOImplements
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.ReuseEquivalence
open VeriTile Triton FP.Structural FP.Structural.Safety Whole
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def oldPair (p : Parameters) := (source p).surfaceBody.drop 10 |>.take 2
def newPair (p : Parameters) := (reuseSource p).surfaceBody.drop 10 |>.take 2

theorem pairs_equal (M : Algebra α) (p : Parameters) (s : State α) :
    run M (oldPair p) s = run M (newPair p) s := by
  simp only [oldPair,newPair,source,VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source,reuseSource,typed,VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.typed,reuseTyped,ComputeKernel.surfaceBody,
    List.drop,List.take,run,step,evalExpr,FP.Structural.evalOp_unfold]
  cases h : s.regs .nat [] "start_n" with
  | none => simp [h,bind]
  | some v =>
    simp [h,bind,State.setReg,FP.Structural.numeric,FP.Structural.bop]
    congr 1
    funext d sh key
    by_cases a : key="offs_n" <;> by_cases b : key="block_stard_index" <;> simp_all

theorem decomposition (p : Parameters) :
    (source p).surfaceBody = (source p).surfaceBody.take 10 ++ oldPair p ++ (source p).surfaceBody.drop 12 := rfl

theorem candidate_decomposition (p : Parameters) :
    (reuseSource p).surfaceBody = (source p).surfaceBody.take 10 ++ newPair p ++ (source p).surfaceBody.drop 12 := rfl

theorem execution (M : Algebra α) (p : Parameters) (s : State α) :
    exec M (source p) s = exec M (reuseSource p) s := by
  change run M (source p).surfaceBody s = run M (reuseSource p).surfaceBody s
  rw [decomposition,candidate_decomposition]
  simp only [run_append]
  have equal : run M (oldPair p) = run M (newPair p) := funext (pairs_equal M p)
  rw [equal]

theorem pair_safety (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList M bounds (oldPair p) s ∧ traceSafeList M bounds (newPair p) s := by
  cases h : s.regs .nat [] "start_n" <;>
    simp [oldPair,newPair,source,VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source,reuseSource,typed,VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.typed,reuseTyped,ComputeKernel.surfaceBody,
    h,bind,FP.Structural.numeric,FP.Structural.bop,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeOp_unfold,run,step,evalExpr,
    FP.Structural.evalOp_unfold,State.setReg]

theorem safety (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (source p) s ↔ TraceSafe M bounds (reuseSource p) s := by
  unfold TraceSafe
  rw [decomposition,candidate_decomposition]
  simp only [List.append_assoc]
  rw [traceSafeList_append,traceSafeList_append]
  constructor <;> rintro ⟨frontSafe,tail⟩ <;> refine ⟨frontSafe,?_⟩ <;> intro t ran
  all_goals have h := tail t ran
  all_goals simp only [traceSafeList_append] at h ⊢
  · exact ⟨(pair_safety M p bounds t).2,by simpa only [pairs_equal] using h.2⟩
  · exact ⟨(pair_safety M p bounds t).1,by simpa only [pairs_equal] using h.2⟩

theorem launch_execution (M : Program → Algebra α) (c : Call) (scale : ℝ)
    (pid : Program) (s : State α) :
    (Interface.invocation M c scale pid).run s = (Interface.reuseInvocation M c scale pid).run s :=
  execution (M pid) (c.parameters scale) ((Interface.invocation M c scale pid).initial s)

theorem launch_safety (M : Program → Algebra α) (c : Call) (scale : ℝ)
    (pid : Program) (bounds : RegionName → Nat) (s : State α) :
    (Interface.invocation M c scale pid).Safe bounds s ↔
      (Interface.reuseInvocation M c scale pid).Safe bounds s :=
  safety (M pid) (c.parameters scale) bounds ((Interface.invocation M c scale pid).initial s)

theorem schedule_execution (M : Program → Algebra α) (c : Call) (scale : ℝ)
    (work : List Program) (s : State α) :
    runSchedule (work.map (Interface.invocation M c scale)) s =
      runSchedule (work.map (Interface.reuseInvocation M c scale)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,launch_execution]
    congr 1
    funext t
    exact ih t

theorem schedule_safety (M : Program → Algebra α) (c : Call) (scale : ℝ)
    (work : List Program) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (work.map (Interface.invocation M c scale)) s ↔
      ScheduleSafe bounds (work.map (Interface.reuseInvocation M c scale)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,ScheduleSafe,launch_safety,launch_execution,ih]

theorem equivalent (M : Program → Algebra α) (c : Call) (m : Metadata c) (scale : ℝ)
    (valid : Valid c m) (work : List Program) (order : (programs c.grid).Perm work) :
    (Interface.symbolicIO M c m scale work).EquivalentOn (Interface.reuse M c m scale work)
      (Interface.MetadataValues c m) := by
  have correct : (Interface.symbolicIO M c m scale work).ImplementsOn (Interface.MetadataValues c m)
      (fun xs => Interface.result (Reference.score M (fun _ _ x => x) c m scale xs)) :=
    IOImplements.observed M c m scale work order (fun _ _ x => x) id valid _ (fun _ _ _ _ _ => rfl)
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ := correct vals domain bounds bounded initial binding
  refine ⟨safe,(schedule_safety M c scale work bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_execution M c scale work initial).symm.trans ran
  · intro o i active
    rfl

#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.ReuseEquivalence
