import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.Interface
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.AffineEquivalence
open VeriTile Triton FP.Structural FP.Structural.Safety TokenAttnReduceV
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def originalOffset (p : Parameters) : Op .nat [p.block] :=
  .add .nat .scalarL
    (.mul .nat .nil (.ref .nat [] "cur_head") (.constNat p.probHeadStride))
    (.mul .nat .scalarR
      (.add .nat .scalarL (.ref .nat [] "cur_batch_in_all_start_index")
        (.ref .nat [p.block] "offs_n")) (.constNat p.probTokenStride))

def affineOffset (p : Parameters) : Op .nat [p.block] :=
  .add .nat .scalarL
    (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "cur_head") (.constNat p.probHeadStride))
      (.mul .nat .nil (.ref .nat [] "cur_batch_in_all_start_index") (.constNat p.probTokenStride)))
    (.mul .nat .scalarR (.ref .nat [p.block] "offs_n") (.constNat p.probTokenStride))

theorem offset_eval (M : Algebra α) (p : Parameters) (s : State α) :
    FP.Structural.evalOp M none (originalOffset p) s = FP.Structural.evalOp M none (affineOffset p) s := by
  simp only [originalOffset,affineOffset,FP.Structural.evalOp_unfold]
  cases h : s.regs .nat [] "cur_head" <;>
    cases b : s.regs .nat [] "cur_batch_in_all_start_index" <;>
    cases n : s.regs .nat [p.block] "offs_n"
  all_goals simp [h,b,n,bind,Option.bind,FP.Structural.numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]
  funext i
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]

theorem decomposition (p : Parameters) :
    (kernel p).surfaceBody = (kernel p).surfaceBody.take 10 ++
      .assign .nat [p.block] "p_offs" (.alg (originalOffset p)) :: (kernel p).surfaceBody.drop 11 := rfl

theorem candidate_decomposition (p : Parameters) :
    (affineKernel p).surfaceBody = (kernel p).surfaceBody.take 10 ++
      .assign .nat [p.block] "p_offs" (.alg (affineOffset p)) :: (kernel p).surfaceBody.drop 11 := rfl

theorem step_equal (M : Algebra α) (p : Parameters) (s : State α) :
    step M (.assign .nat [p.block] "p_offs" (.alg (originalOffset p))) s =
      step M (.assign .nat [p.block] "p_offs" (.alg (affineOffset p))) s := by
  simp only [FP.Structural.step,FP.Structural.evalExpr,offset_eval]

theorem execution (M : Algebra α) (p : Parameters) (s : State α) :
    exec M (kernel p) s = exec M (affineKernel p) s := by
  change FP.Structural.run M (kernel p).surfaceBody s = FP.Structural.run M (affineKernel p).surfaceBody s
  rw [decomposition,candidate_decomposition,run_append,run_append]
  cases h : run M ((kernel p).surfaceBody.take 10) s with
  | none => rfl
  | some t => simp only [Option.bind_some,run,step_equal]

/-- The changed statement is pure natural arithmetic, with exactly the same
result and failures. Thus all source memory accesses retain their full trace
safety obligations, including the distinct P and V masks. -/
theorem safety (M : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (kernel p) s ↔ TraceSafe M bounds (affineKernel p) s := by
  unfold TraceSafe
  rw [decomposition,candidate_decomposition,traceSafeList_append,traceSafeList_append]
  constructor <;> rintro ⟨frontSafe,tail⟩ <;> refine ⟨frontSafe,?_⟩ <;> intro t ran
  all_goals have h := tail t ran
  all_goals simp only [traceSafeList_cons] at h ⊢
  all_goals refine ⟨?_,?_⟩
  all_goals first
    | solve | simp [stepSafe_unfold,safeExpr_unfold,safeOp_unfold,originalOffset,affineOffset]
    | simpa only [step_equal] using h.2

theorem launch_execution (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (pid : Grid.PID) (s : State α) :
    (Interface.invocation M R p pid).run s = (Interface.affineInvocation M R p pid).run s :=
  execution (withSum M R) p ((Interface.invocation M R p pid).initial s)

theorem launch_safety (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (pid : Grid.PID) (bounds : RegionName → Nat) (s : State α) :
    (Interface.invocation M R p pid).Safe bounds s ↔ (Interface.affineInvocation M R p pid).Safe bounds s :=
  safety (withSum M R) p bounds ((Interface.invocation M R p pid).initial s)

theorem schedule_execution (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (s : State α) :
    runSchedule (work.map (Interface.invocation M R p)) s =
      runSchedule (work.map (Interface.affineInvocation M R p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,launch_execution]
    congr 1
    funext t
    exact ih t

theorem schedule_safety (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (work.map (Interface.invocation M R p)) s ↔
      ScheduleSafe bounds (work.map (Interface.affineInvocation M R p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,ScheduleSafe,launch_safety,launch_execution,ih]

/-- Complete original/candidate grid, with all source safety and frames. -/
theorem equivalent (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (work : List Grid.PID)
    (order : (Grid.fullGrid B H).Perm work)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    (Interface.programIO M R p B H counts work).EquivalentOn
      (Interface.affine M R p B H counts work) (Interface.LengthsMatch p B H counts) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    Interface.implements M R p B H counts work order separate unique vals domain bounds bounded initial binding
  refine ⟨safe,(schedule_safety M R p work bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_execution M R p work initial).symm.trans ran
  · intro o i active
    rfl

#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.AffineEquivalence
