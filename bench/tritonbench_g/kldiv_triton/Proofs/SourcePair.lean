import bench.tritonbench_g.kldiv_triton.Kernels
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.KldivTriton.SourcePair
open VeriTile Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def rowPointer (v : Bool) (name : RegName) (R : RegionName) (stride : Nat) : ComputeStmt :=
  .assign .ptr [] name (.alg (.ptrAdd .nil (.ptrBase R) (.castIntToNat
    (if v then .mul .int .nil (.castNatToInt (.constNat stride)) (.ref .int [] "pid")
    else .mul .int .nil (.ref .int [] "pid") (.castNatToInt (.constNat stride))))))
def forwardPrefix (v : Bool) (sy st so : Nat) : List ComputeStmt :=
  [.assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),
   rowPointer Bool.false "y_ptr" "y_ptr" sy,
   rowPointer Bool.false "gt_ptr" "gt_ptr" st,rowPointer v "loss_ptr" "loss_ptr" so]
def backwardPrefix (v : Bool) (so : Nat) : List ComputeStmt :=
  [.assign .int [] "pid" (.alg (.castNatToInt (.programId 0))),rowPointer v "input_ptr" "input_ptr" so]

theorem forward_decomposition (sy st so N B : Nat) (flag v : Bool) (red : Nat) :
    (if v then forwardCandidateSource sy st so N B flag red else forwardSource sy st so N B flag red).surfaceBody =
      forwardPrefix v sy st so ++ (forwardSource sy st so N B flag red).surfaceBody.drop 4 := by cases v <;> rfl

theorem backward_decomposition (so st N B : Nat) (flag v : Bool) :
    (if v then backwardCandidateSource so st N B flag else backwardSource so st N B flag).surfaceBody =
      backwardPrefix v so ++ (backwardSource so st N B flag).surfaceBody.drop 2 := by cases v <;> rfl

theorem rowPointer_step (M : Algebra α) (s : State α) (name : RegName) (R : RegionName) (stride : Nat) :
    step M (rowPointer Bool.true name R stride) s = step M (rowPointer Bool.false name R stride) s := by
  simp only [rowPointer,Bool.false_eq_true,ite_true,ite_false,step,evalExpr,evalOp_unfold,
    Option.bind_eq_bind,Option.pure_def,Option.bind]
  cases h : s.regs .int [] "pid"
  · rfl
  · simp [numeric,FP.Structural.bop,Int.mul_comm]

theorem forward_prefix_run (M : Algebra α) (s : State α) (sy st so : Nat) :
    run M (forwardPrefix Bool.true sy st so) s=run M (forwardPrefix Bool.false sy st so) s := by
  simp only [forwardPrefix,run,rowPointer_step]

theorem backward_prefix_run (M : Algebra α) (s : State α) (so : Nat) :
    run M (backwardPrefix Bool.true so) s=run M (backwardPrefix Bool.false so) s := by
  simp only [backwardPrefix,run,rowPointer_step]

theorem forward_run (M : Algebra α) (sy st so N B : Nat) (flag : Bool) (red : Nat) (s : State α) :
    FP.Structural.exec M (forwardCandidateSource sy st so N B flag red) s=FP.Structural.exec M (forwardSource sy st so N B flag red) s := by
  change run M (forwardCandidateSource sy st so N B flag red).surfaceBody s =
    run M (forwardSource sy st so N B flag red).surfaceBody s
  conv_lhs => rw [show (forwardCandidateSource sy st so N B flag red).surfaceBody = _ from
    forward_decomposition sy st so N B flag Bool.true red]
  conv_rhs => rw [show (forwardSource sy st so N B flag red).surfaceBody = _ from
    forward_decomposition sy st so N B flag Bool.false red]
  rw [run_append,run_append,forward_prefix_run]

theorem backward_run (M : Algebra α) (so st N B : Nat) (flag : Bool) (s : State α) :
    FP.Structural.exec M (backwardCandidateSource so st N B flag) s=FP.Structural.exec M (backwardSource so st N B flag) s := by
  change run M (backwardCandidateSource so st N B flag).surfaceBody s =
    run M (backwardSource so st N B flag).surfaceBody s
  conv_lhs => rw [show (backwardCandidateSource so st N B flag).surfaceBody = _ from
    backward_decomposition so st N B flag Bool.true]
  conv_rhs => rw [show (backwardSource so st N B flag).surfaceBody = _ from
    backward_decomposition so st N B flag Bool.false]
  rw [run_append,run_append,backward_prefix_run]

private theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ cmd ∈ code, ∀ s, stepSafe M bounds cmd s) (s : State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons cmd rest ih =>
    apply traceSafeList_cons_intro (safe cmd (by simp) s)
    intro t _
    exact ih (fun next h => safe next (by simp [h])) t

private theorem prefix_safe (M : Algebra α) (bounds : RegionName → Nat)
    (sy st so : Nat) (v : Bool) (s : State α) :
    traceSafeList M bounds (forwardPrefix v sy st so) s ∧ traceSafeList M bounds (backwardPrefix v so) s := by
  constructor <;> apply pure_trace <;> intro cmd member before
  all_goals simp only [forwardPrefix,backwardPrefix,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl | rfl
  all_goals cases v <;> simp only [rowPointer,Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    stepSafe_unfold,safeExpr_unfold,safeOp_unfold,true_and,and_true]

theorem forward_safe (M : Algebra α) (sy st so N B : Nat) (flag : Bool) (red : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (forwardCandidateSource sy st so N B flag red) s ↔
      TraceSafe M bounds (forwardSource sy st so N B flag red) s := by
  unfold TraceSafe
  conv_lhs => rw [show (forwardCandidateSource sy st so N B flag red).surfaceBody = _ from
    forward_decomposition sy st so N B flag Bool.true red]
  conv_rhs => rw [show (forwardSource sy st so N B flag red).surfaceBody = _ from
    forward_decomposition sy st so N B flag Bool.false red]
  rw [traceSafeList_append,traceSafeList_append]
  simp only [(prefix_safe M bounds sy st so Bool.true s).1,(prefix_safe M bounds sy st so Bool.false s).1,
    true_and,forward_prefix_run]

theorem backward_safe (M : Algebra α) (so st N B : Nat) (flag : Bool)
    (bounds : RegionName → Nat) (s : State α) :
    TraceSafe M bounds (backwardCandidateSource so st N B flag) s ↔
      TraceSafe M bounds (backwardSource so st N B flag) s := by
  unfold TraceSafe
  conv_lhs => rw [show (backwardCandidateSource so st N B flag).surfaceBody = _ from
    backward_decomposition so st N B flag Bool.true]
  conv_rhs => rw [show (backwardSource so st N B flag).surfaceBody = _ from
    backward_decomposition so st N B flag Bool.false]
  rw [traceSafeList_append,traceSafeList_append]
  simp only [(prefix_safe M bounds 0 st so Bool.true s).2,(prefix_safe M bounds 0 st so Bool.false s).2,
    true_and,backward_prefix_run]

#axiomsClean forward_run
#axiomsClean backward_run
#axiomsClean forward_safe
#axiomsClean backward_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.SourcePair
