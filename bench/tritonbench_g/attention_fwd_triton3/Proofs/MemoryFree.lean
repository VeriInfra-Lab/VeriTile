import bench.tritonbench_g.attention_fwd_triton3.Proofs.Program
import bench.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetySteps

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Safety FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
open Reference
open VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety (safeBlockPtr2)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
attribute [local simp 1100] FP.Structural.eval_truncDiv

theorem if_else_safe (A : Algebra α) (bounds : RegionName → Nat) (s : State α)
    (condition : Op .bool []) (yes no : List ComputeStmt)
    (checked : safeOp A none bounds s condition)
    (hy : traceSafeList A bounds yes s) (hn : traceSafeList A bounds no s) :
    stepSafe A bounds (.ifThenElse condition yes no) s := by
  rw [stepSafe_unfold]
  refine ⟨checked,?_⟩
  cases evalOp A none condition s with
  | none => trivial
  | some value => dsimp; split <;> assumption

theorem if_safe (A : Algebra α) (bounds : RegionName → Nat) (s : State α)
    (condition : Op .bool []) (yes : List ComputeStmt)
    (checked : safeOp A none bounds s condition)
    (hy : traceSafeList A bounds yes s) : stepSafe A bounds (.ifThen condition yes) s := by
  rw [stepSafe_unfold]
  refine ⟨checked,?_⟩
  cases evalOp A none condition s with
  | none => trivial
  | some value => dsimp; split <;> trivial

theorem safeAdvance2 (A : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (ptr : Op .blockPtr shape)
    (a b : Op .int []) :
    safeOp A precision bounds s (.advanceBlockPtrDyn ptr [a,b]) ↔
      safeOp A precision bounds s ptr ∧ safeOp A precision bounds s a ∧ safeOp A precision bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

syntax "prove_memory_free" : tactic
macro_rules
| `(tactic| prove_memory_free) => `(tactic|
  first
  | exact traceSafeList_nil _ _ _
  | (apply traceSafeList_cons_intro
     · prove_memory_free
     · intro next _; prove_memory_free)
  | (apply if_else_safe
     · prove_memory_free
     · prove_memory_free
     · prove_memory_free)
  | (apply if_safe
     · prove_memory_free
     · prove_memory_free)
  | (try simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
     try rw [safeAdvance2]
     try rw [safeBlockPtr2]
     simp only [safeOp_unfold,true_and,and_true]))

theorem append_of_run {M : Algebra α} {bounds : RegionName → Nat}
    {before after : List ComputeStmt} {s t : State α}
    (safe : traceSafeList M bounds before s) (ran : run M before s=some t)
    (tail : traceSafeList M bounds after t) : traceSafeList M bounds (before++after) s := by
  apply (traceSafeList_append M bounds before after s).mpr
  refine ⟨safe,?_⟩
  intro u hu
  have eq := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using tail

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
