import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyAligned

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
open VeriTile Triton FP.Structural Safety
noncomputable section
variable {α : Type} [Inhabited α]

theorem all_steps_safe (M : Algebra α) (bounds : RegionName → Nat)
    (body : List ComputeStmt) (safe : ∀ st∈body,∀ s,stepSafe M bounds st s) (s : State α) :
    traceSafeList M bounds body s := by
  induction body generalizing s with
  | nil => exact traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply traceSafeList_cons_intro
    · exact safe st List.mem_cons_self s
    · intro t _
      exact ih (fun x hx=>safe x (List.mem_cons_of_mem st hx)) t

theorem prefix_iff (M : Algebra α) (bounds : RegionName → Nat)
    (before after : List ComputeStmt) (s ready : State α)
    (safe : traceSafeList M bounds before s) (ran : run M before s=some ready) :
    traceSafeList M bounds (before++after) s ↔ traceSafeList M bounds after ready := by
  rw [traceSafeList_append]
  simp only [safe,ran,true_and,Option.some.injEq]
  constructor
  · intro h; exact h ready rfl
  · intro h t ht; simpa only [←ht] using h

theorem singleton_iff (M : Algebra α) (bounds : RegionName → Nat)
    (statement : ComputeStmt) (s : State α) :
    traceSafeList M bounds [statement] s ↔ stepSafe M bounds statement s := by
  rw [traceSafeList_cons]
  simp only [traceSafeList_nil,implies_true,forall_const,and_true]

theorem false_branch_iff (M : Algebra α) (bounds : RegionName → Nat)
    (yes no : List ComputeStmt) (s : State α) :
    traceSafeList M bounds [.ifThenElse (.constBool Bool.false) yes no] s ↔
      traceSafeList M bounds no s := by
  rw [singleton_iff,stepSafe_unfold]
  simp only [safeOp_unfold,evalOp_unfold,true_and,Bool.false_eq_true,↓reduceIte]

theorem signed_loop_tail_iff (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (start stop : Op .int []) (stride : Op .nat [])
    (body tail : List ComputeStmt) (s : State α) (a b : Int) (d : Nat)
    (sa : safeOp M none bounds s start) (sb : safeOp M none bounds s stop)
    (sd : safeOp M none bounds s stride)
    (ha : evalOp M none start s=some (fun _=>a))
    (hb : evalOp M none stop s=some (fun _=>b))
    (hd : evalOp M none stride s=some (fun _=>d)) :
    traceSafeList M bounds (.forRangeInt idx start stop stride body::tail) s ↔
      rangeIntSafe M bounds idx a b d body s ∧
      ∀ t,rangeInt M idx a b d body s=some t → traceSafeList M bounds tail t := by
  rw [traceSafeList_cons]
  simp only [stepSafe_unfold,step,sa,sb,sd,ha,hb,hd,true_and,bind,Option.bind_some]

theorem signed_loop_only_iff (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (start stop : Op .int []) (stride : Op .nat [])
    (body : List ComputeStmt) (s : State α) (a b : Int) (d : Nat)
    (sa : safeOp M none bounds s start) (sb : safeOp M none bounds s stop)
    (sd : safeOp M none bounds s stride)
    (ha : evalOp M none start s=some (fun _=>a))
    (hb : evalOp M none stop s=some (fun _=>b))
    (hd : evalOp M none stride s=some (fun _=>d)) :
    stepSafe M bounds (.forRangeInt idx start stop stride body) s ↔
      rangeIntSafe M bounds idx a b d body s := by
  simp only [stepSafe_unfold,sa,sb,sd,ha,hb,hd,true_and]

theorem nat_loop_tail_iff (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (start stop stride : Nat) (body tail : List ComputeStmt) (s : State α) :
    traceSafeList M bounds (.forRange idx start stop stride body::tail) s ↔
      rangeSafe M bounds idx start stop stride body s ∧
      ∀ t,range M idx start stop stride body s=some t → traceSafeList M bounds tail t := by
  rw [traceSafeList_cons]
  simp only [stepSafe_unfold,step]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
