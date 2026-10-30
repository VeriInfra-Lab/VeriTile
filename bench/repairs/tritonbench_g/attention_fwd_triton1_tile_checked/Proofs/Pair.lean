import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.RealInterface
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.Loop

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Pair
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option maxHeartbeats 200000
set_option maxRecDepth 32000

def body (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) : List ComputeStmt :=
  match ((kernel c scale storeFlag ifcond).surfaceBody[3]? : Option ComputeStmt) with
  | some (.forRange _ _ _ _ b) => b
  | _ => []

def pointers (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :=
  (body c scale storeFlag ifcond).take 5

theorem body_split (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    body c scale storeFlag ifcond = pointers c scale storeFlag ifcond ++
      (Program.body c scale storeFlag ifcond).drop 5 := rfl

theorem offset_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) :
    evalOp M p (.shiftLeft .nil (.ref .nat [] "i") (.constNat 5)) s =
      evalOp M p (Program.times "i" 32) s := by
  cases hi : s.regs .nat [] "i"
  all_goals simp [Program.times,evalOp_unfold,hi,numeric,FP.Structural.bop,Nat.shiftLeft_eq,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold FP.Structural.bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.shiftLeft_eq]

theorem pointer_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (s : State α) :
    run M (Program.pointers c) s = run M (pointers c scale storeFlag ifcond) s := by
  simp only [pointers,body,kernel,shiftedSource,ComputeKernel.surfaceBody,
    List.getElem?_cons_zero,List.getElem?_cons_succ,List.take_succ_cons,List.take_zero]
  simp only [Program.pointers,Program.pointer,Program.times,Program.headTimes,run,step,evalExpr,evalOp_unfold]
  simp only [evalOffsets]
  simp only [offset_eval,Program.times]

theorem safe_pointer {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) (name region : String)
    (parent block strides : List Nat) (base x y : Op .nat [])
    (hb : Safety.safeOp M none bounds s base)
    (hx : Safety.safeOp M none bounds s x) (hy : Safety.safeOp M none bounds s y) :
    Safety.stepSafe M bounds (Program.pointer name region parent block strides base [x,y]) s := by
  simp only [Program.pointer,Safety.stepSafe_unfold,Safety.safeExpr_unfold]
  rw [Safety.safeOp_unfold]
  change Safety.safeOp M none bounds s base ∧ ∀ off∈[x,y],Safety.safeOp M none bounds s off
  refine ⟨hb,?_⟩
  intro off member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl
  · exact hx
  · exact hy

theorem pure_list_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun q member => safe q (by simp [member])) t

theorem pointer_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State α) :
    Safety.traceSafeList M bounds (Program.pointers c) s ∧
      Safety.traceSafeList M bounds (pointers c scale storeFlag ifcond) s := by
  constructor
  · apply pure_list_safe
    intro st member t
    simp only [Program.pointers,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl
    all_goals apply safe_pointer
    all_goals simp only [Program.times,Program.headTimes,Safety.safeOp_unfold,true_and,and_true]
  · apply pure_list_safe
    intro st member t
    simp only [pointers,body,kernel,shiftedSource,ComputeKernel.surfaceBody,
      List.getElem?_cons_zero,List.getElem?_cons_succ,List.take_succ_cons,List.take_zero,
      List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl
    all_goals apply safe_pointer
    all_goals simp only [Safety.safeOp_unfold,true_and,and_true]

theorem original_body_split (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    Program.body c scale storeFlag ifcond = Program.pointers c ++
      (Program.body c scale storeFlag ifcond).drop 5 := rfl

theorem body_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (s : State α) :
    run M (Program.body c scale storeFlag ifcond) s = run M (body c scale storeFlag ifcond) s := by
  rw [original_body_split,body_split,run_append,run_append,pointer_run]

theorem body_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State α) :
    Safety.traceSafeList M bounds (Program.body c scale storeFlag ifcond) s ↔
      Safety.traceSafeList M bounds (body c scale storeFlag ifcond) s := by
  rw [original_body_split,body_split,Safety.traceSafeList_append,Safety.traceSafeList_append]
  simp only [(pointer_safe M bounds c scale storeFlag ifcond s).1,
    (pointer_safe M bounds c scale storeFlag ifcond s).2,true_and]
  rw [pointer_run M c scale storeFlag ifcond s]

theorem range_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (start : Nat) (s : State α) :
    range M "i" start c.chunks 1 (Program.body c scale storeFlag ifcond) s =
      range M "i" start c.chunks 1 (body c scale storeFlag ifcond) s := by
  apply range_congr_on_reachable
  intro cur t _ _ _
  exact body_run M c scale storeFlag ifcond _

theorem range_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (start : Nat) (s : State α) :
    Safety.rangeSafe M bounds "i" start c.chunks 1 (Program.body c scale storeFlag ifcond) s ↔
      Safety.rangeSafe M bounds "i" start c.chunks 1 (body c scale storeFlag ifcond) s := by
  induction start using (measure (fun i => c.chunks-i)).wf.induction generalizing s with
  | h cur ih =>
    rw [Safety.rangeSafe_unfold,Safety.rangeSafe_unfold]
    simp only [Nat.one_ne_zero,if_false]
    by_cases hc : cur<c.chunks
    · simp only [hc,if_true,←body_safe,←body_run]
      cases he : run M (Program.body c scale storeFlag ifcond) (s.setReg "i" .nat [] (fun _=>cur)) with
      | none => rfl
      | some t =>
        simp only []
        rw [ih (cur+1) (by simp_wf; omega) t]
    · simp [hc]

def codePrefix (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :=
  (Program.code c scale storeFlag ifcond).take 2

theorem original_quoted (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    Program.kernel c scale storeFlag ifcond =
      .mk ["q","k","v","h","o"] [] (codePrefix c scale storeFlag ifcond ++
        [.forRange "i" 0 c.chunks 1
          (Program.body c scale storeFlag ifcond)]) := rfl

theorem candidate_quoted (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    kernel c scale storeFlag ifcond =
      .mk ["q","k","v","h","o"] [] (.ifThen (.constBool Bool.false) [] :: codePrefix c scale storeFlag ifcond ++
        [.forRange "i" 0 c.chunks 1
          (body c scale storeFlag ifcond)]) := rfl

/-- BT is bound to 32 outside the raw candidate. Its retained static
assertion has no runtime effect, including failure and safety observations. -/
theorem assertion_run {α : Type} [Inhabited α] (M : Algebra α)
    (code : List ComputeStmt) (s : State α) :
    run M (.ifThen (.constBool Bool.false) [] :: code) s = run M code s := by
  rw [run]
  simp only [step,evalOp_unfold,bind,Bool.false_eq_true,↓reduceIte,Option.bind_some]

theorem assertion_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds (.ifThen (.constBool Bool.false) [] :: code) s ↔
      Safety.traceSafeList M bounds code s := by
  simp [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeOp_unfold,
    step,evalOp_unfold]

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (s : State α) :
    exec M (Program.kernel c scale storeFlag ifcond) s =
      exec M (kernel c scale storeFlag ifcond) s := by
  rw [original_quoted,candidate_quoted]
  change run M (_ ++ _) s = run M (.ifThen (.constBool Bool.false) [] :: _) s
  rw [assertion_run]
  change run M (codePrefix c scale storeFlag ifcond ++
    [.forRange "i" 0 c.chunks 1 (Program.body c scale storeFlag ifcond)]) s =
    run M (codePrefix c scale storeFlag ifcond ++
      [.forRange "i" 0 c.chunks 1 (body c scale storeFlag ifcond)]) s
  simp only [run_append]
  apply congrArg (fun f => (run M (codePrefix c scale storeFlag ifcond) s).bind f)
  funext t
  simp only [run,step,evalOp_unfold,bind,Option.bind_some]
  rw [range_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State α) :
    TraceSafe M bounds (Program.kernel c scale storeFlag ifcond) s ↔
      TraceSafe M bounds (kernel c scale storeFlag ifcond) s := by
  rw [original_quoted,candidate_quoted]
  change Safety.traceSafeList M bounds (_ ++ _) s ↔
    Safety.traceSafeList M bounds (.ifThen (.constBool Bool.false) [] :: _) s
  rw [assertion_safe]
  change Safety.traceSafeList M bounds (codePrefix c scale storeFlag ifcond ++
    [.forRange "i" 0 c.chunks 1 (Program.body c scale storeFlag ifcond)]) s ↔
    Safety.traceSafeList M bounds (codePrefix c scale storeFlag ifcond ++
      [.forRange "i" 0 c.chunks 1 (body c scale storeFlag ifcond)]) s
  simp only [Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  simp only [Safety.traceSafeList_cons,Safety.traceSafeList_nil,implies_true,and_true,
    Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,Option.bind_some,true_and]
  exact range_safe M bounds c scale storeFlag ifcond 0 t

theorem invocation_run {α : Type} [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (head : Nat) (s : State (Term α)) :
    (KernelIO.invocation α c scale storeFlag ifcond head).run s =
      (invocation α c scale storeFlag ifcond head).run s := by
  change exec algebra (Program.kernel c scale storeFlag ifcond)
    ((KernelIO.invocation α c scale storeFlag ifcond head).initial s) =
      exec algebra (kernel c scale storeFlag ifcond)
        ((KernelIO.invocation α c scale storeFlag ifcond head).initial s)
  exact kernel_run algebra c scale storeFlag ifcond _

theorem invocation_safe {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (head : Nat) (s : State (Term α)) :
    (KernelIO.invocation α c scale storeFlag ifcond head).Safe bounds s ↔
      (invocation α c scale storeFlag ifcond head).Safe bounds s := by
  change TraceSafe algebra bounds (Program.kernel c scale storeFlag ifcond)
    ((KernelIO.invocation α c scale storeFlag ifcond head).initial s) ↔
      TraceSafe algebra bounds (kernel c scale storeFlag ifcond)
        ((KernelIO.invocation α c scale storeFlag ifcond head).initial s)
  exact kernel_safe algebra bounds c scale storeFlag ifcond _

theorem mapped_run {α : Type} [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (heads : List Nat) (s : State (Term α)) :
    runSchedule (heads.map (KernelIO.invocation α c scale storeFlag ifcond)) s =
      runSchedule (heads.map (invocation α c scale storeFlag ifcond)) s := by
  induction heads generalizing s with
  | nil => rfl
  | cons h hs ih =>
    simp only [List.map_cons,runSchedule,←invocation_run]
    cases he : (KernelIO.invocation α c scale storeFlag ifcond h).run s with
    | none => rfl
    | some t => exact ih t

theorem mapped_safe {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (heads : List Nat)
    (s : State (Term α)) :
    ScheduleSafe bounds (heads.map (KernelIO.invocation α c scale storeFlag ifcond)) s ↔
      ScheduleSafe bounds (heads.map (invocation α c scale storeFlag ifcond)) s := by
  induction heads generalizing s with
  | nil => rfl
  | cons h hs ih =>
    simp only [List.map_cons,ScheduleSafe,←invocation_safe,←invocation_run]
    exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => ih t)

theorem schedule_run {α : Type} [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (s : State (Term α)) :
    runSchedule (KernelIO.schedule α c scale storeFlag ifcond) s =
      runSchedule (schedule α c scale storeFlag ifcond) s := by
  unfold KernelIO.schedule schedule
  split
  · rfl
  · exact mapped_run c scale storeFlag ifcond _ s

theorem schedule_safe {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (s : State (Term α)) :
    ScheduleSafe bounds (KernelIO.schedule α c scale storeFlag ifcond) s ↔
      ScheduleSafe bounds (schedule α c scale storeFlag ifcond) s := by
  unfold KernelIO.schedule schedule
  split
  · rfl
  · exact mapped_safe bounds c scale storeFlag ifcond _ s

theorem equivalent {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    (KernelIO.io ops literal c scale storeFlag ifcond).Equivalent
      (variant ops literal c scale storeFlag ifcond) := by
  intro vals _ bounds bounded initial pinned
  have safe := KernelIO.schedule_safe c scale storeFlag ifcond bounds
    (KernelIO.windows ops literal c scale storeFlag ifcond vals bounds bounded) initial
  refine ⟨safe,(schedule_safe bounds c scale storeFlag ifcond initial).mp safe,
    SharedGrid.result c scale storeFlag ifcond initial,
    SharedGrid.result c scale storeFlag ifcond initial,?_,?_,?_,?_,?_⟩
  · change runSchedule (KernelIO.schedule α c scale storeFlag ifcond) initial = _
    rw [KernelIO.full_run]
    exact SharedGrid.run_success c scale storeFlag ifcond initial
  · change runSchedule (schedule α c scale storeFlag ifcond) initial = _
    rw [←schedule_run,KernelIO.full_run]
    exact SharedGrid.run_success c scale storeFlag ifcond initial
  · intro o j active
    rfl
  · exact KernelIO.frame ops literal c scale storeFlag ifcond vals initial
  · exact KernelIO.frame ops literal c scale storeFlag ifcond vals initial

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Pair
