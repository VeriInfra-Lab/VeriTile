import bench.tritonbench_g.nested_loops_processing.Proofs.TypedFP
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SourceSafety
open VeriTile.Triton FP.Structural Kernels TypedExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem trace_of_steps {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ stmt ∈ code, ∀ s : State α, Safety.stepSafe M bounds stmt s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons stmt rest ih =>
    apply Safety.traceSafeList_cons_intro
    · exact safe stmt List.mem_cons_self s
    · intro next _
      exact ih (fun stmt h => safe stmt (List.mem_cons_of_mem _ h)) next

private theorem range_two_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (body : List ComputeStmt)
    (safe : ∀ s : State α, Safety.traceSafeList M bounds body s) (s : State α) :
    Safety.stepSafe M bounds (.forRange idx 0 2 1 body) s := by
  simp only [Safety.stepSafe_unfold]
  rw [Safety.rangeSafe_unfold]
  simp only [Nat.one_ne_zero,Nat.reduceLT,↓reduceIte]
  refine ⟨safe _,?_⟩
  cases h0 : run M body (s.setReg idx .nat [] (fun _ => 0)) with
  | none => trivial
  | some t =>
    dsimp only
    rw [Safety.rangeSafe_unfold]
    simp only [Nat.one_ne_zero,Nat.reduceAdd,Nat.reduceLT,↓reduceIte]
    refine ⟨safe _,?_⟩
    cases h1 : run M body (t.setReg idx .nat [] (fun _ => 1)) with
    | none => trivial
    | some u => simp [Safety.rangeSafe_unfold]

/-- Clamping supplies actual capacity even before applying the source masks. -/
private theorem clamp_bound (count cap address : Nat) (positive : 0 < count)
    (bounded : count ≤ cap) : min address (count-1) < cap := by
  have := Nat.min_le_right address (count-1)
  omega

theorem inner_safe {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (positive : 0 < count) (bounds : RegionName → Nat)
    (hin : count ≤ bounds "input") (hout : count ≤ bounds "output") (s : State α) :
    Safety.traceSafeList M bounds (inner count cols) s := by
  apply trace_of_steps
  intro stmt member state
  simp only [inner,middle,outer,source,kernel,ComputeKernel.surfaceBody,
    List.getElem?_cons_zero,List.getElem?_cons_succ,Bool.false_eq_true,↓reduceIte,
    List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.safeAlg_unfold,true_and,and_true]
  all_goals
    cases ha : state.regs .nat [2,2] "a_offsets" <;>
    cases hc : state.regs .nat [2,2] "c_offsets" <;>
    cases hu : state.regs .bool [2,2] "unique" <;>
    simp [Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOffsets,evalDeltas,
      evalOp_unfold,FP.Structural.bop,natLt,natLe,numeric,Option.bind,
      Broadcast.leftIndex,Broadcast.rightIndex,ha,hc,hu,← min_def_lt,
      clamp_bound count _ _ positive hin,clamp_bound count _ _ positive hout]

theorem middle_safe {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (positive : 0 < count) (bounds : RegionName → Nat)
    (hin : count ≤ bounds "input") (hout : count ≤ bounds "output") (s : State α) :
    Safety.traceSafeList M bounds (middle count cols) s := by
  apply trace_of_steps
  intro stmt member state
  have shape : middle count cols = middlePrefix count cols ++
      [.forRange "k" 0 2 1 (inner count cols)] := middle_decomposition count cols
  rw [shape] at member
  rcases List.mem_append.mp member with pure | loop
  · simp only [middlePrefix,middle,outer,source,kernel,ComputeKernel.surfaceBody,
      List.getElem?_cons_zero,List.getElem?_cons_succ,Bool.false_eq_true,↓reduceIte,
      List.take_succ_cons,List.take_zero,List.mem_cons,List.not_mem_nil,or_false] at pure
    rcases pure with rfl | rfl
    all_goals
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
        Safety.safeAlg_unfold,true_and,and_true]
    all_goals
      cases ha : state.regs .nat [2,2] "a_offsets" <;>
      simp [Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOffsets,evalDeltas,
        evalOp_unfold,FP.Structural.bop,natLt,natLe,numeric,Option.bind,
        Broadcast.leftIndex,Broadcast.rightIndex,ha,← min_def_lt,
        clamp_bound count _ _ positive hin]
  · have eq := List.mem_singleton.mp loop
    subst stmt
    exact range_two_safe M bounds "k" _ (inner_safe M count cols positive bounds hin hout) state

theorem outer_safe {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (positive : 0 < count) (bounds : RegionName → Nat)
    (hin : count ≤ bounds "input") (hout : count ≤ bounds "output") (s : State α) :
    Safety.traceSafeList M bounds (outer count cols) s := by
  apply trace_of_steps
  intro stmt member state
  have shape : outer count cols =
      (outer count cols).take 1 ++ [.forRange "j" 0 2 1 (middle count cols)] ++
      (outer count cols).drop 2 := outer_decomposition count cols
  rw [shape] at member
  rcases List.mem_append.mp member with head | tail
  · rcases List.mem_append.mp head with load | loop
    · simp only [outer,source,kernel,ComputeKernel.surfaceBody,
        List.getElem?_cons_zero,List.getElem?_cons_succ,Bool.false_eq_true,↓reduceIte,
        List.take_succ_cons,List.take_zero,List.mem_singleton] at load
      subst stmt
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
        Safety.safeAlg_unfold,true_and,and_true]
      cases ha : state.regs .nat [2,2] "a_offsets" <;>
      simp [Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOffsets,evalDeltas,
        evalOp_unfold,FP.Structural.bop,natLt,natLe,numeric,Option.bind,
        Broadcast.leftIndex,Broadcast.rightIndex,ha,← min_def_lt,
        clamp_bound count _ _ positive hin]
    · have eq := List.mem_singleton.mp loop
      subst stmt
      exact range_two_safe M bounds "j" _ (middle_safe M count cols positive bounds hin hout) state
  · simp only [outer,source,kernel,ComputeKernel.surfaceBody,
      List.getElem?_cons_zero,List.getElem?_cons_succ,Bool.false_eq_true,↓reduceIte,
      List.drop_succ_cons,List.drop_zero,List.mem_singleton] at tail
    subst stmt
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.safeAlg_unfold]

/-- Both genuine integer-source variants are safe through all nested-loop
states. The positive allocation bound validates each clamped actual address;
the source masks and barriers are retained rather than erased. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (candidate : Bool)
    (count cols : Nat) (positive : 0 < count) (bounds : RegionName → Nat)
    (hin : count ≤ bounds "input") (hout : count ≤ bounds "output") (s : State α) :
    TraceSafe M bounds (kernel candidate "input" "output" count cols) s := by
  apply trace_of_steps
  intro stmt member state
  have outerLoop := range_two_safe M bounds "i" _ (outer_safe M count cols positive bounds hin hout) state
  cases candidate
  all_goals
    simp only [kernel,ComputeKernel.surfaceBody,Bool.false_eq_true,↓reduceIte,
      List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | exact outerLoop
    | simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.safeAlg_unfold]

#axiomsClean inner_safe
#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.SourceSafety
