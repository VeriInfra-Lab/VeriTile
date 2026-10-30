import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedFP
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SafetyInvariant

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SourceSafety
open VeriTile.Triton FP.Structural Kernels SignedExecution
open SignedSafety
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private def leaves (count cols : Nat) : List ComputeStmt :=
  SignedExecution.inner count cols ++ middlePrefix count cols ++ outerPrefix count cols ++ outerSuffix count cols

private theorem leaf_data (count cols : Nat) (st : ComputeStmt)
    (member : st ∈ leaves count cols) : DataOnly st := by
  simp [leaves,SignedExecution.inner,SignedExecution.middle,SignedExecution.outer,
    middlePrefix,outerPrefix,outerSuffix,source,kernel,originalKernel,quotedOriginal,
    ComputeKernel.surfaceBody] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [DataOnly]

private theorem clamp_int (count cap : Nat) (positive : 0<count) (bounded : count≤cap)
    (x : Int) : (min x ((count : Int)-1)).toNat < cap := by
  have := min_le_right x ((count : Int)-1)
  omega

private theorem leaf_safe (M : Algebra α) (count cols : Nat) (positive : 0<count)
    (bounds : RegionName → Nat) (hin : count≤bounds "input") (hout : count≤bounds "output")
    (st : ComputeStmt) (member : st ∈ leaves count cols) (s : State α)
    (context : Context count s) : Safety.stepSafe M bounds st s := by
  rcases context with ⟨hn,hs⟩
  simp [leaves,SignedExecution.inner,SignedExecution.middle,SignedExecution.outer,
    middlePrefix,outerPrefix,outerSuffix,source,kernel,originalKernel,quotedOriginal,
    ComputeKernel.surfaceBody] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.safeAlg_unfold,true_and,and_true]
  all_goals
    cases ha : s.regs .int [2,2] "a_offsets" <;>
    cases hc : s.regs .int [2,2] "c_offsets" <;>
    cases hu : s.regs .bool [2,2] "unique" <;>
    simp [Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOffsets,evalDeltas,
      evalOp_unfold,bop,numericLt,numericLe,natLt,natLe,numeric,Option.bind,
      Broadcast.leftIndex,Broadcast.rightIndex,ha,hc,hu,hn,hs,←min_def_lt,
      clamp_int count _ positive hin,clamp_int count _ positive hout]

private theorem leaf_code (M : Algebra α) (count cols : Nat) (positive : 0<count)
    (bounds : RegionName → Nat) (hin : count≤bounds "input") (hout : count≤bounds "output")
    (code : List ComputeStmt) (subset : ∀ st ∈ code, st ∈ leaves count cols) :
    Code M bounds count code :=
  of_flat M bounds count code (fun st h => leaf_data count cols st (subset st h))
    (fun st h => leaf_safe M count cols positive bounds hin hout st (subset st h))

private theorem middle_code (M : Algebra α) (count cols : Nat) (positive : 0<count)
    (bounds : RegionName → Nat) (hin : count≤bounds "input") (hout : count≤bounds "output") :
    Code M bounds count (SignedExecution.middle count cols) := by
  have innerChecked := leaf_code M count cols positive bounds hin hout (SignedExecution.inner count cols)
    (by intro st h; simp [leaves,h])
  have prefixChecked := leaf_code M count cols positive bounds hin hout (middlePrefix count cols)
    (by intro st h; simp [leaves,h])
  rw [middle_decomposition]
  exact append prefixChecked (cons (SignedSafety.range_two M bounds count "k" (by decide) (by decide) _ innerChecked)
    (nil M bounds count))

private theorem outer_code (M : Algebra α) (count cols : Nat) (positive : 0<count)
    (bounds : RegionName → Nat) (hin : count≤bounds "input") (hout : count≤bounds "output") :
    Code M bounds count (SignedExecution.outer count cols) := by
  have prefixChecked := leaf_code M count cols positive bounds hin hout (outerPrefix count cols)
    (by intro st h; simp [leaves,h])
  have suffixChecked := leaf_code M count cols positive bounds hin hout (outerSuffix count cols)
    (by intro st h; simp [leaves,h])
  rw [outer_decomposition]
  exact append (append prefixChecked (cons (SignedSafety.range_two M bounds count "j" (by decide) (by decide) _
    (middle_code M count cols positive bounds hin hout)) (nil M bounds count))) suffixChecked

private def initBody (candidate : Bool) (count cols : Nat) : List ComputeStmt :=
  (kernel candidate "input" "output" count cols).surfaceBody.take 3
private def restPure (candidate : Bool) (count cols : Nat) : List ComputeStmt :=
  ((kernel candidate "input" "output" count cols).surfaceBody.drop 3).take 7
private def initialized (count cols : Nat) (s : State α) : State α :=
  ((s.setReg "stride_m" .int [] (fun _ => (cols : Int))).setReg "stride_n" .int [] (fun _ => (1 : Int))).setReg
    "n_elements" .int [] (fun _ => (count : Int))

private theorem init_run (M : Algebra α) (candidate : Bool) (count cols : Nat) (s : State α) :
    run M (initBody candidate count cols) s = some (initialized count cols s) := by
  cases candidate <;> simp [initBody,kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,initialized]

private theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (pure : ∀ st ∈ code, ∀ s : State α, Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (pure st (by simp) s)
    intro t _
    exact ih (fun st h => pure st (by simp [h])) t

private theorem init_safe (M : Algebra α) (candidate : Bool) (count cols : Nat)
    (bounds : RegionName → Nat) (s : State α) : Safety.traceSafeList M bounds (initBody candidate count cols) s := by
  apply pure_trace
  intro st member state
  cases candidate
  all_goals
    simp [initBody,kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,
      ComputeKernel.surfaceBody] at member
    rcases member with rfl | rfl | rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

private theorem restPure_code (M : Algebra α) (candidate : Bool) (count cols : Nat)
    (bounds : RegionName → Nat) : Code M bounds count (restPure candidate count cols) := by
  apply of_flat
  · intro st member
    cases candidate
    all_goals
      simp [restPure,kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody] at member
      rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals simp [DataOnly,kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody]
  · intro st member state _
    cases candidate
    all_goals
      simp [restPure,kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody] at member
      rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals simp [kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- Actual casts, initialized runtime arguments, signed clamps, all three loops,
masks and barriers are checked along the actual interpreter successors. -/
theorem source_safe (M : Algebra α) (candidate : Bool) (count cols : Nat) (positive : 0<count)
    (bounds : RegionName → Nat) (hin : count≤bounds "input") (hout : count≤bounds "output") (s : State α) :
    TraceSafe M bounds (kernel candidate "input" "output" count cols) s := by
  have shape : (kernel candidate "input" "output" count cols).surfaceBody =
      initBody candidate count cols ++ (restPure candidate count cols ++
        [.forRangeDyn "i" (.constNat 0) (.constNat 2) (.constNat 1) (SignedExecution.outer count cols)]) := by
    cases candidate <;> rfl
  change Safety.traceSafeList M bounds _ s
  rw [shape,Safety.traceSafeList_append]
  refine ⟨init_safe M candidate count cols bounds s,?_⟩
  intro t ran
  have eq : initialized count cols s = t := Option.some.inj ((init_run M candidate count cols s).symm.trans ran)
  subst t
  apply (append (restPure_code M candidate count cols bounds)
    (cons (SignedSafety.range_two M bounds count "i" (by decide) (by decide) _
      (outer_code M count cols positive bounds hin hout)) (nil M bounds count))).safe
  simp [Context,initialized]

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.SourceSafety
