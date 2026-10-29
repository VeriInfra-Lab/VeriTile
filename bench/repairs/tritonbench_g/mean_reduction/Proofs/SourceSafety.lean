import bench.repairs.tritonbench_g.mean_reduction.Kernels
import bench.tritonbench_g.mean_reduction.Proofs.WholeFPValue
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MeanReduction.Access
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MeanReduction WholeFPValue
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem pure_list_safe {α : Type} [Inhabited α] (A : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt)
    (pure : ∀ st ∈ code, ∀ s : State α, Safety.stepSafe A bounds st s) (s : State α) :
    Safety.traceSafeList A bounds code s := by
  induction code generalizing s with
  | nil => simp only [Safety.traceSafeList_unfold]
  | cons st rest ih =>
    rw [Safety.traceSafeList_unfold]
    refine ⟨pure st List.mem_cons_self s, ?_⟩
    cases step A st s with
    | none => trivial
    | some t => exact ih (fun q member => pure q (List.mem_cons_of_mem st member)) t

private theorem prefix_safe {α : Type} [Inhabited α] (A : Algebra α)
    (X Mean : RegionName) (M N : Nat) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList A bounds (codePrefix X Mean M N) s := by
  apply pure_list_safe
  intro st member t
  simp only [codePrefix,source,typedKernel,ComputeKernel.surfaceBody,List.take,
    List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- Both row and column source masks guard the actual streaming pointer. -/
theorem body_safe {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α)
    (s : State α) (ctx : Context X Mean M N pid memory acc s)
    (bounds : RegionName → Nat) (input : M*N ≤ bounds X) :
    Safety.traceSafeList A bounds (body N) (s.setReg "off" .nat [] (fun _ => off)) := by
  rcases ctx with ⟨hX,hMean,hRow,hAcc,hMem⟩
  simp [body,source,typedKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,
    Region.cast,Option.bind,ComputeDType.eraseDType,ofFloat,toFloat,hX,hMean,hRow,hAcc,hMem]
  intro i j row col
  apply Nat.lt_of_lt_of_le _ input
  calc
    (pid*8+i.val)*N+(off+j.val) < (pid*8+i.val)*N+N := Nat.add_lt_add_left col _
    _ = (pid*8+i.val+1)*N := by simp [Nat.add_mul]
    _ ≤ M*N := Nat.mul_le_mul_right N row

/-- The final store is guarded by the original row mask. -/
theorem suffix_safe {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N pid : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α)
    (s : State α) (ctx : Context X Mean M N pid memory acc s)
    (bounds : RegionName → Nat) (output : M ≤ bounds Mean) :
    Safety.traceSafeList A bounds (suffix N) s := by
  rcases ctx with ⟨hX,hMean,hRow,hAcc,hMem⟩
  simp [suffix,source,typedKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,
    Region.cast,Option.bind,ComputeDType.eraseDType,ofFloat,toFloat,hMean,hRow,hAcc,store]
  intro i active
  exact Nat.lt_of_lt_of_le active output

/-- Access safety for the actual full source range, with no positivity
restriction on N and no numerical assumption on the primitive backend. -/
theorem source_safe {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (s : State α) (bounds : RegionName → Nat)
    (input : M*N ≤ bounds X) (output : M ≤ bounds Mean) :
    TraceSafe A bounds (source X Mean M N) s := by
  change Safety.traceSafeList A bounds (source X Mean M N).surfaceBody s
  rw [complete_source,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prefix_safe A X Mean M N s bounds, ?_⟩
  intro t ht
  have eq := Option.some.inj ((prefix_runs A X Mean M N s).symm.trans ht)
  subst t
  rw [List.singleton_append,Safety.traceSafeList_unfold]
  refine ⟨?_, ?_⟩
  · rw [Safety.stepSafe_unfold]
    apply Safety.rangeSafe_of_invariant A bounds "off" N 8 (body N)
      (fun _ t => ∃ acc, Context X Mean M N (s.pids 0) s.mem acc t)
    · rintro off _ t ⟨acc,ctx⟩
      exact body_safe A X Mean M N (s.pids 0) off s.mem acc t ctx bounds input
    · rintro off _ t ⟨acc,ctx⟩ u hu
      have eq := Option.some.inj ((body_runs A X Mean M N (s.pids 0) off s.mem acc t ctx).symm.trans hu)
      subst u
      exact ⟨_, body_context A X Mean M N (s.pids 0) off s.mem acc t ctx⟩
    · exact ⟨_,prefix_context A X Mean M N s⟩
  · obtain ⟨u,hu,ctx⟩ := range_runs A X Mean M N (s.pids 0) s.mem 0
      (fun _ => A.literal (some .fp32) .real 0) _ (prefix_context A X Mean M N s)
    simp only [step,hu]
    exact suffix_safe A X Mean M N (s.pids 0) s.mem _ u ctx bounds output

/-- The candidate changes only the first pure integer multiplication. Its
actual successor is identical before every load, loop iteration and store. -/
theorem candidate_safe {α : Type} [Inhabited α] (A : Algebra α)
    (M N : Nat) (s : State α) (bounds : RegionName → Nat)
    (input : M*N ≤ bounds "X") (output : M ≤ bounds "Mean") :
    TraceSafe A bounds (indexCommutedKernel M N) s := by
  have original := source_safe A "X" "Mean" M N s bounds input output
  change Safety.traceSafeList A bounds (source "X" "Mean" M N).surfaceBody s at original
  rw [←List.take_append_drop 1 (source "X" "Mean" M N).surfaceBody,
    Safety.traceSafeList_append] at original
  change Safety.traceSafeList A bounds (indexCommutedKernel M N).surfaceBody s
  rw [←List.take_append_drop 1 (indexCommutedKernel M N).surfaceBody,
    Safety.traceSafeList_append]
  constructor
  · simp [indexCommutedKernel,indexCommutedSource,typedKernel,ComputeKernel.surfaceBody,
      List.take,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,step,evalExpr,evalOp_unfold,numeric,
      Option.bind,FP.Structural.bop]
  · intro t ht
    have prefixEq : run A ((indexCommutedKernel M N).surfaceBody.take 1) s=
        run A ((source "X" "Mean" M N).surfaceBody.take 1) s := by
      simp [indexCommutedKernel,indexCommutedSource,source,typedKernel,ComputeKernel.surfaceBody,
        List.take,run,step,evalExpr,evalOp_unfold,numeric,Option.bind]
      congr 1
      funext i
      simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
    exact original.2 t (prefixEq.symm.trans ht)

theorem variant_safe {α : Type} [Inhabited α] (A : Algebra α) (variant : Bool)
    (M N : Nat) (s : State α) (bounds : RegionName → Nat)
    (input : M*N ≤ bounds "X") (output : M ≤ bounds "Mean") :
    TraceSafe A bounds (if variant then indexCommutedKernel M N else kernel M N) s := by
  cases variant
  · exact source_safe A "X" "Mean" M N s bounds input output
  · exact candidate_safe A M N s bounds input output

#axiomsClean source_safe
#axiomsClean variant_safe
end
end VeriTile.Bench.Repairs.MeanReduction.Access
