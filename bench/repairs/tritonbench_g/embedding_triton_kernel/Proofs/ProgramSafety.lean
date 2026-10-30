import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.StructuralAccess

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.StructuralAccess
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem contract_at (zero : α) (p : Parameters) (bounds : RegionBounds)
    (initial s : State α) (k : Nat) (inv : Invariant zero p initial k s)
    (h : AccessContract p (initial.pids 0) (fun r => (initial.mem p.ids r).read .int) bounds) :
    AccessContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds := by
  have ids : ∀ i,s.mem p.ids i=initial.mem p.ids i :=
    fun i => inv.frame _ _ (Or.inl h.ids_distinct)
  simpa only [inv.pids,ids] using h

/-- Safety of all remaining actual loop iterations, using the existing memory
invariant only to preserve the input metadata and launch coordinates. -/
theorem range_safe (rest : Algebra α) (zero : α) (p : Parameters) (bounds : RegionBounds)
    (initial s : State α) (k : Nat) (inv : Invariant zero p initial k s)
    (h : AccessContract p (initial.pids 0) (fun r => (initial.mem p.ids r).read .int) bounds) :
    Safety.rangeSafe (backend rest zero) bounds "start_nn" k 64 1 (body .int32 p) s := by
  rw [Safety.rangeSafe_unfold]
  simp only [Nat.one_ne_zero,if_false]
  split_ifs with hk
  · refine ⟨chunk_safe rest zero p bounds s k hk inv.context (contract_at zero p bounds initial s k inv h),?_⟩
    rw [(chunk rest zero .int32 p s k inv.context).property]
    exact range_safe rest zero p bounds initial _ (k+1)
      (invariant_step rest zero .int32 p
        ⟨h.ids_distinct,h.weight_distinct,h.width_covers,h.stride_covers⟩ initial s k hk inv) h
termination_by 64-k

private theorem prefix_safe (rest : Algebra α) (zero : α) (p : Parameters)
    (bounds : RegionBounds) (s : State α) :
    Safety.traceSafeList (backend rest zero) bounds (prefixCode .int32 p) s := by
  simp [prefixCode,source,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,
    ComputeDType.eraseDType,State.setReg]

/-- The complete quoted source's active loads/stores are backed. The proof
follows all 64 loop iterations through their actual execution states. -/
theorem source_safe_of_access (rest : Algebra α) (zero : α) (p : Parameters)
    (bounds : RegionBounds) (s : State α)
    (h : AccessContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    FP.Structural.TraceSafe (backend rest zero) bounds (source .int32 p) s := by
  obtain ⟨context,pids,mem⟩ := prefix_properties rest zero .int32 p s
  have base : Invariant zero p s 0 (prefixRun rest zero .int32 p s).val := by
    refine ⟨context,pids,by omega,?_,?_⟩
    · intro r off _; exact congrFun (congrFun mem r) off
    · intro a ha; omega
  unfold FP.Structural.TraceSafe
  rw [decomposition]
  apply (Safety.traceSafeList_append ..).mpr
  refine ⟨prefix_safe rest zero p bounds s,?_⟩
  intro t ran
  have same : t=(prefixRun rest zero .int32 p s).val :=
    Option.some.inj (ran.symm.trans (prefixRun rest zero .int32 p s).property)
  subst t
  apply Safety.traceSafeList_cons_intro
  · simpa only [Safety.stepSafe_unfold] using range_safe rest zero p bounds s _ 0 base h
  · intro _ _; exact Safety.traceSafeList_nil _ _ _

/-- Backward-compatible application of the original kernel memory contract. -/
theorem source_safe (rest : Algebra α) (zero : α) (p : Parameters)
    (bounds : RegionBounds) (s : State α)
    (h : KernelContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    FP.Structural.TraceSafe (backend rest zero) bounds (source .int32 p) s :=
  source_safe_of_access rest zero p bounds s (AccessContract.ofKernel h)

/-- The retained candidate differs only in a pure prefix multiplication.
Its actual access trace begins from the same prefix state. -/
theorem candidate_safe_of_access (rest : Algebra α) (zero : α) (p : Parameters)
    (bounds : RegionBounds) (s : State α)
    (h : AccessContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    FP.Structural.TraceSafe (backend rest zero) bounds (indexCommutedSource .int32 p) s := by
  let M := backend rest zero
  have first : run M ((source .int32 p).surfaceBody.take 1) s =
      run M ((indexCommutedSource .int32 p).surfaceBody.take 1) s := by
    simp [source,indexCommutedSource,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalOp_unfold,numeric,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    exact Int.mul_comm _ _
  have safe := source_safe_of_access rest zero p bounds s h
  change Safety.traceSafeList M bounds (source .int32 p).surfaceBody s at safe
  rw [←List.take_append_drop 1 (source .int32 p).surfaceBody] at safe
  have sourceParts := (Safety.traceSafeList_append ..).mp safe
  change Safety.traceSafeList M bounds (indexCommutedSource .int32 p).surfaceBody s
  rw [←List.take_append_drop 1 (indexCommutedSource .int32 p).surfaceBody]
  apply (Safety.traceSafeList_append ..).mpr
  constructor
  · simp [source,indexCommutedSource,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,step,evalExpr,evalComputeOp,
      evalOp_unfold,numeric,ComputeDType.eraseDType,Option.bind]
  · intro t ran
    exact sourceParts.2 t (first.trans ran)

#axiomsClean source_safe_of_access
#axiomsClean candidate_safe_of_access

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.StructuralAccess
