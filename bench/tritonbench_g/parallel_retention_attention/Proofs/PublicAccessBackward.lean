import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBackwardParts
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessQuery
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardComposition

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicBackwardAccess
open VeriTile.Triton FP.Structural FPStages
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (DotPrimitive withDot BackwardConfig BackwardAccumulator descendingCount)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem first_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (start stop : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hb : Bounds bounds c) :
    Safety.rangeSafe (withDot M D) bounds "j" start stop 1 (backwardFirstBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "j" stop 1 _
    (fun _ u => ∃ a, BackwardReady M c s u a) _ _ start t ⟨acc,h⟩
  · intro n hn u hu
    obtain ⟨a,ha⟩ := hu
    exact first_body_safe M D bounds c s _ n a (backwardReady_j M c s u a ha n) hb
      (by simp [State.setReg])
  · intro n hn u hu v hv
    obtain ⟨a,ha⟩ := hu
    obtain ⟨v',run,ready⟩ := backward_first_body_run M D c s _ n a
      (backwardReady_j M c s u a ha n) (by simp [State.setReg])
    rw [hv] at run
    cases run
    exact ⟨_,ready⟩

theorem second_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (start stop : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc) (hb : Bounds bounds c) :
    Safety.rangeSafe (withDot M D) bounds "i" start stop c.BTS (backwardSecondBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "i" stop c.BTS _
    (fun n u => ∃ a, BackwardDiagonalReady M c s u n a) _ _ start t ⟨acc,h⟩
  · intro n hn u hu
    obtain ⟨a,ha⟩ := hu
    exact second_body_safe M D bounds c s _ n a (backwardDiagonalReady_i M c s u n a ha n) hb
      (by simp [State.setReg])
  · intro n hn u hu v hv
    obtain ⟨a,ha⟩ := hu
    obtain ⟨v',run,ready⟩ := backward_second_body_run M D c s _ n a
      (backwardDiagonalReady_i M c s u n a ha n) (by simp [State.setReg])
    rw [hv] at run
    cases run
    exact ⟨_,ready⟩

theorem key_value_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s : State α)
    (hb : Bounds bounds c) :
    TraceSafe (withDot M D) bounds (backwardKernel c) s := by
  obtain ⟨initial,hp,hi⟩ := backward_prefix_run M D c s
  obtain ⟨first,hr,hf⟩ := backward_first_range_run M D c s initial 0 (descendingCount c) _ hi
  obtain ⟨diagonal,hm,hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second,hs,hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL)
    ((c.i_c+1)*c.BTL) _ hd
  unfold TraceSafe
  rw [backward_source_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨prefix_safe M D bounds c s hb,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  apply Safety.traceSafeList_cons_of_step (t := first)
  · simpa [backwardFirstRange,backwardFirstBody,backwardKernel,backwardSource,
      ComputeKernel.surfaceBody,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,Option.bind,descendingCount] using
      first_range_safe M D bounds c s initial 0 (descendingCount c) _ hi hb
  · simpa only [backward_first_range_statement] using hr
  · rw [Safety.traceSafeList_append]
    refine ⟨middle_safe M D bounds c s first _ hf,?_⟩
    intro u hu
    rw [hm] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t := second)
    · simpa [backwardSecondRange,backwardSecondBody,backwardKernel,backwardSource,
        ComputeKernel.surfaceBody,Safety.stepSafe_unfold,Safety.safeOp_unfold,
        evalOp_unfold,numeric,bop,Option.bind] using
        second_range_safe M D bounds c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd hb
    · simpa only [backward_second_range_statement] using hs
    · exact suffix_safe M D bounds c s second _ hz hb

/-- The original backward shell includes DQ and both DK/DV outputs. Bounds do
not depend on values or initial memory, so aliases remain admitted here. -/
theorem backward_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s : State α)
    (hq : PublicQueryAccess.Bounds bounds c dq) (hkv : Bounds bounds c) :
    TraceSafe (withDot M D) bounds (FPBackward.source c dq) s := by
  change Safety.traceSafeList (withDot M D) bounds
    ((FPQuery.source c dq).surfaceBody ++ [.effectMarker "tl.debug_barrier"] ++
      (backwardKernel c).surfaceBody) s
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨PublicQueryAccess.query_safe M D bounds c dq s hq,?_⟩
  intro t ht
  apply Safety.traceSafeList_cons_of_step (t := t)
  · simp [Safety.stepSafe_unfold]
  · simp [step]
  · exact key_value_safe M D bounds c t hkv

#axiomsClean key_value_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicBackwardAccess
