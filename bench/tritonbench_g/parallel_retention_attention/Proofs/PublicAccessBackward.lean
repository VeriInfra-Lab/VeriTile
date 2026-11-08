import bench.tritonbench_g.parallel_retention_attention.Proofs.CallerSafetyParts
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBackwardParts
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessQueryTail
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardComposition

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicBackwardAccess
open VeriTile.Triton FP.Structural FPStages
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (DotPrimitive withDot BackwardConfig BackwardAccumulator descendingCount descendingStart)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem first_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (start : Nat) (bound : start≤descendingCount c) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hb : Bounds bounds c) :
    Safety.rangeIntSafe (withDot M D) bounds "__rev_i"
      (DescendingRange.lower c+(start : Int)*(c.BTS : Int)) (DescendingRange.upper c)
      c.BTS (backwardFirstBody c) t := by
  by_cases zero : c.BTS=0
  · simp [Safety.rangeIntSafe,zero]
  have positive : 0<c.BTS:=by omega
  have clock (n : Nat) (live : n<descendingCount c) :
      DescendingRange.lower c+(n : Int)*(c.BTS : Int)=-(descendingStart c n : Int) := by
    have equal:=congrArg Neg.neg (DescendingRange.source_index c positive n live)
    simpa only [neg_neg] using equal
  apply VeriTile.Bench.Repairs.ParallelAttentionKernel.DescendingRange.clock_safe
    (withDot M D) bounds "__rev_i" (backwardFirstBody c)
    (DescendingRange.lower c) (DescendingRange.upper c) c.BTS (descendingCount c)
    (DescendingRange.source_active c positive) (fun _ u=>∃ a,BackwardReady M c s u a)
    _ _ start bound t ⟨acc,h⟩
  · intro n live u hu
    obtain ⟨a,ha⟩:=hu
    rw [clock n live]
    exact first_body_safe M D bounds c s _ (descendingStart c n) a
      (backwardReady_j M c s u a ha (-(descendingStart c n : Int))) hb
      (by simp [State.setReg]) (hb.indices.descending_lt n)
  · intro n live u hu v ran
    obtain ⟨a,ha⟩:=hu
    rw [clock n live] at ran
    obtain ⟨v',finished,ready⟩:=backward_first_body_run M D c s _ (descendingStart c n) a
      (backwardReady_j M c s u a ha (-(descendingStart c n : Int))) (by simp [State.setReg])
      hb.indices.toHelperIndexBounds (hb.indices.descending_lt n)
    rw [ran] at finished
    cases finished
    exact ⟨_,ready⟩

theorem second_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (start stop : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc) (hb : Bounds bounds c)
    (stopBound : stop≤2147483648) :
    Safety.rangeIntSafe (withDot M D) bounds "i" (start : Int) (stop : Int) c.BTS (backwardSecondBody c) t := by
  apply rangeInt_safe (withDot M D) bounds "i" (backwardSecondBody c)
    (fun n u=>∃ a,BackwardDiagonalReady M c s u n a) start stop c.BTS _ _ t ⟨acc,h⟩
  · intro n live u hu
    obtain ⟨a,ha⟩:=hu
    exact second_body_safe M D bounds c s _ n a (backwardDiagonalReady_i M c s u n a ha n)
      hb (by simp [State.setReg]) (by omega)
  · intro n live u hu v ran
    obtain ⟨a,ha⟩:=hu
    obtain ⟨v',finished,ready⟩:=backward_second_body_run M D c s _ n a
      (backwardDiagonalReady_i M c s u n a ha n) (by simp [State.setReg])
      hb.indices.toHelperIndexBounds (by omega)
    rw [ran] at finished
    cases finished
    exact ⟨_,ready⟩

theorem first_range_statement_safe {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α) (cap : RegionName → Nat)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (positive : 0<c.BTS) :
    Safety.stepSafe (withDot M D) cap (backwardFirstRange c) t ↔
      Safety.rangeIntSafe (withDot M D) cap "__rev_i"
        (DescendingRange.lower c)
        (DescendingRange.upper c)
        c.BTS (backwardFirstBody c) t := by
  have offset : ((c.T+c.BTS-1 : Nat) : Int)=(c.T : Int)+(c.BTS : Int)-1 := by omega
  have nonnegative : ¬((c.T : Int)+(c.BTS : Int)-1<0) := by omega
  simp [backwardFirstRange,backwardFirstBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind,h.argument13,h.argument1,
    integralFloorDiv,numericLt,natLt,nonnegative,
    DescendingRange.lower,
    DescendingRange.upper,
    Int.natCast_mul,Int.natCast_add,Int.natCast_ediv,offset]
  simp only [←Int.add_sub_assoc,nonnegative,if_false]
  have same : (c.T : Int)+(c.BTS : Int)-1=(c.T+c.BTS-1 : Nat) := by omega
  simp only [same]
  simp only [sub_eq_add_neg,add_comm]

theorem second_range_statement_safe {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α) (cap : RegionName → Nat)
    (c : BackwardConfig) (s t : State α) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    Safety.stepSafe (withDot M D) cap (backwardSecondRange c) t ↔
      Safety.rangeIntSafe (withDot M D) cap "i" ((c.i_c*c.BTL : Nat) : Int) (((c.i_c+1)*c.BTL : Nat) : Int)
        c.BTS (backwardSecondBody c) t := by
  simp [backwardSecondRange,backwardSecondBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind,h.argument1,Int.natCast_mul,Int.natCast_add]

theorem key_value_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s : State α)
    (hb : Bounds bounds c) (positive : 0<c.BTS) :
    TraceSafe (withDot M D) bounds (backwardKernel c) s := by
  obtain ⟨initial,hp,hi⟩ := backward_prefix_run M D c s hb.indices.toHelperIndexBounds
  obtain ⟨first,hr,hf⟩ := backward_first_range_run M D c s initial 0 (by omega) _ hi hb.indices
  simp only [Nat.cast_zero,zero_mul,add_zero] at hr
  have stopBound : (c.i_c+1)*c.BTL≤2147483648 := by
    have := hb.indices.time
    nlinarith
  obtain ⟨diagonal,hm,hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second,hs,hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL)
    ((c.i_c+1)*c.BTL) _ hd hb.indices.toHelperIndexBounds stopBound
  unfold TraceSafe
  rw [backward_source_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨prefix_safe M D bounds c s hb,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  apply Safety.traceSafeList_cons_of_step (t := first)
  · apply (first_range_statement_safe M D bounds c s initial _ hi positive).mpr
    simpa only [Nat.cast_zero,zero_mul,add_zero] using first_range_safe M D bounds c s initial 0 (by omega) _ hi hb
  · simpa only [backward_first_range_statement M D c s initial _ hi positive] using hr
  · rw [Safety.traceSafeList_append]
    refine ⟨middle_safe M D bounds c s first _ hf,?_⟩
    intro u hu
    rw [hm] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t := second)
    · exact (second_range_statement_safe M D bounds c s diagonal _ hd.toBackwardReady).mpr
        (second_range_safe M D bounds c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd hb stopBound)
    · simpa only [backward_second_range_statement M D c s diagonal _ hd.toBackwardReady] using hs
    · exact suffix_safe M D bounds c s second _ hz hb

theorem key_value_tail_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s captured : State α)
    (capturedArgs : BackwardArgumentReady c s captured)
    (hb : Bounds bounds c) (positive : 0<c.BTS) :
    Safety.traceSafeList (withDot M D) bounds (backwardTail c) captured := by
  obtain ⟨initial,hp,hi⟩ := backward_numerical_run M D c s captured capturedArgs hb.indices.toHelperIndexBounds
  obtain ⟨first,hr,hf⟩ := backward_first_range_run M D c s initial 0 (by omega) _ hi hb.indices
  simp only [Nat.cast_zero,zero_mul,add_zero] at hr
  have stopBound : (c.i_c+1)*c.BTL≤2147483648 := by
    have := hb.indices.time
    nlinarith
  obtain ⟨diagonal,hm,hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second,hs,hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL)
    ((c.i_c+1)*c.BTL) _ hd hb.indices.toHelperIndexBounds stopBound
  rw [backward_tail_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨numerical_safe M D bounds c s captured capturedArgs hb,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  apply Safety.traceSafeList_cons_of_step (t := first)
  · apply (first_range_statement_safe M D bounds c s initial _ hi positive).mpr
    simpa only [Nat.cast_zero,zero_mul,add_zero] using first_range_safe M D bounds c s initial 0 (by omega) _ hi hb
  · simpa only [backward_first_range_statement M D c s initial _ hi positive] using hr
  · rw [Safety.traceSafeList_append]
    refine ⟨middle_safe M D bounds c s first _ hf,?_⟩
    intro u hu
    rw [hm] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t := second)
    · exact (second_range_statement_safe M D bounds c s diagonal _ hd.toBackwardReady).mpr
        (second_range_safe M D bounds c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd hb stopBound)
    · simpa only [backward_second_range_statement M D c s diagonal _ hd.toBackwardReady] using hs
    · exact suffix_safe M D bounds c s second _ hz hb

/-- Safety of the actual pasted parent, including parameter rebinding, PID
prologue, both helper calls, the barrier and all three output stores. -/
theorem backward_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s : State α)
    (hq : PublicQueryAccess.Bounds bounds c dq) (hkv : Bounds bounds c)
    (indices : Caller.Indices c s) (positive : 0<c.BTS) :
    TraceSafe (withDot M D) bounds (FPBackward.source c dq) s := by
  obtain ⟨decoded,hp,hd,registers⟩:=Caller.prologue_run (withDot M D) c dq s indices
  obtain ⟨queryArgs,hqa,hqargs⟩:=Caller.query_arguments_run (withDot M D) c dq decoded registers
  obtain ⟨query,hqrun,hqm,hqu,frame⟩:=FPQuery.tail_run M D c dq decoded queryArgs hqargs hq.indices
  obtain ⟨keyArgs,hka,hkargs⟩:=Caller.key_arguments_run (withDot M D) c dq query
    (Caller.registers_after_query registers frame)
  change Safety.traceSafeList (withDot M D) bounds (Caller.kernel c dq).surfaceBody s
  rw [Caller.source_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨Caller.prologue_safe (withDot M D) bounds c dq s,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  rw [Safety.traceSafeList_append]
  refine ⟨Caller.query_arguments_safe (withDot M D) bounds c dq decoded,?_⟩
  intro u hu
  rw [hqa] at hu
  cases hu
  rw [Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [Caller.query_tail_eq]
    exact PublicQueryAccess.tail_safe M D bounds c dq decoded queryArgs hqargs hq
  · intro u hu
    rw [Caller.query_tail_eq,hqrun] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t:=query)
    · simp [Safety.stepSafe_unfold]
    · simp [step]
    · rw [Safety.traceSafeList_append]
      refine ⟨Caller.key_arguments_safe (withDot M D) bounds c dq query,?_⟩
      intro u hu
      rw [hka] at hu
      cases hu
      rw [Caller.key_tail_eq]
      exact key_value_tail_safe M D bounds c query keyArgs hkargs hkv positive

#axiomsClean key_value_safe
#axiomsClean key_value_tail_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicBackwardAccess
