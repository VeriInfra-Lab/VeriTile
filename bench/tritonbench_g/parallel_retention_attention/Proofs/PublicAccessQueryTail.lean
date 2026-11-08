import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessQuery
import bench.tritonbench_g.parallel_retention_attention.Proofs.QueryTail
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess
open VeriTile.Triton FP.Structural FPQuery
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig)
noncomputable section
set_option maxHeartbeats 350000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem tail_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s captured : State α)
    (capturedArgs : ArgumentReady c s captured)
    (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (tailCode c dq) captured := by
  obtain ⟨initial,hp,hi⟩ := numerical_run M D c dq s captured capturedArgs hb.indices
  obtain ⟨first,hr,last,hf⟩ := first_range_run M D c s initial 0 (c.i_c*c.BTL) _ hi
  obtain ⟨diagonal,hm,hd⟩ := middle_run M D c dq s first last _ hf hb.indices
  obtain ⟨second,hs,finish,hz⟩ := second_range_run M D c s diagonal (c.i_c*c.BTL)
    ((c.i_c+1)*c.BTL) _ hd hb.indices.lanes
  have hrZero : rangeInt (withDot M D) "_" 0 ((c.i_c*c.BTL : Nat) : Int) c.BTS (firstBody c) initial=some first := by
    simpa only [Int.natCast_zero] using hr
  rw [tail_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨numerical_safe M D bounds c dq s captured capturedArgs hb,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  apply Safety.traceSafeList_cons_of_step (t := first)
  · simpa [firstRange,source,ComputeKernel.surfaceBody,firstBody,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,Option.bind,hi.argument1,Int.natCast_mul] using
      first_range_safe M D bounds c dq s initial 0 (c.i_c*c.BTL) _ hi hb
  · simpa only [first_range_statement M D c initial hi.argument1] using hrZero
  · rw [Safety.traceSafeList_append]
    refine ⟨middle_safe M D bounds c dq s first last _ hf,?_⟩
    intro u hu
    rw [hm] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t := second)
    · simpa [secondRange,source,ComputeKernel.surfaceBody,secondBody,Safety.stepSafe_unfold,Safety.safeOp_unfold,
        evalOp_unfold,numeric,bop,Option.bind,hd.argument1,Int.natCast_mul,Int.natCast_add] using
        second_range_safe M D bounds c dq s diagonal (c.i_c*c.BTL)
          ((c.i_c+1)*c.BTL) _ hd hb
    · simpa only [second_range_statement M D c diagonal hd.argument1] using hs
    · exact suffix_safe M D bounds c dq s second finish _ hz hb


#axiomsClean tail_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess
