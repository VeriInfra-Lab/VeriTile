import bench.repairs.tritonbench_g.fused_recurrent_retention.Misc.TensorBridge
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Division
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Reader
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

def unitShape : Shape := ⟨1,1,1,1,1⟩
def tensor (id : Nat) : TensorObject :=
  ⟨id,⟨id,0,1,0,![1,1,1,1]⟩,![1,1,1,1],.fp32,.cuda,Bool.false,Bool.false,Bool.true⟩
def args (seed final : Bool) : ForwardArguments :=
  ⟨tensor 1,tensor 2,tensor 3,if seed then some (tensor 4) else none,20,final⟩

theorem unit_admitted : unitShape.Admitted := by
  norm_num [unitShape,Shape.Admitted,Shape.BK,Shape.BV]

private def safeDecidable (e : IExpr) : Decidable e.Safe := by
  induction e with
  | atom x => unfold IExpr.Safe signed32; infer_instance
  | add a b ha hb => letI := ha; letI := hb; unfold IExpr.Safe signed32; infer_instance
  | sub a b ha hb => letI := ha; letI := hb; unfold IExpr.Safe signed32; infer_instance
  | mul a b ha hb => letI := ha; letI := hb; unfold IExpr.Safe signed32; infer_instance
local instance (e : IExpr) : Decidable e.Safe := safeDecidable e

theorem unit_bounds (phase : SourcePhase) (seed final : Bool) :
    SourceAddress.Bounds phase seed final unitShape (args seed final).buffers := by
  cases phase <;> cases seed <;> cases final
  all_goals constructor
  all_goals first
    | solve | norm_num [unitShape,Shape.BK,Shape.BV,Shape.NK,Shape.NV,Shape.BH,tileCount,signed32]
    | solve | intro i; fin_cases i <;> norm_num [args,tensor,ForwardArguments.buffers,ForwardArguments.saved,
        SavedContext.buffers,detach,signed32]
    | solve | intro hp i; fin_cases i <;> norm_num [args,tensor,ForwardArguments.buffers,ForwardArguments.saved,
        SavedContext.buffers,detach,signed32] at *
    | solve
      | intro iv hiv ik hik h hh lk hlk lv hlv
        have hiv0 : iv=0 := by simpa [unitShape,Shape.NV,Shape.BV,tileCount] using hiv
        have hik0 : ik=0 := by simpa [unitShape,Shape.NK,Shape.BK,tileCount] using hik
        have hh0 : h=0 := by simpa [unitShape,Shape.BH] using hh
        have hlk0 : lk=0 := by simpa [unitShape,Shape.BK] using hlk
        have hlv0 : lv=0 := by simpa [unitShape,Shape.BV] using hlv
        subst iv; subst ik; subst h; subst lk; subst lv
        norm_num [SourceAddress.roots,args,tensor,ForwardArguments.buffers,ForwardArguments.saved,
          SavedContext.buffers,detach,unitShape,Shape.BK,Shape.BV,Shape.NK,Shape.NV,Shape.BH,tileCount] <;> decide
    | solve
      | intro iv hiv ik hik h hh lk hlk lv hlv r hr
        have hiv0 : iv=0 := by simpa [unitShape,Shape.NV,Shape.BV,tileCount] using hiv
        have hik0 : ik=0 := by simpa [unitShape,Shape.NK,Shape.BK,tileCount] using hik
        have hh0 : h=0 := by simpa [unitShape,Shape.BH] using hh
        have hlk0 : lk=0 := by simpa [unitShape,Shape.BK] using hlk
        have hlv0 : lv=0 := by simpa [unitShape,Shape.BV] using hlv
        subst iv; subst ik; subst h; subst lk; subst lv
        have hr' : r≤1 := hr
        interval_cases r <;> norm_num [SourceAddress.formed,signed64,args,tensor,
          ForwardArguments.buffers,ForwardArguments.saved,SavedContext.buffers,detach,
          unitShape,Shape.BK,Shape.BV,Shape.NK,Shape.NV,Shape.BH,tileCount] <;> decide

theorem unit_scope (seed final : Bool) : ForwardScope unitShape (args seed final) 0 := by
  refine ⟨unit_admitted,?_,?_,?_,rfl,rfl,rfl,rfl,rfl,rfl,?_,?_,?_,?_,?_,?_⟩
  · exact ⟨rfl,rfl,rfl,rfl,rfl⟩
  · exact ⟨rfl,rfl,rfl,rfl,rfl⟩
  · exact ⟨rfl,rfl,rfl,rfl,rfl⟩
  · cases seed <;> simp [args,tensor,TensorObject.RawFP32,unitShape,contiguous4]
  · cases seed
    · simpa [args] using unit_bounds .forward Bool.false final
    · simpa [args] using unit_bounds .forward Bool.true final
  all_goals cases seed <;> cases final <;> norm_num [args,tensor,unitShape,Shape.BH,
    ForwardArguments.buffers,ForwardArguments.saved,SavedContext.buffers,detach,
    InputBuffers.qAddress,InputBuffers.kAddress,InputBuffers.vAddress,InputBuffers.seedAddress,stateOffset]

/-- Nonempty executable admissions, including nonzero terminal cotangent
metadata and arbitrary zero strides on read-only cotangents. -/
theorem unit_forward_guard (seed final : Bool) :
    Admission.forwardCheck unitShape (args seed final) 0 (fun _=>2^40)=Bool.true := by
  cases seed <;> cases final <;> decide

def expandedCotangent (id : Nat) : TensorObject :=
  {tensor id with view := {(tensor id).view with strides := fun _=>0}}

theorem unit_backward_guard : Admission.backwardCheck unitShape (args Bool.true Bool.true).saved
    (expandedCotangent 5) (some (expandedCotangent 6)) 0 (fun _=>2^40) Bool.false=Bool.true := by
  decide

def unit_call (seed final : Bool) (before : Memory α) : ForwardCall α unitShape {n | n<10} where
  args := args seed final
  currentDevice := 0
  scope := Admission.checked_forward _ _ _ _ (unit_forward_guard seed final)
  allocation := {
    storage := fun i => 10+i.val
    distinct := by intro i j he; apply Fin.ext; change 10+i.val=10+j.val at he; omega
    fresh := by intro i; change ¬10+i.val<10; omega }
  inputs := by
    cases seed <;> cases final <;> simp [args,tensor,ForwardArguments.buffers,ForwardArguments.saved,
      SavedContext.buffers,detach,InputBuffers.storageIds,Set.mem_setOf_eq]
  before := before

#specNonCircular mathState avoiding [state,sourceForward,sourceBackward,forwardRun,backwardRun]
#specNonCircular mathAdjoint avoiding [reverseCarry,adjoint,sourceForward,sourceBackward,forwardRun,backwardRun]
#specNonCircular mathOutput avoiding [resultO,partialO,sourceForward,forwardRun]
#specNonCircular mathF avoiding [state,sourceForward,forwardRun]
#specNonCircular mathDQ avoiding [resultQ,partialQ,sourceBackward,backwardRun]
#specNonCircular mathDK avoiding [resultK,partialK,sourceBackward,backwardRun]
#specNonCircular mathDV avoiding [resultV,partialV,sourceBackward,backwardRun]
#auditStmt correct
#auditStmt reference_fp
#auditStmt fp_equiv
#print_spec correct
#print_spec reference_fp
#print_spec fp_equiv
#axiomsClean unit_admitted
#axiomsClean unit_bounds
#axiomsClean unit_scope
#axiomsClean unit_call
#axiomsClean unit_forward_guard
#axiomsClean unit_backward_guard
#axiomsClean kernel_forward_real
#axiomsClean kernel_backward_real
#axiomsClean kernel_forward_native
#axiomsClean kernel_backward_native
#axiomsClean TensorBridge.checked_quoted_returned
#axiomsClean TensorBridge.checked_quoted_fp_returned
#axiomsClean correct
#axiomsClean reference_fp
#axiomsClean fp_equiv
#axiomsClean finalOnlyWitness_nonzero
#axiomsClean mathAdjoint_final_contribution
#axiomsClean mathDK_final_contribution
#axiomsClean mathDV_final_contribution
#axiomsClean state_real
#axiomsClean adjoint_real
#axiomsClean resultO_real
#axiomsClean resultQ_real
#axiomsClean resultK_real
#axiomsClean resultV_real
#axiomsClean sourceCell_partial
#axiomsClean sourceForward_normalized
#axiomsClean sourceBackward_normalized
#axiomsClean programForwardCarry_at
#axiomsClean programReverseCarry_at
#axiomsClean output_owner
#axiomsClean gradient_owner
#axiomsClean final_owner
#axiomsClean output_writer_unique
#axiomsClean gradient_writer_unique
#axiomsClean final_writer_unique
#axiomsClean forward_output
#axiomsClean forward_state
#axiomsClean backward_dq
#axiomsClean backward_dk
#axiomsClean backward_dv
#axiomsClean SourceAddress.exact
#axiomsClean source_division_exact
#axiomsClean source_remainder_exact
#axiomsClean reverse_time
#axiomsClean dead_reverse_time
#axiomsClean source_partial_pointer
#axiomsClean dense_source_address
#axiomsClean Code.native_refines
#axiomsClean Numeric.native_forward
#axiomsClean Numeric.native_backward
#axiomsClean nativeArithmetic_eq
#axiomsClean HostSumPlan.execute_tree
#axiomsClean CompilerPlans.dot1lanes1_covers
#axiomsClean CompilerPlans.dot16lanes2_covers
#axiomsClean CompilerPlans.dot16lanes16_covers
#axiomsClean CompilerPlans.dot32lanes1_covers
#axiomsClean CompilerPlans.dot32lanes32_covers
#axiomsClean raw_allocation_frame
#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean detach_storage
#axiomsClean saved_identity
#axiomsClean saved_values
#axiomsClean no_initial_gradient
#axiomsClean backward_returns_fresh
#axiomsClean Quoted.forward_full_values
#axiomsClean Quoted.backward_full_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Reader
